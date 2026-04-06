import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import torch.nn as nn
import os
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import Chem
import math
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple
from KCM import EitlemKcatPredictor

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

import rdkit
from rdkit import Chem

if rdkit.__version__ >= '2023.03':
    def generate_rdkit_fp(mol, fpSize=1024):
        return Chem.RDKFingerprint(mol, fpSize=fpSize)
else:
    from rdkit.Chem import AllChem


    def generate_rdkit_fp(mol, fpSize=1024):
        gen = AllChem.GetRDKitFPGenerator(fpSize=fpSize)
        return gen.GetFingerprint(mol)


def generate_mol_feature(mol, mol_type: str = 'ECFP', radius: int = 4, nbits: int = 1024) -> torch.Tensor:
    """Generate molecular feature fingerprints"""
    if mol_type == "ECFP":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits).ToList()
    elif mol_type == "MACCSKeys":
        fp = MACCSkeys.GenMACCSKeys(mol).ToList()
    elif mol_type == "RDKIT":
        fp = generate_rdkit_fp(mol, nbits).ToList()
    else:
        raise ValueError(f"Unsupported mol_type: {mol_type}")
    return torch.FloatTensor(fp).unsqueeze(0)


class KCATDataset(Dataset):
    """KCAT prediction dataset, supports protein features in NumPy array format"""

    def __init__(self, data_df, protein_feature_col='esm2_features',
                 smile_col='reactant_smiles', kcat_col='log10kcat_max',
                 mol_type='MACCSKeys', radius=4, nbits=1024, log10=True):
        super(KCATDataset, self).__init__()
        self.df = data_df
        self.protein_feature_col = protein_feature_col
        self.smile_col = smile_col
        self.kcat_col = kcat_col
        self.mol_type = mol_type
        self.radius = radius
        self.nbits = nbits
        self.log10 = log10

    def __getitem__(self, idx):
        pro_emb_np = self.df.iloc[idx][self.protein_feature_col]
        pro_emb = torch.FloatTensor(pro_emb_np)

        smile = self.df.iloc[idx][self.smile_col]
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {idx}: {smile}")
        mol_feature = generate_mol_feature(mol, self.mol_type, self.radius, self.nbits)

        kcat = self.df.iloc[idx][self.kcat_col]
        if not self.log10:
            kcat = math.log2(kcat)

        return Data(x=mol_feature, pro_emb=pro_emb, y=torch.FloatTensor([kcat]))

    def collate_fn(self, batch):
        return Batch.from_data_list(batch, follow_batch=['pro_emb'])

    def __len__(self):
        return len(self.df)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
    """Calculate evaluation metrics: MAE, RMSE, R2"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    ss_total = np.sum((target - np.mean(target)) ** 2)
    ss_residual = np.sum((pred - target) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0

    return mae, rmse, r2


class Trainer:
    """Model trainer"""

    def __init__(self, device: torch.device, loss_fn: nn.Module, log_interval: int = 100):
        self.device = device
        self.loss_fn = loss_fn
        self.log_interval = log_interval

    def train(self, model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int,
              scheduler=None) -> Tuple[float, float, float, float]:
        model.train()
        total_loss = 0.0
        pred_list = []
        target_list = []

        with tqdm(enumerate(data_loader), total=len(data_loader)) as pbar:
            for batch_idx, batch in pbar:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                pred = model(batch)
                loss = self.loss_fn(pred, batch.y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred_list.append(pred.detach())
                target_list.append(batch.y.detach())

                if (batch_idx + 1) % self.log_interval == 0:
                    pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if scheduler:
            scheduler.step()

        pred = torch.cat(pred_list)
        target = torch.cat(target_list)
        mae, rmse, r2 = calculate_metrics(pred, target)
        avg_loss = total_loss / len(data_loader)

        return mae, rmse, r2, avg_loss


class Tester:
    """Model tester"""

    def __init__(self, device: torch.device, loss_fn: nn.Module):
        self.device = device
        self.loss_fn = loss_fn

    def test(self, model: nn.Module, data_loader: DataLoader) -> Tuple[float, float, float, float]:
        model.eval()
        total_loss = 0.0
        pred_list = []
        target_list = []

        with torch.no_grad(), tqdm(enumerate(data_loader), total=len(data_loader)) as pbar:
            for batch_idx, batch in pbar:
                batch = batch.to(self.device)

                pred = model(batch)
                loss = self.loss_fn(pred, batch.y)

                total_loss += loss.item()
                pred_list.append(pred)
                target_list.append(batch.y)

                pbar.set_description(f"Test Loss: {loss.item():.4f}")

        pred = torch.cat(pred_list)
        target = torch.cat(target_list)
        mae, rmse, r2 = calculate_metrics(pred, target)
        avg_loss = total_loss / len(data_loader)

        return mae, rmse, r2, avg_loss

    def save_model(self, model: nn.Module, path: str, best: bool = False) -> None:
        save_path = path + ("_best" if best else "") + ".pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train KCAT prediction model with pkl files")
    parser.add_argument("--train1_pkl", type=str, required=True, help="Path to first training pkl file")
    parser.add_argument("--train2_pkl", type=str, required=True, help="Path to second training pkl file")
    parser.add_argument("--test_pkl", type=str, required=True, help="Path to testing pkl file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for models and logs")
    parser.add_argument("--mol_type", type=str, default="MACCSKeys", choices=["ECFP", "MACCSKeys", "RDKIT"],
                        help="Molecular feature type")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--log10", type=bool, default=True, help="Whether KCAT is log10 transformed")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading first training data from {args.train1_pkl}")
    train_df1 = pd.read_pickle(args.train1_pkl)

    print(f"Reading second training data from {args.train2_pkl}")
    train_df2 = pd.read_pickle(args.train2_pkl)

    train_df = pd.concat([train_df1, train_df2], ignore_index=True)

    print(f"Reading test data from {args.test_pkl}")
    test_df = pd.read_pickle(args.test_pkl)
    print(f"Combined train size: {len(train_df)}, Test size: {len(test_df)}")

    mol_feature_dim = 167 if args.mol_type == "MACCSKeys" else 1024

    model = EitlemKcatPredictor(
        mol_in_dim=mol_feature_dim,
        hidden_dim=512,
        protein_dim=1280,
        layer=10,
        dropout=0.5,
        att_layer=10
    ).to(device)
    print(f"KCAT model created with mol feature dim: {mol_feature_dim}, dropout: 0.5")

    train_dataset = KCATDataset(
        train_df,
        protein_feature_col='esm2_features',
        smile_col='reactant_smiles',
        kcat_col='log10kcat_max',
        mol_type=args.mol_type,
        log10=args.log10
    )
    test_dataset = KCATDataset(
        test_df,
        protein_feature_col='esm2_features',
        smile_col='reactant_smiles',
        kcat_col='log10kcat_max',
        mol_type=args.mol_type,
        log10=args.log10
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 80],
        gamma=0.9
    )

    loss_fn = nn.MSELoss()

    trainer = Trainer(device, loss_fn)
    tester = Tester(device, loss_fn)

    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_r2 = -float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_mae, train_rmse, train_r2, train_loss = trainer.train(model, train_loader, optimizer, epoch, scheduler)
        print(
            f"Epoch {epoch}/{args.epochs} - Train: Loss={train_loss:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)

        test_mae, test_rmse, test_r2, test_loss = tester.test(model, test_loader)
        print(
            f"Epoch {epoch}/{args.epochs} - Test: Loss={test_loss:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}")

        writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalars("RMSE", {"train": train_rmse, "test": test_rmse}, epoch)
        writer.add_scalars("MAE", {"train": train_mae, "test": test_mae}, epoch)
        writer.add_scalars("R2", {"train": train_r2, "test": test_r2}, epoch)

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_epoch = epoch
            model_path = os.path.join(args.output_dir, "kcat_model_best")
            tester.save_model(model, model_path, best=True)
            print(f"Best model saved at epoch {epoch} with test R2: {test_r2:.4f}")

    model_path = os.path.join(args.output_dir, "kcat_model_final")
    tester.save_model(model, model_path)

    print(f"Training completed. Best model found at epoch {best_epoch} with R2: {best_r2:.4f}")
    writer.close()


if __name__ == "__main__":
    main()