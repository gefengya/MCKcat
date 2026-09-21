from torch import nn
import sys
import re
import torch
from eitlem_utils import Tester, Trainer, get_pair_info
from KCM import EitlemKcatPredictor  # Keep only the kcat model import
# Comment out Km/KKm-related imports (not needed)
# from KMP import EitlemKmPredictor
# from ensemble import ensemble
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import EitlemDataSet, EitlemDataLoader
import os
import shutil
import argparse


def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device):
    """Keep the original kcat training logic, but later calls fix Type='KCAT' and kkmPath=None (no transfer)"""
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"

    if os.path.exists(f'../Results/{Type}/{train_info}'):
        return None

    # Fixed number of epochs: since there is no iteration, training from scratch is set to 100 epochs
    Epoch = 100  # Fix directly to 100 epochs; no need to adjust according to the iteration count

    file_model = f'../Results/{Type}/{train_info}/Weight/'

    # Keep only the kcat model initialization logic (Km-related code removed)
    if Type == 'KCAT':
        if molType == 'MACCSKeys':
            model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
        else:
            model = EitlemKcatPredictor(1024, 512, 1280, 10, 0.5, 10)
    # Comment out Km model initialization (not needed)
    # else:
    #     if molType == 'MACCSKeys':
    #         model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
    #     else:
    #         model = EitlemKmPredictor(1024, 512, 1280, 10, 0.5, 10)

    # No transfer (kkmPath is fixed to None): no need to load KKm weights, skip the weight transfer logic
    # (The original transfer-related code can be kept, but it never actually runs because kkmPath=None)
    if kkmPath is not None:
        trained_weights = torch.load(kkmPath)
        weights = model.state_dict()
        pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in weights}
        weights.update(pretrained_para)
        model.load_state_dict(weights)

    if not os.path.exists(file_model):
        os.makedirs(file_model)
    file_model += 'Eitlem_'
    """Train setting."""
    # Load only the kcat dataset (Type='KCAT')
    train_pair_info, test_pair_info = get_pair_info("../Data/", Type, False)
    train_set = EitlemDataSet(train_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    test_set = EitlemDataSet(test_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
    model = model.to(device)

    # No transfer (kkmPath=None): use the optimizer configuration for training from scratch
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)

    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)

    print("start to train kcat model...")
    writer = SummaryWriter(f'../Results/{Type}/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration}iter epoch {epoch} train:")
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(model, valid_loader, len(test_pair_info), desc=f"{Iteration}iter epoch {epoch} valid:")
        scheduler.step()
        # Record kcat training logs
        writer.add_scalars("loss",{'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
        writer.add_scalars("RMSE",{'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
        writer.add_scalars("MAE",{'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
        writer.add_scalars("R2",{'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
        # Save kcat model weights
        tester.save_model(model, file_model+f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}')
    pass


# Comment out the KKMTrainer function (no need to train KKm)
# def KKMTrainer(kcatPath, kmPath, TrainType, Iteration, log10, molType, device):
#     ... (all the original code is commented out or removed)


# Simplified getPath function: get only kcat model weights (no Km/KKm needed)
def getPath(Type, TrainType, Iteration):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    fileList = os.listdir(file_model)
    return os.path.join(file_model, fileList[0])


# Replace the original TransferLearing function: train kcat only once, with no iteration
def train_only_kcat(TrainType, log10=False, molType='MACCSKeys', device=None):
    """Train the kcat model only once, with no iteration and no transfer"""
    Iteration = 1  # Fix the iteration count to 1 (a single training run)
    Type = 'KCAT'   # Fix the training type to kcat
    kkmPath = None  # No transfer; do not load KKm weights

    # Directly call kcat training (fixed parameters: no transfer, kcat type, 1 iteration)
    kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device)

    # After training finishes, print the path where the weights are saved
    kcat_weight_path = getPath(Type, TrainType, Iteration)
    print(f"kcat model training finished! Weight saved to: {kcat_weight_path}")


def parse_args():
    """Simplified command-line arguments: remove the iteration count (fixed to 1), keep key arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--TrainType', type=str, required=True, help="Training task name (custom, e.g. only_kcat_train)")
    parser.add_argument('-l', '--log10', type=bool, required=False, default=True, help="Whether to apply log10 transform to kcat (default True)")
    parser.add_argument('-m', '--molType', type=str, required=False, default='MACCSKeys', help="Substrate molecular fingerprint type (MACCSKeys/ECFP/RDKIT, default MACCSKeys)")
    parser.add_argument('-d', '--device', type=int, required=True, help="GPU index (e.g. 0; set -1 if no GPU is available)")
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()

    # Initialize the device (GPU/CPU)
    if torch.cuda.is_available() and args.device != -1:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    print(f"use device {device}")

    # Call the simplified function: train kcat only
    train_only_kcat(
        TrainType=args.TrainType,
        log10=args.log10,
        molType=args.molType,
        device=device
    )
