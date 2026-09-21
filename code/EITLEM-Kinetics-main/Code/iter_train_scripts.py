#from torch import nn
#import sys
#import re
#import torch
#import pandas as pd  # Added: used to process and save prediction results
#import numpy as np   # Ensure it is imported (used for numerical computation)
## -------------------------- Added: import the required functions from sklearn.metrics --------------------------
#from sklearn.metrics import mean_squared_error, r2_score
#from eitlem_utils import Tester, Trainer, get_pair_info
#from KCM import EitlemKcatPredictor
#from KMP import EitlemKmPredictor
#from ensemble import ensemble
#from torch.utils.tensorboard import SummaryWriter
#from tqdm import tqdm
#from dataset import EitlemDataSet, EitlemDataLoader
#import os
#import shutil
#import argparse
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all logs, 1=suppress INFO, 2=suppress WARNING, 3=suppress ERROR and below
#import tensorflow as tf
#
## -------------------------- 1. Rewrite the ModifiedTester class (fixed function dependencies) --------------------------
#class ModifiedTester(object):
#    def __init__(self, device, loss_fn, log10=False):
#        self.device = device
#        self.loss_fn = loss_fn
#        self.log10 = log10
#        self.saved_file_path = None
#
#    def test(self, model, loader, N, desc):
#        testY = []  # Store the ground-truth values of the test set
#        testPredict = []  # Store the predicted values of the test set
#        loss_total = 0
#        model.eval()
#        with torch.no_grad():
#            for data in tqdm(loader, desc=desc, leave=False):
#                # Model prediction
#                pre_value = model(data.to(self.device))
#                # Accumulate the loss
#                if self.loss_fn is not None:
#                    loss_total += self.loss_fn(pre_value, data.value).item()
#                # Collect ground-truth and predicted values (convert to CPU lists to avoid GPU memory usage)
#                testY.extend(data.value.cpu().tolist())
#                testPredict.extend(pre_value.cpu().tolist())
#
#        # Label de-normalization (consistent with the original code logic, ensuring values are restored to the original scale)
#        if not self.log10:
#            testY = np.log10(np.power(2, testY))
#            testPredict = np.log10(np.power(2, testPredict))
#        else:
#            testY = np.array(testY)
#            testPredict = np.array(testPredict)
#
#        # Compute evaluation metrics (mean_squared_error and r2_score are now imported and can be called normally)
#        MAE = np.abs(testY - testPredict).sum() / N
#        rmse = np.sqrt(mean_squared_error(testY, testPredict))
#        r2 = r2_score(testY, testPredict)
#
#        # Added: return the ground-truth and predicted values (for later saving)
#        return MAE, rmse, r2, loss_total/N, testY, testPredict
#
#    # Keep the original save_model method
#    def save_model(self, model, file_path):
#        torch.save(model.state_dict(), file_path)
#        if self.saved_file_path is not None:
#            os.remove(self.saved_file_path)
#        self.saved_file_path = file_path
#
#
## -------------------------- 2. Other functions (kineticsTrainer, getPath, etc.) remain unchanged --------------------------
#def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device):
#    print(2)
#    Epoch = 100
#    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
#    Type = "KCAT"  # Note: hardcoded to KCAT in the original code; remove this line to support KM
#
#    # 1. Define the save paths for the model and prediction results
#    file_model_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/2/Results/KCAT/Weight/'
#    file_result_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/2/Results/KCAT/Predictions/'  # Prediction results directory
#    kkmPath = None  # Hardcoded in the original code; remove this line to load a pre-trained model
#
#    # Create the directories (model directory + prediction results directory)
#    if not os.path.exists(file_model_dir):
#        os.makedirs(file_model_dir)
#    if not os.path.exists(file_result_dir):
#        os.makedirs(file_result_dir)  # Ensure the prediction results directory exists
#    file_model_prefix = file_model_dir + 'Eitlem_'
#
#    # 2. Model initialization (keep the original logic)
#    if kkmPath is not None:
#        print(Type)
#        trained_weights = torch.load(kkmPath)
#        if Type == 'KCAT':
#            model = EitlemKcatPredictor(167 if molType == 'MACCSKeys' else 1024, 512, 1280, 10, 0.5, 10)
#            pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in model.state_dict()}
#        else:
#            model = EitlemKmPredictor(167 if molType == 'MACCSKeys' else 1024, 512, 1280, 10, 0.5, 10)
#            pretrained_para = {k[3:]: v for k, v in trained_weights.items() if 'km' in k and k[3:] in model.state_dict()}
#        model.load_state_dict({**model.state_dict(), **pretrained_para})
#    else:
#        print(Type)
#        if Type == 'KCAT':
#            if molType == 'MACCSKeys':
#                model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
#                print(2)
#            else:
#                model = EitlemKcatPredictor(1024, 512, 1280, 10, 0.5, 10)
#        else:
#            if molType == 'MACCSKeys':
#                model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
#            else:
#                model = EitlemKmPredictor(1024, 512, 1280, 10, 0.5, 10)
#    model = model.to(device)
#
#    # 3. Data loading (keep the original logic)
#    train_pair_info, test_pair_info = get_pair_info("/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2", Type, False)
#    train_set = EitlemDataSet(train_pair_info, f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2/esm2_t33_650M_UR50D2/', f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2/index_smiles', 1024, 4, log10, molType)
#    test_set = EitlemDataSet(test_pair_info, f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2/esm2_t33_650M_UR50D2/', f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/Feature/2/index_smiles', 1024, 4, log10, molType)
#    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
#    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
#
#    # 4. Optimizer and loss function (keep the original logic)
#    if kkmPath is not None:
#        out_param = list(map(id, model.out.parameters()))
#        rest_param = filter(lambda x: id(x) not in out_param, model.parameters())
#        optimizer = torch.optim.AdamW([
#            {'params': rest_param, 'lr': 1e-4},
#            {'params': model.out.parameters(), 'lr': 1e-3},
#        ])
#        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
#    else:
#        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
#    loss_fn = nn.MSELoss()
#
#    # 5. Initialize the modified Tester (replace the original Tester; this is critical!)
#    tester = ModifiedTester(device, loss_fn, log10=log10)  # Use the new Tester that returns ground-truth/predicted values
#    trainer = Trainer(device, loss_fn, log10=log10)  # Keep the original Trainer
#
#    print("start to train...")
#    writer = SummaryWriter(f'../Results/{Type}/{train_info}/logs/')
#
#    # 6. Training loop: added saving of prediction results
#    for epoch in range(1, Epoch + 1):
#        # Training phase (keep the original logic)
#        train_MAE, train_rmse, train_r2, loss_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration}iter epoch {epoch} train:")
#
#        # Test phase: additionally receive the ground-truth values (testY) and predicted values (testPredict)
#        MAE_dev, RMSE_dev, R2_dev, loss_dev, testY, testPredict = tester.test(
#            model, valid_loader, len(test_pair_info), desc=f"{Iteration}iter epoch {epoch} valid:"
#        )
#
#        # -------------------------- Added: save prediction results to CSV --------------------------
#        # Build the result DataFrame (including sample index, ground-truth value, predicted value, and the evaluation metrics of the current epoch)
#        result_df = pd.DataFrame({
#            "sample_index": range(len(testY)),  # Sample index (corresponds to the test set order)
#            "true_value": testY,                # De-normalized ground-truth value
#            "predicted_value": testPredict,     # De-normalized predicted value
#            "epoch": epoch,                     # Training epoch
#            "MAE": MAE_dev,                     # Test set MAE for this epoch
#            "RMSE": RMSE_dev,                   # Test set RMSE for this epoch
#            "R2": R2_dev                        # Test set R2 for this epoch
#        })
#        # Save path (includes epoch and molType for easy distinction)
#        result_save_path = f"{file_result_dir}epoch_{epoch}_{molType}_test_predictions.csv"
#        result_df.to_csv(result_save_path, index=False, encoding="utf-8")
#        print(f"Epoch {epoch} test set prediction results saved to: {result_save_path}")
#
#        # 7. Original logic: learning rate update, log writing, model saving
#        scheduler.step()
#        writer.add_scalars("loss", {'train_loss': loss_train, 'dev_loss': loss_dev}, epoch)
#        writer.add_scalars("RMSE", {'train_RMSE': train_rmse, 'dev_RMSE': RMSE_dev}, epoch)
#        writer.add_scalars("MAE", {'train_MAE': train_MAE, 'dev_MAE': MAE_dev}, epoch)
#        writer.add_scalars("R2", {'train_R2': train_r2, 'dev_R2': R2_dev}, epoch)
#
#        saved_path = file_model_prefix + f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}'
#        tester.save_model(model, saved_path)
#        print(f"Epoch {epoch} model saved to: {saved_path}\n")
#
#
## -------------------------- 3. Other functions (getPath, TransferLearing, parse_args, main) keep the original logic --------------------------
#def getPath(Type, TrainType, Iteration):
#    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
#    file_model = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/turnup/cold_enzyme/2/Results/{Type}/{train_info}/Weight/'
#    fileList = os.listdir(file_model)
#    return os.path.join(file_model, fileList[0])
#
#def TransferLearing(Iterations, TrainType, log10, molType='MACCSKeys', device=None):
#    for iteration in range(1, Iterations + 1):
#        print(iteration)
#        if iteration == 1:
#            kineticsTrainer(None, TrainType, 'KCAT', iteration, log10, molType, device)
#            # kineticsTrainer(None, TrainType, 'KM', iteration, log10, molType, device)  # Uncomment to support KM
#        else:
#            kkmPath = getPath('KKM', TrainType, iteration-1)
#            kineticsTrainer(kkmPath, TrainType, 'KCAT', iteration, log10, molType, device)
#            # kineticsTrainer(kkmPath, TrainType, 'KM', iteration, log10, molType, device)  # Uncomment to support KM
#
#        kcatPath = getPath('KCAT', TrainType, iteration)
#        # kmPath = getPath('KM', TrainType, iteration)  # Uncomment to support KM
#        # KKMTrainer(kcatPath, kmPath, TrainType, iteration, log10, molType, device)  # Uncomment to train KKM
#
#def parse_args():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-i', '--Iteration', type=int, required=True)
#    parser.add_argument('-t', '--TrainType', type=str, required=True)
#    parser.add_argument('-l', '--log10', type=bool, required=False, default=True)
#    parser.add_argument('-m', '--molType', type=str, required=False, default='MACCSKeys')
#    parser.add_argument('-d', '--device', type=int, required=True)
#    return parser.parse_args()
#
#if __name__ == '__main__':
#    torch.backends.cudnn.allow_tf32 = True
#    torch.backends.cuda.matmul.allow_tf32 = True
#    args = parse_args()
#    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
#    print(f"use device {device}")
#    TransferLearing(args.Iteration, args.TrainType, args.log10, args.molType, device)

from torch import nn
import sys
import re
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from eitlem_utils import Tester, Trainer, get_pair_info
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
from ensemble import ensemble
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import EitlemDataSet, EitlemDataLoader
import os
import shutil
import argparse

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# -------------------------- 1. Rewrite the ModifiedTester class --------------------------
class ModifiedTester(object):
    def __init__(self, device, loss_fn, log10=False):
        self.device = device
        self.loss_fn = loss_fn
        self.log10 = log10
        self.saved_file_path = None

    def test(self, model, loader, N, desc):
        testY = []
        testPredict = []
        loss_total = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(loader, desc=desc, leave=False):
                pre_value = model(data.to(self.device))
                if self.loss_fn is not None:
                    loss_total += self.loss_fn(pre_value, data.value).item()
                testY.extend(data.value.cpu().tolist())
                testPredict.extend(pre_value.cpu().tolist())

        # Label de-normalization
        if not self.log10:
            testY = np.log10(np.power(2, testY))
            testPredict = np.log10(np.power(2, testPredict))
        else:
            testY = np.array(testY)
            testPredict = np.array(testPredict)

        # Compute evaluation metrics
        MAE = np.abs(testY - testPredict).sum() / N
        rmse = np.sqrt(mean_squared_error(testY, testPredict))
        r2 = r2_score(testY, testPredict)

        return MAE, rmse, r2, loss_total / N, testY, testPredict

    def save_model(self, model, file_path):
        torch.save(model.state_dict(), file_path)
        if self.saved_file_path is not None and os.path.exists(self.saved_file_path):
            os.remove(self.saved_file_path)
        self.saved_file_path = file_path


# -------------------------- 2. kineticsTrainer (added dynamic path parameters) --------------------------
def kineticsTrainer(
    kkmPath, TrainType, Type, Iteration, log10, molType, device,
    enzyme_type, num_id  # Added: loop parameters (e.g. cold_enzyme, 2)
):
    print(f"\n=== Start training: enzyme_type={enzyme_type}, num_id={num_id}, iteration={Iteration} ===")
    Epoch = 100
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    Type = "KCAT"  # Fix to KCAT; remove this line if KM is needed

    # -------------------------- Dynamic path generation (core modification) --------------------------
    # Base path (contains the loop parameters enzyme_type and num_id)
    base_data_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/turnup/{enzyme_type}/{num_id}/'
    # Model save path
    file_model_dir = f'{base_data_dir}Results/Weight/'
    # Prediction results save path
    file_result_dir = f'{base_data_dir}Results/Predictions/'
    # Log path
    log_dir = f'{base_data_dir}Results/KCAT/logs/'
    # Data feature path
    feature_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/turnup/{enzyme_type}/Feature/{num_id}/'
    esm_dir = f'{feature_dir}esm2_t33_650M_UR50D/'
    index_smiles_path = f'{feature_dir}index_smiles'

    # Create the directories
    for dir_path in [file_model_dir, file_result_dir, log_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    file_model_prefix = file_model_dir + 'Eitlem_'

    # Model initialization
    if kkmPath is not None and os.path.exists(kkmPath):
        print(f"Loading pre-trained model: {kkmPath}")
        trained_weights = torch.load(kkmPath)
        model = EitlemKcatPredictor(167 if molType == 'MACCSKeys' else 1024, 512, 1280, 10, 0.5, 10)
        pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in model.state_dict()}
        model.load_state_dict({**model.state_dict(), **pretrained_para})
    else:
        print("Initializing a new model")
        model = EitlemKcatPredictor(167 if molType == 'MACCSKeys' else 1024, 512, 1280, 10, 0.5, 10)
    model = model.to(device)

    # Data loading (paths use the dynamic parameters)
    train_pair_info, test_pair_info = get_pair_info(feature_dir, Type, False)
    train_set = EitlemDataSet(
        train_pair_info, esm_dir, index_smiles_path, 1024, 4, log10, molType
    )
    test_set = EitlemDataSet(
        test_pair_info, esm_dir, index_smiles_path, 1024, 4, log10, molType
    )
    train_loader = EitlemDataLoader(
        data=train_set, batch_size=200, shuffle=True, drop_last=False,
        num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True
    )
    valid_loader = EitlemDataLoader(
        data=test_set, batch_size=200, drop_last=False,
        num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True
    )

    # Optimizer and loss function
    if kkmPath is not None:
        out_param = list(map(id, model.out.parameters()))
        rest_param = filter(lambda x: id(x) not in out_param, model.parameters())
        optimizer = torch.optim.AdamW([
            {'params': rest_param, 'lr': 1e-4},
            {'params': model.out.parameters(), 'lr': 1e-3},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
    loss_fn = nn.MSELoss()

    # Initialize utility classes
    tester = ModifiedTester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    writer = SummaryWriter(log_dir)

    # Added: track the highest test set R2 and the corresponding model path
    best_r2 = -float('inf')
    best_model_path = None

    # Training loop (for each epoch)
    for epoch in range(1, Epoch + 1):
        # Train
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(
            model, train_loader, optimizer, len(train_pair_info),
            f"enzyme={enzyme_type}, id={num_id}, iteration {Iteration}-Epoch{epoch} train"
        )
        # Test
        MAE_dev, RMSE_dev, R2_dev, loss_dev, testY, testPredict = tester.test(
            model, valid_loader, len(test_pair_info),
            f"enzyme={enzyme_type}, id={num_id}, iteration {Iteration}-Epoch{epoch} test"
        )
        scheduler.step()

        # Save prediction results
        result_df = pd.DataFrame({
            "sample_index": range(len(testY)),
            "true_value": testY.round(4),
            "predicted_value": testPredict.round(4),
            "epoch": epoch,
            "MAE": round(MAE_dev, 4),
            "RMSE": round(RMSE_dev, 4),
            "R2": round(R2_dev, 4)
        })
        result_save_path = f"{file_result_dir}epoch_{epoch}_{molType}_test_predictions.csv"
        result_df.to_csv(result_save_path, index=False, encoding="utf-8")

        # Write logs
        writer.add_scalars("loss", {'train_loss': loss_train, 'dev_loss': loss_dev}, epoch)
        writer.add_scalars("RMSE", {'train_RMSE': train_rmse, 'dev_RMSE': RMSE_dev}, epoch)
        writer.add_scalars("MAE", {'train_MAE': train_MAE, 'dev_MAE': MAE_dev}, epoch)
        writer.add_scalars("R2", {'train_R2': train_r2, 'dev_R2': R2_dev}, epoch)

        # Added: compare the current R2 with the highest R2; if higher, save the current model
        if R2_dev > best_r2:
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_r2 = R2_dev
            best_model_path = file_model_prefix + f'{molType}_best_devR2_{R2_dev:.4f}_epoch{epoch}.pth'
            tester.save_model(model, best_model_path)

        print(f"Epoch {epoch} | test R2: {R2_dev:.4f} | results saved to: {result_save_path}")

    writer.close()
    print(f"=== enzyme_type={enzyme_type}, num_id={num_id}, iteration={Iteration} training finished ===\n")


# -------------------------- 3. getPath (adapted for dynamic paths) --------------------------
def getPath(Type, TrainType, Iteration, enzyme_type, num_id):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    file_model_dir = f'/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/turnup/{enzyme_type}/{num_id}/Results/Weight/'
    if not os.path.exists(file_model_dir):
        print(f"Warning: path does not exist {file_model_dir}")
        return None
    fileList = os.listdir(file_model_dir)
    if not fileList:
        print(f"Warning: directory is empty {file_model_dir}")
        return None
    return os.path.join(file_model_dir, fileList[0])


# -------------------------- 4. Outer loop control (added enzyme_type and num_id loops) --------------------------
def run_multi_enzyme(
    Iterations, TrainType, log10, molType='MACCSKeys', device=None,
    enzyme_types=["cold_enzyme"],  # List of enzyme types to loop over
    num_ids=[1, 2, 3, 4, 5]  # List of numeric IDs to loop over
):
    # First-level loop: iterate over enzyme types (e.g. cold_enzyme, hot_enzyme, etc.)
    for enzyme_type in enzyme_types:
        # Second-level loop: iterate over numeric IDs (e.g. 1, 2, 3, etc.)
        for num_id in num_ids:
            print(f"\n=====================================")
            print(f"Start processing: enzyme_type={enzyme_type}, num_id={num_id}")
            print(f"=====================================\n")

            # Third-level loop: iterate over the number of iterations (original TransferLearing logic)
            for iteration in range(1, Iterations + 1):
                print(f"---------- iteration {iteration}/{Iterations} ----------")
                if iteration == 1:
                    # First iteration: no pre-trained model
                    kineticsTrainer(
                        kkmPath=None,
                        TrainType=TrainType,
                        Type='KCAT',
                        Iteration=iteration,
                        log10=log10,
                        molType=molType,
                        device=device,
                        enzyme_type=enzyme_type,  # Pass the loop parameter
                        num_id=num_id  # Pass the loop parameter
                    )
                else:
                    # Subsequent iterations: load the KKM model from the previous iteration
                    kkmPath = getPath(
                        Type='KKM',
                        TrainType=TrainType,
                        Iteration=iteration - 1,
                        enzyme_type=enzyme_type,  # Pass the loop parameter
                        num_id=num_id  # Pass the loop parameter
                    )
                    if kkmPath is None:
                        print(f"Iteration {iteration}: previous model not found, skipping")
                        continue
                    kineticsTrainer(
                        kkmPath=kkmPath,
                        TrainType=TrainType,
                        Type='KCAT',
                        Iteration=iteration,
                        log10=log10,
                        molType=molType,
                        device=device,
                        enzyme_type=enzyme_type,
                        num_id=num_id
                    )

                # Get the model path of the current iteration (uncomment if KKM training is needed)
                # kcatPath = getPath('KCAT', TrainType, iteration, enzyme_type, num_id)


# -------------------------- 5. Command-line arguments and main function --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--Iteration', type=int, required=True, help="Total number of iterations")
    parser.add_argument('-t', '--TrainType', type=str, required=True, help="Training type identifier")
    parser.add_argument('-l', '--log10', type=bool, default=True, help="Whether to apply log10 to the labels")
    parser.add_argument('-m', '--molType', type=str, default='MACCSKeys', help="Molecular feature type")
    parser.add_argument('-d', '--device', type=int, required=True, help="GPU device index")
    # Added: loop parameters (cold_enzyme and 2 by default, can be modified from the command line)
    parser.add_argument('-e', '--enzyme_types', type=str, nargs='+', default=["cold_enzyme"], help="List of enzyme types to loop over, e.g. cold_enzyme hot_enzyme")
    parser.add_argument('-n', '--num_ids', type=int, nargs='+', default=[2], help="List of numeric IDs to loop over, e.g. 1 2 3")
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Start the multi-level loop training
    run_multi_enzyme(
        Iterations=args.Iteration,
        TrainType=args.TrainType,
        log10=args.log10,
        molType=args.molType,
        device=device,
        enzyme_types=args.enzyme_types,  # Pass the list of enzyme types
        num_ids=args.num_ids  # Pass the list of numeric IDs
    )
