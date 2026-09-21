import sys
import torch
import pandas as pd
from tqdm import tqdm
from dataset import EitlemDataSet, EitlemDataLoader
from KMP import EitlemKmPredictor
from ensemble import ensemble
from KCM import EitlemKcatPredictor
from KKMP import EitlemKKmPredictor


def predict(Type, modelPath, pairInfo, embedingPath, smilesPath, log10, device, molType):
    Dataset = EitlemDataSet(pairInfo, embedingPath, smilesPath, 1024, 4, log10, molType)

    Loader = EitlemDataLoader(
        data=Dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False
    )

    if molType == 'MACCSKeys':
        molDim = 167
    else:
        molDim = 1024

    if Type == 'KCAT':
        model = EitlemKcatPredictor(molDim, 512, 1280, 10, 0.5, 10)
    elif Type == 'KM':
        model = EitlemKmPredictor(molDim, 512, 1280, 10, 0.5, 10)
    elif Type == 'KKM':
        model = ensemble(molDim, 512, 1280, 10, 0.5, 10)
    elif Type == 'KKMP':
        model = EitlemKKmPredictor(molDim, 512, 1280, 10, 0.5, 10)

    model.load_state_dict(torch.load(modelPath, map_location=device))
    model = model.to(device)
    model.eval()

    testY, testPredict = [], []
    with torch.no_grad():
        for data in tqdm(Loader):
            pre = model(data.to(device))
            testY.extend(data.value.cpu().tolist())
            testPredict.extend(pre.cpu().tolist())

    return testY, testPredict


if __name__ == '__main__':
    task_type = "KCAT"
    model_weight = "/mnt/usb1/gfy/code/EITLEM-Kinetics-main/Results/KCAT/Transfer-only_kcat_train-KCAT-train-1/Weight/train/Eitlem_MACCSKeys_last_epoch"
    # Binary pair file, used by the dataset
    pair_info_path = "/mnt/usb1/gfy/code/data/Feature/DSM/HIS3/KCATTestPairInfo"
    # Change this to the path of the original csv file you processed at the beginning, e.g. test.csv
    raw_csv_path = "/mnt/usb1/gfy/code/data/HIS3.csv"

    embed_file = "/mnt/usb1/gfy/code/data/Feature/DSM/HIS3/esm2_t33_650M_UR50D"
    smiles_file = "/mnt/usb1/gfy/code/data/Feature/DSM/HIS3/index_smiles"
    log10_flag = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mol_feature = "MACCSKeys"

    true_v, pred_v = predict(
        Type=task_type,
        modelPath=model_weight,
        pairInfo=pair_info_path,  # pass the binary pair file
        embedingPath=embed_file,
        smilesPath=smiles_file,
        log10=log10_flag,
        device=device,
        molType=mol_feature
    )


# Safe function for reading the original csv
def read_csv_safe(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="gbk")


# Read the original text csv instead of the binary KCATMCKcat_N20PairInfo
df = read_csv_safe(raw_csv_path)
df["pred_log10[kcat]"] = pred_v
df["true_log10[kcat]"] = true_v
df.to_csv("HIS3——predict_result.csv", index=False)
print("[DONE] Prediction finished! Results saved to predict_result.csv")
