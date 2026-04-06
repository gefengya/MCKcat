import numpy as np
import pickle
import pandas as pd
import os
from os.path import join
import warnings
import torch
import esm
from drfp import DrfpEncoder
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy import stats
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib as mpl

# Suppress warnings
warnings.filterwarnings("ignore")

# -------------------------- Global Configuration --------------------------
DATA_ROOT = "./data"
FEATURE_ROOT = join(DATA_ROOT, "features")
MODEL_ROOT = join(DATA_ROOT, "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESM_MODEL_NAME = "esm1b_t33_650M_UR50S"
MAX_SEQ_LENGTH = 1022
BATCH_SIZE = 16

TARGET_COL = "log10_kcat"
SOURCE_COL = "geomean_kcat"

os.makedirs(FEATURE_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)


# -------------------------- 1. CSV to PKL Utility --------------------------
def csv_to_pkl(csv_path, pkl_path):
    if not os.path.exists(pkl_path):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except:
            df = pd.read_csv(csv_path, encoding="gbk")

        if SOURCE_COL in df.columns and TARGET_COL not in df.columns:
            df.rename(columns={SOURCE_COL: TARGET_COL}, inplace=True)
        if TARGET_COL in df.columns:
            df = df.dropna(subset=[TARGET_COL])

        df.to_pickle(pkl_path)
        print(f"✅ Converted CSV to PKL: {os.path.basename(csv_path)} → {os.path.basename(pkl_path)}")
    else:
        print(f"ℹ️ PKL already exists, skipping: {os.path.basename(pkl_path)}")

    return pd.read_pickle(pkl_path)


# -------------------------- 2. ESM Feature Extraction --------------------------
def load_esm_model():
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
    model = model.eval().to(DEVICE)
    return model, alphabet


def extract_esm_features(df, seq_col="enzyme_sequence"):
    if seq_col not in df.columns:
        raise ValueError(f"Missing sequence column: {seq_col}")

    model, alphabet = load_esm_model()
    batch_converter = alphabet.get_batch_converter()

    df[seq_col] = df[seq_col].apply(
        lambda x: x[:MAX_SEQ_LENGTH] if isinstance(x, str) and len(x) > MAX_SEQ_LENGTH else x
    )

    sequences = df[seq_col].tolist()
    features = []

    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="🔧 Extracting ESM features"):
        batch_seq = sequences[i:i+BATCH_SIZE]
        batch_labels = [f"seq_{i+j}" for j in range(len(batch_seq))]
        batch_data = list(zip(batch_labels, batch_seq))

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(DEVICE)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33]

        for rep in token_reps:
            seq_rep = rep[1:-1].mean(dim=0).cpu().numpy()
            features.append(seq_rep)

    return np.array(features)


# -------------------------- 3. DRFP Feature Extraction --------------------------
def extract_drfp_features(df, smiles_col="reaction_smiles"):
    if smiles_col not in df.columns:
        raise ValueError(f"Missing SMILES column: {smiles_col}")

    smiles_list = df[smiles_col].fillna("").tolist()
    fps, _ = DrfpEncoder.encode(smiles_list, nBits=2048)
    return fps


# -------------------------- 4. Full Preprocessing Pipeline --------------------------
def preprocess_data(csv_train_path, csv_test_path, dataset_name, fold=None):
    fold_suffix = f"_fold_{fold}" if fold is not None else ""

    pkl_train_path = join(FEATURE_ROOT, dataset_name, f"train{fold_suffix}_raw.pkl")
    pkl_test_path = join(FEATURE_ROOT, dataset_name, f"test{fold_suffix}_raw.pkl")
    final_train_path = join(FEATURE_ROOT, dataset_name, f"train{fold_suffix}_with_features.pkl")
    final_test_path = join(FEATURE_ROOT, dataset_name, f"test{fold_suffix}_with_features.pkl")

    os.makedirs(join(FEATURE_ROOT, dataset_name), exist_ok=True)

    print(f"\n📥 Converting {dataset_name}{fold_suffix} to PKL...")
    df_train = csv_to_pkl(csv_train_path, pkl_train_path)
    df_test = csv_to_pkl(csv_test_path, pkl_test_path)

    if not os.path.exists(final_train_path) or not os.path.exists(final_test_path):
        print(f"🔍 Extracting features for {dataset_name}{fold_suffix} (ESM + DRFP)...")

        print(f"   • Extracting ESM for training set ({len(df_train)} samples)")
        esm_train = extract_esm_features(df_train, seq_col="enzyme_sequence")
        print(f"   • Extracting ESM for test set ({len(df_test)} samples)")
        esm_test = extract_esm_features(df_test, seq_col="enzyme_sequence")

        print(f"   • Extracting DRFP for training set")
        drfp_train = extract_drfp_features(df_train, smiles_col="reaction_smiles")
        print(f"   • Extracting DRFP for test set")
        drfp_test = extract_drfp_features(df_test, smiles_col="reaction_smiles")

        df_train["ESM1b"] = esm_train.tolist()
        df_test["ESM1b"] = esm_test.tolist()
        df_train["drfp"] = drfp_train.tolist()
        df_test["drfp"] = drfp_test.tolist()

        df_train.to_pickle(final_train_path)
        df_test.to_pickle(final_test_path)
        print(f"✅ Feature extraction done for {dataset_name}{fold_suffix}")
    else:
        print(f"ℹ️ Features exist for {dataset_name}{fold_suffix}, loading directly...")

    return final_train_path, final_test_path


# -------------------------- 5. Model Training --------------------------
def train_models(train_path, test_path, dataset_name, fold=None):
    print(f"\n📊 Loading training data for {dataset_name}...")
    data_train = pd.read_pickle(train_path)
    data_test = pd.read_pickle(test_path)

    if TARGET_COL not in data_train.columns or TARGET_COL not in data_test.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    model_dir = join(MODEL_ROOT, dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"📦 Models will be saved to: {model_dir}")

    train_ESM1b = np.array(list(data_train["ESM1b"]))
    train_drfp = np.array(list(data_train["drfp"]))
    train_Y = np.array(data_train[TARGET_COL].dropna())

    test_ESM1b = np.array(list(data_test["ESM1b"]))
    test_drfp = np.array(list(data_test["drfp"]))
    test_Y = np.array(data_test[TARGET_COL].dropna())

    valid_train = ~np.isnan(train_Y)
    valid_test = ~np.isnan(test_Y)

    train_ESM1b = train_ESM1b[valid_train]
    train_drfp = train_drfp[valid_train]
    train_Y = train_Y[valid_train]

    test_ESM1b = test_ESM1b[valid_test]
    test_drfp = test_drfp[valid_test]
    test_Y = test_Y[valid_test]

    if len(train_Y) == 0 or len(test_Y) == 0:
        print(f"❌ No valid data for {dataset_name}{fold_suffix}, skipping training")
        return


    # -------------------------- Model 1: ESM1b only --------------------------
    print("\n" + "-"*50)
    print(f"🚀 Training Model 1: ESM1b")
    print("-"*50)
    param_esm = {
        'learning_rate': 0.2831145406836757,
        'max_delta_step': 0.07686715986169101,
        'max_depth': int(np.round(4.96836783761305)),
        'min_child_weight': 6.905400087083855,
        'reg_alpha': 1.717314107718892,
        'reg_lambda': 2.470354543039016,
        'objective': 'reg:squarederror',
        'device': DEVICE.type
    }
    num_round = 313

    dtrain_esm = xgb.DMatrix(train_ESM1b, label=train_Y)
    dtest_esm = xgb.DMatrix(test_ESM1b, label=test_Y)
    bst_esm = xgb.train(param_esm, dtrain_esm, num_round, verbose_eval=False)

    esm_model_path = join(model_dir, f"xgb_esm1b{fold_suffix}.model")
    bst_esm.save_model(esm_model_path)

    y_pred_esm = bst_esm.predict(dtest_esm)
    mse_esm = np.mean(np.square(test_Y - y_pred_esm))
    r2_esm = r2_score(test_Y, y_pred_esm)
    pearson_esm = stats.pearsonr(test_Y, y_pred_esm)[0]

    print(f"✅ Model 1 saved: {os.path.basename(esm_model_path)}")
    print(f"📊 Performance: Pearson={pearson_esm:.4f}, MSE={mse_esm:.4f}, R²={r2_esm:.4f}")


    # -------------------------- Model 2: DRFP only --------------------------
    print("\n" + "-"*50)
    print(f"🚀 Training Model 2: DRFP")
    print("-"*50)
    param_drfp = {
        'learning_rate': 0.08987247189322463,
        'max_delta_step': 1.1939737318908727,
        'max_depth': int(np.round(11.268531225242574)),
        'min_child_weight': 2.8172720953826302,
        'reg_alpha': 1.9412226989868904,
        'reg_lambda': 4.950543905603358,
        'objective': 'reg:squarederror',
        'device': DEVICE.type
    }
    num_round = 109

    dtrain_drfp = xgb.DMatrix(train_drfp, label=train_Y)
    dtest_drfp = xgb.DMatrix(test_drfp, label=test_Y)
    bst_drfp = xgb.train(param_drfp, dtrain_drfp, num_round, verbose_eval=False)

    drfp_model_path = join(model_dir, f"xgb_drfp{fold_suffix}.model")
    bst_drfp.save_model(drfp_model_path)

    y_pred_drfp = bst_drfp.predict(dtest_drfp)
    mse_drfp = np.mean(np.square(test_Y - y_pred_drfp))
    r2_drfp = r2_score(test_Y, y_pred_drfp)
    pearson_drfp = stats.pearsonr(test_Y, y_pred_drfp)[0]

    print(f"✅ Model 2 saved: {os.path.basename(drfp_model_path)}")
    print(f"📊 Performance: Pearson={pearson_drfp:.4f}, MSE={mse_drfp:.4f}, R²={r2_drfp:.4f}")


    # -------------------------- Model 3: Combined ESM1b + DRFP --------------------------
    print("\n" + "-"*50)
    print(f"🚀 Training Model 3: Combined ESM1b+DRFP")
    print("-"*50)
    train_combined = np.concatenate([train_ESM1b, train_drfp], axis=1)
    test_combined = np.concatenate([test_ESM1b, test_drfp], axis=1)

    param_combined = {
        'learning_rate': 0.05221672412884108,
        'max_delta_step': 1.0767235463496743,
        'max_depth': int(np.round(11.329014411591299)),
        'min_child_weight': 14.724796449973605,
        'reg_alpha': 2.8295816318634452,
        'reg_lambda': 0.6528469146574993,
        'objective': 'reg:squarederror',
        'device': DEVICE.type
    }
    num_round = 299

    dtrain_combined = xgb.DMatrix(train_combined, label=train_Y)
    dtest_combined = xgb.DMatrix(test_combined, label=test_Y)
    bst_combined = xgb.train(param_combined, dtrain_combined, num_round, verbose_eval=False)

    combined_model_path = join(model_dir, f"xgb_combined{fold_suffix}.model")
    bst_combined.save_model(combined_model_path)

    y_pred_combined = bst_combined.predict(dtest_combined)
    mse_combined = np.mean(np.square(test_Y - y_pred_combined))
    r2_combined = r2_score(test_Y, y_pred_combined)
    pearson_combined = stats.pearsonr(test_Y, y_pred_combined)[0]

    print(f"✅ Model 3 saved: {os.path.basename(combined_model_path)}")
    print(f"📊 Performance: Pearson={pearson_combined:.4f}, MSE={mse_combined:.4f}, R²={r2_combined:.4f}")


    # -------------------------- Model 4: Mean Fusion --------------------------
    print("\n" + "-"*50)
    print(f"🚀 Computing Model 4: Mean Fusion")
    print("-"*50)
    y_pred_mean = (y_pred_esm + y_pred_drfp) / 2

    mse_mean = np.mean(np.square(test_Y - y_pred_mean))
    r2_mean = r2_score(test_Y, y_pred_mean)
    pearson_mean = stats.pearsonr(test_Y, y_pred_mean)[0]

    mean_result_path = join(model_dir, f"mean_fusion{fold_suffix}.pkl")
    with open(mean_result_path, "wb") as f:
        pickle.dump({
            "dataset": dataset_name,
            "fold": fold,
            "y_true": test_Y,
            "y_pred_esm": y_pred_esm,
            "y_pred_drfp": y_pred_drfp,
            "y_pred_combined": y_pred_combined,
            "y_pred_mean": y_pred_mean
        }, f)

    print(f"✅ Fusion results saved: {os.path.basename(mean_result_path)}")
    print(f"📊 Performance: Pearson={pearson_mean:.4f}, MSE={mse_mean:.4f}, R²={r2_mean:.4f}")


# -------------------------- 6. Dataset Training Functions --------------------------
def train_warm():
    print("\n" + "="*60)
    print("📌 Training Warm dataset (folds 1–5)")
    print("="*60)

    for fold in range(1, 6):
        print(f"\n" + "="*40)
        print(f"🔄 Processing Warm fold {fold}")
        print("="*40)

        train_csv = join(DATA_ROOT, "turnup", "warm", str(fold), "train_fold_6_with_sub_.csv")
        test_csv = join(DATA_ROOT, "turnup", "warm", str(fold), "test_fold_6_with_sub_.csv")

        if not os.path.exists(train_csv):
            print(f"❌ Missing train file for Warm fold {fold}: {train_csv}")
            continue
        if not os.path.exists(test_csv):
            print(f"❌ Missing test file for Warm fold {fold}: {test_csv}")
            continue

        train_pkl, test_pkl = preprocess_data(train_csv, test_csv, "warm", fold)
        train_models(train_pkl, test_pkl, "warm", fold)

    print(f"\n🎉 Warm dataset training completed!")


def train_cold_enzyme():
    print("\n" + "="*60)
    print("📌 Training Cold-Enzyme dataset (folds 1–5)")
    print("="*60)

    for fold in range(1, 6):
        print(f"\n" + "="*40)
        print(f"🔄 Processing Cold-Enzyme fold {fold}")
        print("="*40)

        train_csv = join(DATA_ROOT, "turnup", "cold_enzyme", str(fold), "train_fold_6_with_sub_.csv")
        test_csv = join(DATA_ROOT, "turnup", "cold_enzyme", str(fold), "test_fold_6_with_sub_.csv")

        if not os.path.exists(train_csv):
            print(f"❌ Missing train file for Cold-Enzyme fold {fold}: {train_csv}")
            continue
        if not os.path.exists(test_csv):
            print(f"❌ Missing test file for Cold-Enzyme fold {fold}: {test_csv}")
            continue

        train_pkl, test_pkl = preprocess_data(train_csv, test_csv, "cold_enzyme", fold)
        train_models(train_pkl, test_pkl, "cold_enzyme", fold)

    print(f"\n🎉 Cold-Enzyme dataset training completed!")


def train_cold_reaction():
    print("\n" + "="*60)
    print("📌 Training Cold-Reaction dataset (folds 1–5)")
    print("="*60)

    for fold in range(1, 6):
        print(f"\n" + "="*40)
        print(f"🔄 Processing Cold-Reaction fold {fold}")
        print("="*40)

        train_csv = join(DATA_ROOT, "turnup", "cold_reaction", str(fold), "train_fold_6_with_sub_.csv")
        test_csv = join(DATA_ROOT, "turnup", "cold_reaction", str(fold), "test_fold_6_with_sub_.csv")

        if not os.path.exists(train_csv):
            print(f"❌ Missing train file for Cold-Reaction fold {fold}: {train_csv}")
            continue
        if not os.path.exists(test_csv):
            print(f"❌ Missing test file for Cold-Reaction fold {fold}: {test_csv}")
            continue

        train_pkl, test_pkl = preprocess_data(train_csv, test_csv, "cold_reaction", fold)
        train_models(train_pkl, test_pkl, "cold_reaction", fold)

    print(f"\n🎉 Cold-Reaction dataset training completed!")


def train_catpred_basic():
    print("\n" + "="*60)
    print("📌 Training CatPred basic dataset (no folds)")
    print("="*60)

    train_csv = join(DATA_ROOT, "catpred_catpro", "train.csv")
    test_csv = join(DATA_ROOT, "catpred_catpro", "Held-out test.csv")

    if not os.path.exists(train_csv):
        print(f"❌ Missing train file: {train_csv}")
        return
    if not os.path.exists(test_csv):
        print(f"❌ Missing test file: {test_csv}")
        return

    train_pkl, test_pkl = preprocess_data(train_csv, test_csv, "catpred_basic", None)
    train_models(train_pkl, test_pkl, "catpred_basic", None)

    print(f"\n🎉 CatPred basic dataset training completed!")


# -------------------------- Main --------------------------
if __name__ == "__main__":
    print("="*60)
    print("📋 Environment Info")
    print("="*60)
    print(f"• Data root: {DATA_ROOT}")
    print(f"• Feature dir: {FEATURE_ROOT}")
    print(f"• Model dir: {MODEL_ROOT}")
    print(f"• Device: {DEVICE}")
    print(f"• ESM model: {ESM_MODEL_NAME}")
    print(f"• Batch size: {BATCH_SIZE}")
    print("="*60)

    train_warm()
    train_cold_enzyme()
    train_cold_reaction()
    train_catpred_basic()

    print("\n" + "="*60)
    print("🎉 All selected datasets finished training!")
    print("="*60)
    print(f"• Features: {FEATURE_ROOT}")
    print(f"• Models: {MODEL_ROOT}")
    print(f"• Fusion results: mean_fusion*.pkl under each dataset folder")
    print("="*60)