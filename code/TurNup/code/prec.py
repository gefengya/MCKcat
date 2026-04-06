import numpy as np
import pandas as pd
import os
from os.path import join
from sklearn.metrics import r2_score
from scipy import stats
import xgboost as xgb
from tqdm import tqdm

# -------------------------- Global Configuration --------------------------
DATA_ROOT = "./data"
MODEL_ROOT = join(DATA_ROOT, "models")
RESULT_ROOT = join(DATA_ROOT, "predict_results")

FEATURE_COLS = {
    "esm": "ESM1b",
    "drfp": "drfp",
    "target": "log10_kcat"
}

MODEL_FILENAMES = {
    "esm1b": "xgb_esm1b{fold_suffix}.model",
    "drfp": "xgb_drfp{fold_suffix}.model",
    "combined": "xgb_combined{fold_suffix}.model"
}

DATASETS = [
    {"name": "warm", "subpath": "TurNup_DB/warm", "is_fold": True, "test_template": "test.csv"},
    {"name": "cold_enzyme", "subpath": "TurNup_DB/cold_enzyme", "is_fold": True, "test_template": "test.csv"},
    {"name": "cold_reaction", "subpath": "TurNup_DB/cold_reaction", "is_fold": True, "test_template": "test.csv"},
    {"name": "catpred_basic", "subpath": "MCKcat_DB", "is_fold": False, "test_template": "Held-out test.csv"}
]

FOLD_RANGE = range(1, 6)


# -------------------------- Helper Functions --------------------------
def load_preprocessed_data(data_path, dataset_name, fold=None):
    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    pkl_dir = join(DATA_ROOT, "features", dataset_name)
    pkl_path = join(pkl_dir, f"test{fold_suffix}_with_features.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Feature-enriched test PKL not found: {pkl_path}\n"
            f"Please run the preprocessing step first for {dataset_name}."
        )

    print(f"📥 Loading {dataset_name}{fold_suffix} test data: {os.path.basename(pkl_path)}")
    data = pd.read_pickle(pkl_path)

    if FEATURE_COLS["target"] not in data.columns:
        candidate_cols = ["log10kcat_max", "geomean_kcat", "log10_kcat"]
        matched_col = [col for col in candidate_cols if col in data.columns]
        if matched_col:
            data.rename(columns={matched_col[0]: FEATURE_COLS["target"]}, inplace=True)
            print(f"ℹ️ Auto-renamed target column '{matched_col[0]}' → '{FEATURE_COLS['target']}'")
        else:
            raise ValueError(f"Target column missing. Candidates: {candidate_cols}")

    for feat_col in [FEATURE_COLS["esm"], FEATURE_COLS["drfp"]]:
        if feat_col not in data.columns:
            raise ValueError(f"Feature column '{feat_col}' missing. Check preprocessing.")

    if "reaction_smiles" not in data.columns:
        print("⚠️ 'reaction_smiles' not found. Assigning default reaction type.")
        data["reaction_smiles"] = "default_reaction"

    return data


def load_xgb_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"🔧 Loading model: {os.path.basename(model_path)}")
    return xgb.Booster(model_file=model_path)


def calculate_scc_by_reaction(y_true, y_pred, reaction_list):
    unique_reactions = np.unique(reaction_list)
    scc_dict = {}
    valid_count = 0

    print(f"\n📊 Calculating SCC per reaction type (total: {len(unique_reactions)})")
    for reaction in tqdm(unique_reactions, desc="Computing SCC"):
        mask = reaction_list == reaction
        sample_count = np.sum(mask)

        if sample_count < 3:
            print(f"  ⚠️ Reaction '{reaction}' has {sample_count} < 3 samples → skipped")
            continue

        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]
        scc, p_val = stats.spearmanr(y_true_sub, y_pred_sub)
        scc_dict[reaction] = {"scc": scc, "p_value": p_val, "sample_count": sample_count}
        valid_count += 1

    if valid_count == 0:
        print("  ❌ No valid reactions for SCC calculation")
        return np.nan, scc_dict
    else:
        mean_scc = np.mean([v["scc"] for v in scc_dict.values()])
        print(f"  ✅ Valid reactions: {valid_count}, Mean SCC: {mean_scc:.4f}")
        return mean_scc, scc_dict


def save_prediction_results(results, dataset_name, model_type, fold=None):
    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    result_dir = join(RESULT_ROOT, dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    pred_filename = f"pred_{dataset_name}_{model_type}{fold_suffix}.csv"
    pred_path = join(result_dir, pred_filename)
    results["pred_df"].to_csv(pred_path, index=False, encoding="utf-8")
    print(f"💾 Predictions saved to: {pred_path}")

    if "scc_dict" in results and results["scc_dict"]:
        scc_filename = f"scc_{dataset_name}_{model_type}{fold_suffix}.csv"
        scc_path = join(result_dir, scc_filename)
        scc_df = pd.DataFrame([
            {
                "reaction_smiles": react,
                "scc": info["scc"],
                "p_value": info["p_value"],
                "sample_count": info["sample_count"]
            }
            for react, info in results["scc_dict"].items()
        ])
        scc_df.to_csv(scc_path, index=False, encoding="utf-8")
        print(f"💾 SCC results saved to: {scc_path}")

    return pred_path, scc_path if "scc_dict" in results else None


# -------------------------- Prediction Functions --------------------------
def predict_esm1b(data, model_path, fold=None):
    dataset_name = data.attrs.get("dataset_name", "unknown")
    print(f"\n" + "="*50)
    print(f"🚀 Predicting with ESM1b ({dataset_name})")
    print("="*50)

    X = np.array(list(data[FEATURE_COLS["esm"]]))
    y_true = np.array(data[FEATURE_COLS["target"]])
    reactions = np.array(data["reaction_smiles"])

    model = load_xgb_model(model_path)
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest)

    metrics = {
        "pearson": stats.pearsonr(y_true, y_pred)[0],
        "mse": np.mean(np.square(y_true - y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mean_scc": np.nan,
        "scc_dict": {}
    }
    metrics["mean_scc"], metrics["scc_dict"] = calculate_scc_by_reaction(y_true, y_pred, reactions)

    print(f"\n📈 ESM1b Performance:")
    print(f"  Pearson: {metrics['pearson']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Mean SCC: {metrics['mean_scc']:.4f}" if not np.isnan(metrics['mean_scc']) else "  Mean SCC: N/A")

    pred_df = pd.DataFrame({
        "sample_id": range(len(y_true)),
        "reaction_smiles": reactions,
        "true_log10_kcat": y_true,
        "pred_log10_kcat_esm1b": y_pred,
        "abs_error": np.abs(y_true - y_pred)
    })

    save_prediction_results(
        results={"pred_df": pred_df, "scc_dict": metrics["scc_dict"], "metrics": metrics},
        dataset_name=dataset_name,
        model_type="esm1b",
        fold=fold
    )

    return {"y_pred": y_pred, "metrics": metrics, "pred_df": pred_df}


def predict_drfp(data, model_path, fold=None):
    dataset_name = data.attrs.get("dataset_name", "unknown")
    print(f"\n" + "="*50)
    print(f"🚀 Predicting with DRFP ({dataset_name})")
    print("="*50)

    X = np.array(list(data[FEATURE_COLS["drfp"]]))
    y_true = np.array(data[FEATURE_COLS["target"]])
    reactions = np.array(data["reaction_smiles"])

    model = load_xgb_model(model_path)
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest)

    metrics = {
        "pearson": stats.pearsonr(y_true, y_pred)[0],
        "mse": np.mean(np.square(y_true - y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mean_scc": np.nan,
        "scc_dict": {}
    }
    metrics["mean_scc"], metrics["scc_dict"] = calculate_scc_by_reaction(y_true, y_pred, reactions)

    print(f"\n📈 DRFP Performance:")
    print(f"  Pearson: {metrics['pearson']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Mean SCC: {metrics['mean_scc']:.4f}" if not np.isnan(metrics['mean_scc']) else "  Mean SCC: N/A")

    pred_df = pd.DataFrame({
        "sample_id": range(len(y_true)),
        "reaction_smiles": reactions,
        "true_log10_kcat": y_true,
        "pred_log10_kcat_drfp": y_pred,
        "abs_error": np.abs(y_true - y_pred)
    })

    save_prediction_results(
        results={"pred_df": pred_df, "scc_dict": metrics["scc_dict"], "metrics": metrics},
        dataset_name=dataset_name,
        model_type="drfp",
        fold=fold
    )

    return {"y_pred": y_pred, "metrics": metrics, "pred_df": pred_df}


def predict_combined(data, model_path, fold=None):
    dataset_name = data.attrs.get("dataset_name", "unknown")
    print(f"\n" + "="*50)
    print(f"🚀 Predicting with Combined (ESM1b+DRFP) ({dataset_name})")
    print("="*50)

    X_esm = np.array(list(data[FEATURE_COLS["esm"]]))
    X_drfp = np.array(list(data[FEATURE_COLS["drfp"]]))
    X_combined = np.concatenate([X_esm, X_drfp], axis=1)

    y_true = np.array(data[FEATURE_COLS["target"]])
    reactions = np.array(data["reaction_smiles"])

    model = load_xgb_model(model_path)
    dtest = xgb.DMatrix(X_combined)
    y_pred = model.predict(dtest)

    metrics = {
        "pearson": stats.pearsonr(y_true, y_pred)[0],
        "mse": np.mean(np.square(y_true - y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mean_scc": np.nan,
        "scc_dict": {}
    }
    metrics["mean_scc"], metrics["scc_dict"] = calculate_scc_by_reaction(y_true, y_pred, reactions)

    print(f"\n📈 Combined Performance:")
    print(f"  Pearson: {metrics['pearson']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Mean SCC: {metrics['mean_scc']:.4f}" if not np.isnan(metrics['mean_scc']) else "  Mean SCC: N/A")

    pred_df = pd.DataFrame({
        "sample_id": range(len(y_true)),
        "reaction_smiles": reactions,
        "true_log10_kcat": y_true,
        "pred_log10_kcat_combined": y_pred,
        "abs_error": np.abs(y_true - y_pred)
    })

    save_prediction_results(
        results={"pred_df": pred_df, "scc_dict": metrics["scc_dict"], "metrics": metrics},
        dataset_name=dataset_name,
        model_type="combined",
        fold=fold
    )

    return {"y_pred": y_pred, "metrics": metrics, "pred_df": pred_df}


def predict_ensemble(data, y_pred_esm1b, y_pred_drfp, fold=None):
    dataset_name = data.attrs.get("dataset_name", "unknown")
    print(f"\n" + "="*50)
    print(f"🚀 Predicting with Ensemble (ESM1b+DRFP avg) ({dataset_name})")
    print("="*50)

    y_pred_ensemble = np.mean([y_pred_esm1b, y_pred_drfp], axis=0)
    y_true = np.array(data[FEATURE_COLS["target"]])
    reactions = np.array(data["reaction_smiles"])

    metrics = {
        "pearson": stats.pearsonr(y_true, y_pred_ensemble)[0],
        "mse": np.mean(np.square(y_true - y_pred_ensemble)),
        "r2": r2_score(y_true, y_pred_ensemble),
        "mean_scc": np.nan,
        "scc_dict": {}
    }
    metrics["mean_scc"], metrics["scc_dict"] = calculate_scc_by_reaction(y_true, y_pred_ensemble, reactions)

    print(f"\n📈 Ensemble Performance:")
    print(f"  Pearson: {metrics['pearson']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Mean SCC: {metrics['mean_scc']:.4f}" if not np.isnan(metrics['mean_scc']) else "  Mean SCC: N/A")

    pred_df = pd.DataFrame({
        "sample_id": range(len(y_true)),
        "reaction_smiles": reactions,
        "true_log10_kcat": y_true,
        "pred_log10_kcat_esm1b": y_pred_esm1b,
        "pred_log10_kcat_drfp": y_pred_drfp,
        "pred_log10_kcat_ensemble": y_pred_ensemble,
        "abs_error_ensemble": np.abs(y_true - y_pred_ensemble)
    })

    save_prediction_results(
        results={"pred_df": pred_df, "scc_dict": metrics["scc_dict"], "metrics": metrics},
        dataset_name=dataset_name,
        model_type="ensemble",
        fold=fold
    )

    return {"y_pred_ensemble": y_pred_ensemble, "metrics": metrics, "pred_df": pred_df}


# -------------------------- Main Pipeline --------------------------
def run_prediction_pipeline():
    print("="*60)
    print("📋 Multi-Dataset Batch Prediction Started")
    print("="*60)
    print(f"• Data root: {DATA_ROOT}")
    print(f"• Model root: {MODEL_ROOT}")
    print(f"• Result root: {RESULT_ROOT}")
    print(f"• Datasets: {[ds['name'] for ds in DATASETS]}")
    print("="*60)

    for ds in DATASETS:
        dataset_name = ds["name"]
        dataset_subpath = ds["subpath"]
        is_fold = ds["is_fold"]
        test_template = ds["test_template"]

        print(f"\n" + "="*60)
        print(f"📌 Processing dataset: {dataset_name}")
        print("="*60)

        if is_fold:
            for fold in FOLD_RANGE:
                print(f"\n" + "-"*50)
                print(f"🔄 Fold {fold} - {dataset_name}")
                print("-"*50)

                try:
                    data = load_preprocessed_data(
                        data_path=join(DATA_ROOT, dataset_subpath, str(fold)),
                        dataset_name=dataset_name,
                        fold=fold
                    )
                    data.attrs["dataset_name"] = f"{dataset_name}_fold{fold}"
                except Exception as e:
                    print(f"❌ Failed to load {dataset_name} fold {fold}: {str(e)}")
                    continue

                fold_suffix = f"_fold_{fold}"
                model_paths = {
                    "esm1b": join(MODEL_ROOT, dataset_name, MODEL_FILENAMES["esm1b"].format(fold_suffix=fold_suffix)),
                    "drfp": join(MODEL_ROOT, dataset_name, MODEL_FILENAMES["drfp"].format(fold_suffix=fold_suffix)),
                    "combined": join(MODEL_ROOT, dataset_name, MODEL_FILENAMES["combined"].format(fold_suffix=fold_suffix))
                }

                try:
                    esm_result = predict_esm1b(data, model_paths["esm1b"], fold=fold)
                    drfp_result = predict_drfp(data, model_paths["drfp"], fold=fold)
                    combined_result = predict_combined(data, model_paths["combined"], fold=fold)
                    ensemble_result = predict_ensemble(
                        data=data,
                        y_pred_esm1b=esm_result["y_pred"],
                        y_pred_drfp=drfp_result["y_pred"],
                        fold=fold
                    )
                    print(f"\n✅ {dataset_name} fold {fold} completed!")
                except Exception as e:
                    print(f"❌ {dataset_name} fold {fold} failed: {str(e)}")
                    continue

        else:
            print(f"\n" + "-"*50)
            print(f"🔄 Processing {dataset_name} (no folds)")
            print("-"*50)

            try:
                data = load_preprocessed_data(
                    data_path=join(DATA_ROOT, dataset_subpath),
                    dataset_name=dataset_name,
                    fold=None
                )
                data.attrs["dataset_name"] = dataset_name
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {str(e)}")
                continue

            fold_suffix = ""
            model_paths = {
                "esm1b": join(MODEL_ROOT, dataset_name, MODEL_FILENAMES["esm1b"].format(fold_suffix=fold_suffix)),
                "drfp": join(MODEL_ROOT, dataset_name, MODEL_FILENAMES["drfp"].format(fold_suffix=fold_suffix)),
                "combined": join(MODEL_ROOT, dataset_name, MODEL_FILENAMES["combined"].format(fold_suffix=fold_suffix))
            }

            try:
                esm_result = predict_esm1b(data, model_paths["esm1b"], fold=None)
                drfp_result = predict_drfp(data, model_paths["drfp"], fold=None)
                combined_result = predict_combined(data, model_paths["combined"], fold=None)
                ensemble_result = predict_ensemble(
                    data=data,
                    y_pred_esm1b=esm_result["y_pred"],
                    y_pred_drfp=drfp_result["y_pred"],
                    fold=None
                )
                print(f"\n✅ {dataset_name} completed!")
            except Exception as e:
                print(f"❌ {dataset_name} failed: {str(e)}")
                continue

    print(f"\n" + "="*60)
    print("🎉 All datasets processed successfully!")
    print("="*60)
    print(f"• Results saved to: {RESULT_ROOT}")
    print(f"• Includes predictions + SCC analysis for 4 model types")
    print(f"• Files: pred_*.csv (predictions), scc_*.csv (per-reaction SCC)")
    print("="*60)


if __name__ == "__main__":
    os.makedirs(RESULT_ROOT, exist_ok=True)
    run_prediction_pipeline()