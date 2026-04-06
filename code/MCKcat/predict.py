import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Reuse core configurations and utility functions from original training code
from train import (
    CFG, init_models, MODEL_CACHE,
    process_csv_with_cache, generate_key, load_cache
)
from MCKcat_model import Model_Regression  # Import your model class


class PredictDataset(Dataset):
    """Dataset class for prediction only (no labels, feature processing only)"""

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)  # Reset index to avoid positional errors
        self.sequence_col = self._find_col("sequence")  # Auto-match sequence column (case-insensitive)
        self.reaction_col = self._find_col("reaction_smiles")  # Auto-match reaction SMILES column
        self._validate_data()  # Validate data integrity

    def _find_col(self, target_name):
        """Automatically find column name (compatible with case differences, e.g., Sequence/SEQUENCE)"""
        target_lower = target_name.lower()
        for col in self.dataframe.columns:
            if col.lower() == target_lower:
                return col
        raise ValueError(f"Column '{target_name}' not found (candidate columns: {list(self.dataframe.columns)})")

    def _validate_data(self):
        """Validate if data contains required features and valid values"""
        # Check if feature columns exist
        required_cols = [self.sequence_col, self.reaction_col, "rxnfp", "uniref50_dim1024"]
        for col in required_cols:
            if col not in self.dataframe.columns:
                raise ValueError(
                    f"Prediction data missing required feature column: {col}, please run feature calculation pipeline first")

        # Filter invalid features (null values/non-arrays)
        invalid_mask = (
                self.dataframe["rxnfp"].isna() |
                self.dataframe["uniref50_dim1024"].isna() |
                ~self.dataframe["rxnfp"].apply(lambda x: isinstance(x, (list, np.ndarray))) |
                ~self.dataframe["uniref50_dim1024"].apply(lambda x: isinstance(x, (list, np.ndarray)))
        )
        if invalid_mask.sum() > 0:
            print(f"Warning: Filtered {invalid_mask.sum()} invalid feature entries (null/non-array values)")
            self.dataframe = self.dataframe[~invalid_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Get single sample features (adapt to model input format)"""
        row = self.dataframe.iloc[idx]

        # Reaction fingerprint (rxnfp): convert to float32 tensor, add batch dimension
        rxnfp = torch.tensor(row["rxnfp"], dtype=CFG.dtype).unsqueeze(0)
        # Protein features (uniref50_dim1024): convert to float32 tensor
        protein = torch.tensor(row["uniref50_dim1024"], dtype=CFG.dtype)
        # Original information (for result matching)
        seq = row[self.sequence_col]
        reaction = row[self.reaction_col]

        return rxnfp, protein, seq, reaction


def load_single_model(model_path, device=None):
    """Load a single trained model (supports device specification)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    # Adapt to device (use CFG.DEVICE from training by default)
    device = device or CFG.DEVICE
    # Initialize model and load weights
    model = Model_Regression().to(CFG.dtype).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Switch to evaluation mode (disable Dropout etc.)
    print(f"Successfully loaded model: {os.path.basename(model_path)} (device: {device})")
    return model


def predict_single_model(model, predict_loader, device=None):
    """Predict with a single model, return predictions and original information"""
    device = device or CFG.DEVICE
    model = model.to(device)
    all_preds = []  # Model predictions (log10kcat_max)
    all_sequences = []  # Corresponding protein sequences
    all_reactions = []  # Corresponding reaction SMILES
    all_raw_indices = []  # Corresponding original data indices (for traceability)

    with torch.no_grad():  # Disable gradient calculation to speed up prediction
        for batch_idx, (rxnfp, protein, seq, reaction) in tqdm(
                enumerate(predict_loader), total=len(predict_loader), desc="Model prediction"
        ):
            # Move data to target device
            rxnfp = rxnfp.to(device)
            protein = protein.to(device)

            # Mixed precision prediction (consistent with training, improves speed)
            with torch.cuda.amp.autocast():
                outputs, _ = model(rxnfp, protein)  # Model output (match your model's return format)

            # Collect results (convert to numpy array for subsequent processing)
            all_preds.extend(outputs.cpu().numpy().ravel())
            all_sequences.extend(seq)
            all_reactions.extend(reaction)
            all_raw_indices.extend([batch_idx] * len(seq))  # Batch index mapping

    # Organize prediction results into DataFrame
    result_df = pd.DataFrame({
        "raw_index": all_raw_indices,
        "sequence": all_sequences,
        "reaction_smiles": all_reactions,
        "pred_log10kcat_max": all_preds
    })
    return result_df


def predict_ensemble(models, predict_loader, device=None):
    """Ensemble prediction (average results from multiple models) to improve prediction stability"""
    if len(models) == 0:
        raise ValueError("Ensemble prediction requires at least 1 model")

    # Step 1: Collect predictions from each model
    model_preds_list = []
    for idx, model in enumerate(models):
        print(f"\n===== Ensemble Model {idx + 1}/{len(models)} Prediction =====")
        pred_df = predict_single_model(model, predict_loader, device)
        model_preds_list.append(pred_df["pred_log10kcat_max"].values)
        # Save individual model results (optional)
        pred_df.to_csv(
            f"ensemble_model_{idx + 1}_predictions.csv",
            index=False, encoding="utf-8"
        )

    # Step 2: Calculate ensemble results (simple average, can be changed to weighted average)
    ensemble_preds = np.mean(model_preds_list, axis=0)

    # Step 3: Organize ensemble results (reuse metadata from first model)
    ensemble_df = pred_df.copy()
    ensemble_df["ensemble_pred_log10kcat_max"] = ensemble_preds
    # Add predictions from each model (optional, for analyzing model consistency)
    for idx, preds in enumerate(model_preds_list):
        ensemble_df[f"model_{idx + 1}_pred_log10kcat_max"] = preds

    print(f"\nEnsemble prediction completed (average of {len(models)} models)")
    return ensemble_df


def predict_pipeline(
        predict_csv,  # Path to CSV with data to predict
        cache_file,  # Path to feature cache file (reuse from training)
        model_paths,  # Model paths (single path string or list of multiple paths)
        output_csv="predict_results.csv",  # Path to save prediction results
        batch_size=None,  # Batch size for prediction (use training config by default)
        use_ensemble=True  # Whether to use ensemble prediction (effective when multiple model_paths)
):
    """
    Complete prediction pipeline: data preprocessing → feature calculation → model prediction → result saving

    Parameter explanation:
    - predict_csv: CSV with data to predict (must contain sequence and reaction_smiles columns)
    - cache_file: Feature cache file (reuse from training to avoid recalculating rxnfp and uniref50)
    - model_paths: Model paths (single model path string / list of multiple model paths)
    - output_csv: Path to save prediction results
    - batch_size: Prediction batch size (default CFG.batch_size, reduce if out of memory)
    - use_ensemble: Whether to ensemble when multiple models (True=average, False=return individual results)
    """
    # --------------------------
    # 1. Initialize environment (models + device)
    # --------------------------
    init_models(force_reload=False)  # Load pretrained feature models (no reload if exists)
    batch_size = batch_size or CFG.batch_size
    device = CFG.DEVICE
    print(f"Prediction environment initialized (device: {device}, batch size: {batch_size})")

    # --------------------------
    # 2. Preprocess prediction data (calculate/load features)
    # --------------------------
    print(f"\n===== Preprocessing prediction data: {os.path.basename(predict_csv)} =====")
    # Process data and calculate features (reuse cache mechanism from training code to avoid recalculating rxnfp and uniref50)
    predict_df = process_csv_with_cache(
        csv_path=predict_csv,
        cache_path=cache_file,
        sequence_col="sequence",  # Adjust according to your CSV column names
        reaction_col="reaction_smiles",  # Adjust according to your CSV column names
        output_col="geomean_kcat",
        # No labels for prediction, use placeholder (ensure CSV has this column or modify process_csv_with_cache)
        output_csv="predict_data_with_features.csv"  # Save data with features (optional)
    )
    # Fix: Manually add placeholder if output_col not in prediction data (avoid process_csv_with_cache error)
    if "geomean_kcat" not in predict_df.columns:
        predict_df["geomean_kcat"] = 0.0
    print(f"Preprocessing completed, valid prediction samples: {len(predict_df)}")

    # --------------------------
    # 3. Create prediction data loader
    # --------------------------
    predict_dataset = PredictDataset(predict_df)
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=batch_size,
        shuffle=False,  # Do not shuffle for prediction to keep result-input correspondence
        num_workers=4,  # Adjust according to CPU cores (set to 0 if out of memory)
        pin_memory=True  # Accelerate GPU data transfer
    )

    # --------------------------
    # 4. Load models (single/multiple)
    # --------------------------
    if isinstance(model_paths, str):
        model_paths = [model_paths]  # Unify to list format
    models = [load_single_model(path, device) for path in model_paths]

    # --------------------------
    # 5. Execute prediction (single/ensemble)
    # --------------------------
    print(f"\n===== Starting prediction (total {len(models)} models) =====")
    if len(models) == 1 or not use_ensemble:
        # Single model prediction or multiple models without ensemble
        final_result = predict_single_model(models[0], predict_loader, device)
        # Append results from other models if multiple models exist
        for idx in range(1, len(models)):
            temp_result = predict_single_model(models[idx], predict_loader, device)
            final_result[f"model_{idx + 1}_pred_log10kcat_max"] = temp_result["pred_log10kcat_max"]
    else:
        # Multiple model ensemble prediction
        final_result = predict_ensemble(models, predict_loader, device)

    # --------------------------
    # 6. Save prediction results
    # --------------------------
    # Merge original data information (add other columns from input CSV if needed)
    input_df = pd.read_csv(predict_csv)
    final_result = pd.merge(
        final_result, input_df,
        left_on=["sequence", "reaction_smiles"],
        right_on=["sequence", "reaction_smiles"],  # Match by sequence and reaction SMILES
        how="left", suffixes=("_pred", "_input")
    )
    # Save results
    final_result.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nPrediction completed! Results saved to: {output_csv}")
    print(f"Predicted samples: {len(final_result)}")
    print(f"Result columns: {list(final_result.columns)}")

    return final_result


def evaluate_predictions(pred_df, true_label_col="real_log10kcat_max"):
    """
    (Optional) Evaluate prediction performance if ground truth labels exist in prediction data (MSE/R2/Pearson correlation etc.)
    """
    # Check if ground truth labels exist
    if true_label_col not in pred_df.columns:
        print("Ground truth label column not found, skipping performance evaluation")
        return None

    # Filter invalid labels (null/infinite values)
    valid_mask = pred_df[true_label_col].notna() & np.isfinite(pred_df[true_label_col])
    valid_df = pred_df[valid_mask]
    if len(valid_df) == 0:
        print("No valid ground truth labels, skipping performance evaluation")
        return None

    # Calculate evaluation metrics (using ensemble prediction as example, can replace with single model column)
    y_true = valid_df[true_label_col].values
    y_pred = valid_df.get("ensemble_pred_log10kcat_max", valid_df["pred_log10kcat_max"]).values

    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Pearson_Corr": pearsonr(y_true, y_pred)[0],
        "Valid_Samples": len(valid_df)
    }

    # Print evaluation results
    print("\n===== Prediction Performance Evaluation =====")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")
    return metrics


# --------------------------
# Prediction entry point (can run directly)
# --------------------------
if __name__ == "__main__":
    # --------------------------
    # 1. Configure prediction parameters (modify according to your actual paths)
    # --------------------------
    PREDICT_CSV = "predict.csv"  # Your data to predict (must contain sequence and reaction_smiles columns)
    CACHE_FILE = "features_cache.pkl.gz"  # Reuse feature cache from training
    MODEL_SAVE_DIR = "model"  # Model save directory (set in training code)
    OUTPUT_CSV = "predict_results.csv"  # Path to save final prediction results

    # Load "best models" for all seeds (can also choose "last epoch models")
    MODEL_PATHS = [
        os.path.join(MODEL_SAVE_DIR, f"model_seed_{seed}.pth")
        for seed in range(10)  # Consistent with num_seeds in training (10 seeds)
    ]
    # Filter non-existent model files (avoid errors)
    MODEL_PATHS = [path for path in MODEL_PATHS if os.path.exists(path)]
    if len(MODEL_PATHS) == 0:
        raise FileNotFoundError(f"No model files found in {MODEL_SAVE_DIR}")

    # --------------------------
    # 2. Execute prediction pipeline
    # --------------------------
    pred_result = predict_pipeline(
        predict_csv=PREDICT_CSV,
        cache_file=CACHE_FILE,
        model_paths=MODEL_PATHS,
        output_csv=OUTPUT_CSV,
        batch_size=8,  # Adjust according to GPU memory (increase if memory allows, e.g., 16/32)
        use_ensemble=True  # Enable ensemble prediction (average of 10 models)
    )

    # --------------------------
    # 3. (Optional) Evaluate performance if ground truth labels exist
    # --------------------------
    if "real_log10kcat_max" in pred_result.columns:
        evaluate_predictions(pred_result, true_label_col="real_log10kcat_max")