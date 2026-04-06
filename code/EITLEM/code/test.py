import torch
import pandas as pd
import numpy as np
import os
import argparse
from rdkit import Chem
from rdkit import RDLogger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# Configure logging (disable RDKit warnings)
RDLogger.DisableLog('rdApp.*')

# Import necessary components from training code
from train import KCATDataset, generate_mol_feature
from KCM import EitlemKcatPredictor

# Define default SMILES string used when processing fails
DEFAULT_SMILES = "O"


def load_model(model_path, device, mol_feature_dim=167):
    """Load trained model"""
    model = EitlemKcatPredictor(
        mol_in_dim=mol_feature_dim,
        hidden_dim=512,
        protein_dim=1280,
        layer=10,
        dropout=0.5,
        att_layer=10
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, data_loader, device):
    """Make predictions using the model"""
    model.eval()
    pred_list = []
    target_list = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch)
            pred_list.append(pred.cpu().numpy())
            target_list.append(batch.y.cpu().numpy())

    # Concatenate results from all batches
    y_pred = np.concatenate(pred_list).flatten()
    y_true = np.concatenate(target_list).flatten()
    return y_pred, y_true


def calculate_metrics(y_pred, y_true):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pcc, p_value = pearsonr(y_true, y_pred)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "PCC": pcc,
        "P-value": p_value
    }


def process_complex_smiles(smiles):
    """
    Process SMILES:
    1. Keep complete substrate SMILES (do not split complexes)
    2. Use default SMILES only if full SMILES is invalid
    """
    # Handle null values
    if pd.isna(smiles):
        print("Detected null SMILES, using default value")
        return DEFAULT_SMILES

    # Ensure input is string
    smiles = str(smiles)

    # Directly validate full SMILES (do not split complexes)
    if is_valid_smiles(smiles):
        return smiles
    else:
        print(f"Invalid full SMILES, using default value: {DEFAULT_SMILES[:50]}...")
        return DEFAULT_SMILES


def is_valid_smiles(smiles):
    """Check if SMILES is valid (can be parsed by RDKit)"""
    try:
        # Handle empty string
        if not smiles or pd.isna(smiles):
            return False
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except Exception as e:
        print(f"SMILES validation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test KCAT prediction model")
    parser.add_argument("--test_pkl", type=str, required=True, help="Path to test pkl file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Output directory for results")
    parser.add_argument("--mol_type", type=str, default="MACCSKeys", choices=["ECFP", "MACCSKeys", "RDKIT"],
                        help="Molecular feature type (must match training setting)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--log10", type=bool, default=True,
                        help="Whether KCAT is log10 transformed (must match training setting)")

    args = parser.parse_args()

    # Validate default SMILES
    if not is_valid_smiles(DEFAULT_SMILES):
        print(f"Warning: Default SMILES {DEFAULT_SMILES[:50]}... is invalid, may cause subsequent errors")
    else:
        print(f"Default SMILES validation passed: {DEFAULT_SMILES[:50]}...")

    # Device configuration
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set molecular feature dimension (consistent with training)
    mol_feature_dim = 167

    # Load data
    print(f"Loading test data from {args.test_pkl}")
    test_df = pd.read_pickle(args.test_pkl)
    original_size = len(test_df)
    print(f"Original test data size: {original_size}")

    # Check for 'reactant_smiles' column
    if 'reactant_smiles' not in test_df.columns:
        raise ValueError("'reactant_smiles' column not found in test data, please check data format")

    # Show first SMILES in original data for debugging
    if original_size > 0:
        print(f"First original SMILES: {test_df['reactant_smiles'].iloc[0][:100]}...")

    # -------------------------- SMILES Processing Logic --------------------------
    # Process SMILES (keep full substrate SMILES)
    print("Processing SMILES (keeping full substrate SMILES)...")
    # Apply SMILES processing function
    test_df['processed_substrate'] = test_df['reactant_smiles'].apply(process_complex_smiles)

    # Show first processed SMILES for debugging
    if len(test_df) > 0:
        print(f"First processed SMILES: {test_df['processed_substrate'].iloc[0][:100]}...")

    # Count samples using default SMILES
    default_count = sum(1 for s in test_df['processed_substrate'] if s == DEFAULT_SMILES)
    print(f"Samples using default SMILES: {default_count}/{original_size}")

    # Count complex SMILES (SMILES containing '.')
    complex_count = sum(1 for s in test_df['reactant_smiles'] if '.' in str(s))
    print(f"Complex SMILES count: {complex_count}/{original_size}")

    # Replace original column with processed SMILES
    test_df['reactant_smiles'] = test_df['processed_substrate']
    test_df = test_df.drop(columns=['processed_substrate'])  # Clean temporary column
    # -------------------------------------------------------------------------

    # Create test dataset and dataloader
    test_dataset = KCATDataset(
        test_df,  # Use all processed data, invalid entries replaced with default
        protein_feature_col='esm2_features',
        smile_col='reactant_smiles',
        kcat_col='log10kcat_max',
        mol_type=args.mol_type,
        log10=args.log10
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn
    )

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device, mol_feature_dim)

    # Start prediction
    print("Starting prediction...")
    y_pred, y_true = predict(model, test_loader, device)

    # Calculate evaluation metrics
    metrics = calculate_metrics(y_pred, y_true)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save prediction results with flags for default SMILES and complex SMILES
    result_df = pd.DataFrame({
        "true_kcat": y_true,
        "predicted_kcat": y_pred,
        "used_default_smiles": [s == DEFAULT_SMILES for s in test_df['reactant_smiles']],
        "is_complex_smiles": ['.' in str(s) for s in test_df['reactant_smiles']]
    })
    result_path = os.path.join(args.output_dir, "prediction_results.csv")
    result_df.to_csv(result_path, index=False)
    print(f"Prediction results saved to {result_path}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Complex SMILES ratio: {complex_count / original_size:.2%}\n")
        f.write(f"Default SMILES sample ratio: {default_count / original_size:.2%}\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")


if __name__ == "__main__":
    main()