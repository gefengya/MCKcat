import pandas as pd
import numpy as np
import pickle
import gzip
import os
import re
import torch
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from transformers import T5Tokenizer, T5EncoderModel
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
)
from tqdm import tqdm
import hashlib
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from sklearn.model_selection import train_test_split
import torch.nn as nn
from os.path import join

# Set up computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "Must have available gpu"

# Global variable for model caching - ensure models are loaded only once
MODEL_CACHE = {
    't5_tokenizer': None,
    't5_model': None,
    'rxnfp_generator': None,
    'models_loaded': False  # New flag to track if models are loaded
}


def init_models(force_reload=False):
    """Initialize required models with caching, load only once (unless forced to reload)"""
    # Return directly if models are already loaded and no force reload
    if MODEL_CACHE['models_loaded'] and not force_reload:
        return

    print("Initializing pretrained models (loaded only once)...")

    if MODEL_CACHE['t5_tokenizer'] is None or MODEL_CACHE['t5_model'] is None or force_reload:
        print("Loading protein feature model...")
        tokenizer_path = "prot_t5_xl_uniref50"
        model_path = "prot_t5_xl_uniref50"

        MODEL_CACHE['t5_tokenizer'] = T5Tokenizer.from_pretrained(
            tokenizer_path, do_lower_case=False
        )
        MODEL_CACHE['t5_model'] = T5EncoderModel.from_pretrained(model_path)
        MODEL_CACHE['t5_model'] = MODEL_CACHE['t5_model'].to(device)
        MODEL_CACHE['t5_model'].eval()

    if MODEL_CACHE['rxnfp_generator'] is None or force_reload:
        print("Loading reaction fingerprint model...")
        model, tokenizer = get_default_model_and_tokenizer()
        MODEL_CACHE['rxnfp_generator'] = RXNBERTFingerprintGenerator(model, tokenizer)

    # Mark models as loaded
    MODEL_CACHE['models_loaded'] = True
    print("All pretrained models loaded successfully")


def generate_key(sequence, reaction_smiles):
    """Generate unique key for cache lookup"""
    combined = f"{sequence}||{reaction_smiles}"
    return hashlib.md5(combined.encode()).hexdigest()


def load_cache(cache_path):
    """Load cache file, create new one if it doesn't exist"""
    if os.path.exists(cache_path):
        print(f"Loading cache file: {cache_path}")
        with gzip.open(cache_path, 'rb') as f:
            try:
                return pickle.load(f)
            except:
                print("Cache file corrupted, creating new cache")
                return {}
    return {}


def save_cache(new_cache, cache_path):
    """Save cache to compressed file in append mode with maximum compression level"""
    # First read existing cache
    existing_cache = load_cache(cache_path)

    # Merge cache: entries in new cache overwrite those with the same name in old cache
    merged_cache = {**existing_cache, **new_cache}

    # Save merged complete cache
    with gzip.open(cache_path, 'wb', compresslevel=9) as f:
        pickle.dump(merged_cache, f)
    print(f"Cache appended and saved to: {cache_path}, total entries: {len(merged_cache)}")


def Seq_to_uniref50_dim1024(sequence):
    """Calculate uniref50_dim1024 features for protein sequence, use float32 to reduce memory usage"""
    max_len = 1000

    # Handle overlong sequences
    if len(sequence) > max_len:
        sequence = sequence[:500] + sequence[-500:]

    # Format sequence
    formatted_seq = ' '.join(list(sequence))
    formatted_seq = [re.sub(r"[UZOB]", "X", formatted_seq)]

    # Encode sequence
    ids = MODEL_CACHE['t5_tokenizer'].batch_encode_plus(
        formatted_seq, add_special_tokens=True, padding=True
    )
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = MODEL_CACHE['t5_model'](
            input_ids=input_ids, attention_mask=attention_mask
        )

    # Process embedding results, convert to float32 to reduce memory usage
    embedding = embedding.last_hidden_state.cpu().numpy().astype(np.float32)
    seq_len = (attention_mask[0] == 1).sum()
    seq_emd = embedding[0][:seq_len - 1]  # Extract valid sequence

    return seq_emd


def calculate_rxnfp(reaction_smiles):
    """Calculate RXNFP fingerprint for reaction SMILES, use float32 to reduce memory usage"""
    try:
        # Generate fingerprint
        fp = MODEL_CACHE['rxnfp_generator'].convert(reaction_smiles)

        # Ensure result is numpy array
        if isinstance(fp, list):
            fp = np.array(fp, dtype=np.float32)  # Convert to float32 first to ensure precision
        elif not isinstance(fp, np.ndarray):
            raise TypeError(f"Expected list or numpy array, got {type(fp)} instead")

        # Convert to float32 and return
        return fp.astype(np.float32)
    except Exception as e:
        print(
            f"Error calculating reaction fingerprint: {e}, SMILES: {reaction_smiles[:50]}...")  # Print partial SMILES for debugging
        return None

def process_csv_with_cache(csv_path, cache_path,
                           sequence_col='sequence',
                           reaction_col='reaction_smiles',
                           output_col='log10kcat_max',
                           output_csv=None):
    """Process CSV file with cache mechanism for feature calculation management"""
    # Ensure models are initialized (but not reloaded)
    init_models()

    # Load data, only read required columns to reduce memory usage
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path, usecols=[sequence_col, reaction_col, output_col])

    # Check if required columns exist
    required_cols = [sequence_col, reaction_col, output_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV file missing required column: {col}")

    # Load existing cache
    existing_cache = load_cache(cache_path)
    print(f"Cache already contains {len(existing_cache)} records")

    # Prepare to store results and newly calculated cache entries
    results = {
        'sequence': [],
        'reaction_smiles': [],
        'log10kcat_max': [],
        'rxnfp': [],
        'uniref50_dim1024': []
    }
    new_cache = {}  # Only store newly calculated entries
    new_entries = 0
    failed_smiles = []  # Record failed SMILES for subsequent analysis

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        sequence = str(row[sequence_col]).strip()
        reaction_smiles = str(row[reaction_col]).strip()

        # Filter invalid SMILES (empty or only whitespace)
        if not reaction_smiles.strip():
            print(f"Reaction SMILES in row {idx} is empty, skipping")
            continue

        # Generate unique key
        key = generate_key(sequence, reaction_smiles)

        # Check cache
        if key in existing_cache:
            # Use cached data
            results['sequence'].append(sequence)
            results['reaction_smiles'].append(reaction_smiles)
            results['log10kcat_max'].append(row[output_col])
            results['rxnfp'].append(existing_cache[key]['rxnfp'])
            results['uniref50_dim1024'].append(existing_cache[key]['uniref50_dim1024'])
        else:
            # Calculate new features
            try:
                # Calculate protein features
                uniref50 = Seq_to_uniref50_dim1024(sequence)

                # Calculate reaction fingerprint
                rxnfp = calculate_rxnfp(reaction_smiles)

                if rxnfp is not None:
                    # Store results
                    results['sequence'].append(sequence)
                    results['reaction_smiles'].append(reaction_smiles)
                    results['log10kcat_max'].append(row[output_col])
                    results['rxnfp'].append(rxnfp)
                    results['uniref50_dim1024'].append(uniref50)

                    # Add to new cache
                    new_cache[key] = {
                        'rxnfp': rxnfp,
                        'uniref50_dim1024': uniref50
                    }
                    new_entries += 1
                else:
                    print(f"Cannot calculate reaction fingerprint for row {idx}, skipping")
                    failed_smiles.append((idx, reaction_smiles))
            except Exception as e:
                print(f"Error processing row {idx}: {e}, skipping")
                failed_smiles.append((idx, reaction_smiles))

    # Save failed SMILES for analysis
    if failed_smiles:
        failed_df = pd.DataFrame(failed_smiles, columns=['index', 'reaction_smiles'])
        failed_df.to_csv('failed_reaction_smiles.csv', index=False)
        print(f"Saved {len(failed_smiles)} failed reaction SMILES to failed_reaction_smiles.csv")

    # Append and save new cache (if there are new entries)
    if new_entries > 0:
        print(f"Added {new_entries} new records to cache")
        save_cache(new_cache, cache_path)

    # Convert results to DataFrame
    result_df = pd.DataFrame(results)

    # Save processed CSV (if specified)
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"Processed CSV saved to: {output_csv}")

    return result_df


# Model training related code
def seed_everything(seed=42):
    """Set random seeds to ensure reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFG:
    """Configuration parameter class"""
    gpu_number = torch.cuda.device_count()
    DEVICE = torch.device('cuda:0')
    EPOCHES = 40
    lr = 5e-5
    weight_decay = 1e-5
    batch_size = 1  # Adjust according to memory
    dtype = torch.float32  # Use float32 for training


class CustomDataset(Dataset):
    """Custom dataset class for data loading and padding"""

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)  # Reset index
        self.max_len = 1000  # Maximum length of protein sequence

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get data by position from DataFrame (use iloc to avoid index issues)
        row = self.dataframe.iloc[idx]

        # Process reaction features (use float32)
        reaction = torch.tensor(row['rxnfp'], dtype=CFG.dtype).unsqueeze(0)

        # Process label (labels usually retain float32 to maintain precision)
        label = torch.tensor(row['log10kcat_max'], dtype=torch.float32).unsqueeze(0)

        # Process protein features (use float32)
        protein = torch.tensor(row['uniref50_dim1024'], dtype=CFG.dtype)

        return reaction, protein, label


def find_column(df, target_name):
    """Find matching column name in DataFrame (case-insensitive)"""
    target_lower = target_name.lower()
    for col in df.columns:
        if col.lower() == target_lower:
            return col
    raise ValueError(f"Matching column name not found: {target_name} (candidate columns: {list(df.columns)})")


def train_model(model, train_data, test_data, CFG, seed, save_dir):
    """Train model and save both best model and last epoch model"""
    # Convert model to float32 mode
    model = model.to(CFG.dtype).to(CFG.DEVICE)
    model.train()

    # Use precision matching data for training
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    criterion = nn.MSELoss()

    # Create DataLoader with batch_size and shuffle
    train_dataloader = DataLoader(
        train_data,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=4,  # Adjust according to CPU cores
        pin_memory=True  # Accelerate GPU data transfer
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Use list to store historical data
    history = []
    best_r2 = -float('inf')

    # Define save paths for best model and last epoch model
    best_model_path = os.path.join(save_dir, f'model_seed_{seed}_best.pth')
    last_model_path = os.path.join(save_dir, f'model_seed_{seed}_last.pth')

    for epoch in range(CFG.EPOCHES):
        # Training phase
        loss_total = 0.0
        count = 0
        for reaction, protein, label in train_dataloader:
            reaction = reaction.to(CFG.DEVICE)
            protein = protein.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)

            # Mixed precision training: convert inputs to float32, keep labels as float32
            with torch.cuda.amp.autocast():
                output, _ = model(reaction, protein)
                loss = criterion(output.float(), label)  # Ensure loss calculation in float32

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            count += 1

        train_loss = loss_total / count
        print(f'Epoch {epoch + 1}/{CFG.EPOCHES} - Train Loss: {train_loss:.4f}')

        # Evaluation phase
        test_loss, mse, r2, pearson_corr, mae = evaluate_model(model, test_dataloader, criterion, CFG)

        # Save best model
        if r2 > best_r2:
            print(f'R² improved from {best_r2:.4f} to {r2:.4f}. Saving best model...')
            best_r2 = r2
            torch.save(model.state_dict(), best_model_path)

        # Record training history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'mse': mse,
            'r2': r2,
            'pearson_corr': pearson_corr,
            'mae': mae
        })

        # Save last epoch model if it's the final epoch
        if epoch == CFG.EPOCHES - 1:
            print(f'Saving last epoch ({CFG.EPOCHES}) model...')
            torch.save(model.state_dict(), last_model_path)

    # Convert to DataFrame after training
    history_df = pd.DataFrame(history)
    # Save training history
    history_df.to_csv(os.path.join(save_dir, f'training_history_seed_{seed}.csv'), index=False)
    print(f'Training completed for seed {seed}. Best R²: {best_r2:.4f}')
    return best_model_path, last_model_path


def evaluate_model(model, test_dataloader, criterion, CFG):
    """Evaluate model and return performance metrics"""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for reaction, protein, label in test_dataloader:
            reaction = reaction.to(CFG.DEVICE)
            protein = protein.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)

            with torch.cuda.amp.autocast():
                outputs, _ = model(reaction, protein)

            loss = criterion(outputs.float(), label)
            test_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    pearson_corr, _ = pearsonr(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)

    print(f'Test Loss: {test_loss / len(test_dataloader):.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'Pearson Correlation: {pearson_corr:.4f}')
    print(f'MAE: {mae:.4f}')

    return test_loss / len(test_dataloader), mse, r2, pearson_corr, mae


def predict_with_ensemble(models, test_data, CFG):
    """Make predictions using ensemble models"""
    all_labels = []
    ensemble_preds = []
    all_sequences = []
    all_reaction_smiles = []

    test_dataloader = DataLoader(
        test_data,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_data_df = test_data.dataframe

    # Make predictions with each model
    model_preds_list = []
    for model_path in models:
        model = Model_Regression().to(CFG.dtype).to(CFG.DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        model_preds = []
        with torch.no_grad():
            for reaction, protein, _ in test_dataloader:
                reaction = reaction.to(CFG.DEVICE)
                protein = protein.to(CFG.DEVICE)

                with torch.cuda.amp.autocast():
                    outputs, _ = model(reaction, protein)

                model_preds.append(outputs.cpu().numpy())

        model_preds = np.concatenate(model_preds).ravel()
        model_preds_list.append(model_preds)

    # Ensemble prediction (simple average)
    ensemble_preds = np.mean(model_preds_list, axis=0)

    return all_labels, ensemble_preds, all_sequences, all_reaction_smiles


if __name__ == "__main__":
    # Import model (select according to actual situation)
    from MCKcat_model import Model_Regression

    # Configure file paths
    TRAIN_CSV = r"../data/TurNup_DB/cold_enzyme/1/test.csv/train.csv"  # Training data CSV
    TEST_CSV = r"../data/TurNup_DB/cold_enzyme/1/test.csv"  # Test data CSV
    CACHE_FILE = "features_cache.pkl.gz"  # Feature cache file
    MODEL_SAVE_DIR = "models_MCKcat"  # Model save directory

    # Pre-initialize all models (loaded only once)
    init_models()

    # Process training data - reuse loaded models
    print("Processing training data...")
    train_df = process_csv_with_cache(
        csv_path=TRAIN_CSV,
        cache_path=CACHE_FILE,
        sequence_col='sequence',
        reaction_col='reaction_smiles',
        output_col='log10kcat_max',
        output_csv="dd.csv"
    )

    # Process test data - reuse loaded models without reloading
    print("Processing test data...")
    test_df = process_csv_with_cache(
        csv_path=TEST_CSV,
        cache_path=CACHE_FILE,  # Use the same cache file
        sequence_col='sequence',
        reaction_col='reaction_smiles',
        output_col='log10kcat_max',
        output_csv="kk.csv"
    )

    # Create datasets
    train_data = CustomDataset(train_df)
    test_data = CustomDataset(test_df)

    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")

    # Ensure save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Train models with multiple seeds
    trained_best_models = []
    trained_last_models = []
    num_seeds = 10
    for seed in range(num_seeds):
        print(f"\n===== Starting model training for seed {seed} =====")
        seed_everything(37 + seed)  # Set seed
        model = Model_Regression()
        best_model_path, last_model_path = train_model(model, train_data, test_data, CFG, seed, MODEL_SAVE_DIR)
        trained_best_models.append(best_model_path)
        trained_last_models.append(last_model_path)
        print(f"Training completed for seed {seed}, best model saved to: {best_model_path}")
        print(f"Training completed for seed {seed}, last epoch model saved to: {last_model_path}")

    print("All models trained successfully!")