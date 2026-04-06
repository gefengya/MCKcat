import torch
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress bar

# ---------------------- Configuration ----------------------
BATCH_SIZE = 1  # Batch size (adjust based on GPU memory, 50-200 recommended)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LENGTH = 1022  # Max sequence length supported by ESM-1b

# ---------------------- Load Model ----------------------
# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model = model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()


# ---------------------- Batch Feature Extraction Function ----------------------
def batch_calculate_esm_features(sequences):
    """
    Compute ESM-1b features in batch (GPU accelerated)
    Args:
        sequences: list of protein sequences (strings)
    Returns:
        features: numpy array with shape (len(sequences), 1280)
    """
    batch_labels = [f"seq_{i}" for i in range(len(sequences))]
    _, _, batch_tokens = batch_converter(list(zip(batch_labels, sequences)))
    batch_tokens = batch_tokens.to(DEVICE)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    features = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        token_rep = token_representations[i, 1: seq_len + 1]
        seq_rep = token_rep.mean(dim=0).cpu().numpy()
        features.append(seq_rep)

    return np.array(features)


# ---------------------- Main Pipeline ----------------------
def main():
    # Process data
    for i in range(1, 2):
        for j in range(6, 7):
            SAVE_PATH = f"/mnt/usb/code/gfy/MyModel/HIS7_esm.pkl"
            df = pd.read_pickle(f"/mnt/usb/code/gfy/MyModel/HIS7.pkl")

            if "sequence" not in df.columns:
                raise ValueError("Missing 'sequence' column")

            # Check sequence lengths
            seq_lengths = df["sequence"].str.len()
            too_long = seq_lengths > MAX_SEQ_LENGTH
            num_too_long = too_long.sum()

            if num_too_long > 0:
                print(f"Warning: {num_too_long} sequences exceed ESM-1b max length ({MAX_SEQ_LENGTH} aa)")
                print(f"Longest sequence: {seq_lengths.max()}")
                print("These sequences will be truncated")

            # Truncate long sequences
            df["sequence"] = df["sequence"].apply(lambda x: x[:MAX_SEQ_LENGTH] if len(x) > MAX_SEQ_LENGTH else x)

            # Initialize feature list
            esm_features = []
            total_sequences = len(df["sequence"])
            print(f"Processing {total_sequences} sequences, batch size={BATCH_SIZE}, device={DEVICE}")

            # Process in batches
            for k in tqdm(range(0, total_sequences, BATCH_SIZE), desc="Progress"):
                batch_sequences = df["sequence"].iloc[k: k + BATCH_SIZE].tolist()
                batch_features = batch_calculate_esm_features(batch_sequences)
                esm_features.extend(batch_features)

            esm_features = np.array(esm_features)
            print(f"Feature extraction finished, shape: {esm_features.shape}")

            # Add to DataFrame
            df["ESM1b"] = esm_features.tolist()

            # Save result
            df.to_pickle(SAVE_PATH)
            print(f"Processing done! Results saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()