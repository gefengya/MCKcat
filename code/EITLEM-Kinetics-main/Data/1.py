import pandas as pd
import torch
import os
from pathlib import Path
from collections import defaultdict

def process_two_csv(train_csv, test_csv, output_dir):
    """
    Process the two CSV files (training and test) and generate:
    - Shared: index_seq (protein ID -> sequence), index_smiles (molecule ID -> substrate SMILES), seq_str.fasta
    - Separately: KCATTrainPairInfo, KCATTestPairInfo
    """
    # 1. Create the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 2. Read the two CSV files (handle encoding issues)
    def read_csv_safe(path):
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")

    df_train = read_csv_safe(train_csv)
    df_test = read_csv_safe(test_csv)
    print(f"Loaded training set: {len(df_train)} entries, test set: {len(df_test)} entries")

    # 3. Check whether the required columns exist
    required_cols = ["sequence", "uniprot", "log10kcat_max", "substrate"]
    missing_train = [col for col in required_cols if col not in df_train.columns]
    missing_test = [col for col in required_cols if col not in df_test.columns]
    if missing_train or missing_test:
        print(f"[ERROR] Missing columns in the training set: {missing_train}")
        print(f"[ERROR] Missing columns in the test set: {missing_test}")
        return

    # 4. Merge the training and test sets to build unified protein and molecule ID mappings
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # 4.1 Deduplicate proteins (key: replace empty uniprot with a space to ensure consistent mapping)
    # Handle empty uniprot values in the merged table: nan -> space
    df_combined["uniprot_filled"] = df_combined["uniprot"].fillna(" ")
    # Generate the unique identifier column (using the processed uniprot_filled)
    df_combined["pro_unique_key"] = df_combined["sequence"] + "_" + df_combined["uniprot_filled"]
    # Deduplicate and keep the key columns
    unique_proteins = df_combined.drop_duplicates("pro_unique_key")[
        ["sequence", "uniprot_filled", "pro_unique_key"]
    ].reset_index(drop=True)
    # Build the protein ID mapping
    pro_id_map = {row["pro_unique_key"]: idx for idx, row in unique_proteins.iterrows()}
    print(f"Number of proteins after merging and deduplication: {len(unique_proteins)}")

    # 4.2 Deduplicate molecules (substrates)
    unique_substrates = df_combined["substrate"].drop_duplicates().reset_index(drop=True)
    smi_id_map = {smiles: idx for idx, smiles in unique_substrates.items()}
    print(f"Number of substrates after merging and deduplication: {len(unique_substrates)}")

    # 5. Generate the training set pair information KCATTrainPairInfo
    train_pair_info = []
    for _, row in df_train.iterrows():
        # Handle empty uniprot values in the current row: nan -> space
        uniprot_val = row["uniprot"] if pd.notna(row["uniprot"]) else " "
        pro_key = str(row["sequence"]) + "_" + str(uniprot_val)
        pro_id = pro_id_map[pro_key]
        smi_id = smi_id_map[row["substrate"]]
        kcat_value = row["log10kcat_max"]
        train_pair_info.append([pro_id, smi_id, kcat_value, [0]])
    train_path = output_dir / "KCATTrainPairInfo"
    torch.save(train_pair_info, train_path)
    print(f"[DONE] Training set pair information: {train_path} ({len(train_pair_info)} entries)")

    # 6. Generate the test set pair information KCATTestPairInfo
    test_pair_info = []
    for _, row in df_test.iterrows():
        # Handle empty uniprot values in the current row: nan -> space (same logic as the merged table)
        uniprot_val = row["uniprot"] if pd.notna(row["uniprot"]) else " "
        pro_key = str(row["sequence"]) + "_" + str(uniprot_val)
        pro_id = pro_id_map[pro_key]
        smi_id = smi_id_map[row["substrate"]]
        kcat_value = row["log10kcat_max"]
        test_pair_info.append([pro_id, smi_id, kcat_value, [0]])
    test_path = output_dir / "KCATTestPairInfo"
    torch.save(test_pair_info, test_path)
    print(f"[DONE] Test set pair information: {test_path} ({len(test_pair_info)} entries)")

    # 7. Generate the shared index_seq (protein ID -> sequence)
    index_seq = {idx: row["sequence"] for idx, row in unique_proteins.iterrows()}
    seq_path = output_dir / "index_seq"
    torch.save(index_seq, seq_path)
    print(f"[DONE] Shared protein mapping: {seq_path} ({len(index_seq)} entries)")

    # 8. Generate the shared index_smiles (molecule ID -> substrate SMILES)
    index_smiles = {idx: smiles for idx, smiles in unique_substrates.items()}
    smiles_path = output_dir / "index_smiles"
    torch.save(index_smiles, smiles_path)
    print(f"[DONE] Shared substrate mapping: {smiles_path} ({len(index_smiles)} entries)")

    # 9. Generate the shared seq_str.fasta
    fasta_path = output_dir / "seq_str.fasta"
    with open(fasta_path, "w", encoding="utf-8") as f:
        for idx, row in unique_proteins.iterrows():
            f.write(f">{idx}\n{row['sequence']}\n")
    print(f"[DONE] Protein FASTA: {fasta_path} ({len(unique_proteins)} entries)")

    # 10. Conversion summary
    print("\n=== Conversion completed ===")
    print(f"Training set pairs: {len(train_pair_info)} entries | Test set pairs: {len(test_pair_info)} entries")
    print(f"Shared proteins: {len(index_seq)} entries | Shared substrates: {len(index_smiles)} entries")

if __name__ == "__main__":
    for i in range(1, 2):
        TRAIN_CSV = f"/mnt/usb3/code/gfy/code/catpred_pipeline/data/CatPred-DB/data/kcat/x_mei/kcat-random_train_en.csv"
        TEST_CSV = f"/mnt/usb3/code/gfy/code/catpred_pipeline/data/CatPred-DB/data/kcat/x_mei/kcat-random_test.csv"  # Confirm the test set path is correct
        OUTPUT_DIR = f"/mnt/usb3/code/gfy/code/EITLEM-Kinetics-main/Data/data/turnup/cold_enzyme_test/Feature/{i}"

        # Check whether the files exist
        for path in [TRAIN_CSV, TEST_CSV]:
            if not os.path.exists(path):
                print(f"[ERROR] Error: file does not exist -> {path}")
                exit(1)
        process_two_csv(TRAIN_CSV, TEST_CSV, OUTPUT_DIR)
