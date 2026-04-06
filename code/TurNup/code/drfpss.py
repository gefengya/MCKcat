import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from drfp import DrfpEncoder
import dill
import pandas as pd
from os.path import join

# Define a function to process data
def process_data(file_path, save_path):
    # Load data
    all_data = pd.read_pickle(file_path)

    print(f"data_len: {len(all_data)}")
    print(f"File loaded into data: {file_path}")

    drfp_list = list()
    encoder = DrfpEncoder()

    # Compute DRFP fingerprints
    for reaction_smiles in all_data['reaction_smiles']:
        # Encode DRFP fingerprint
        drfp = encoder.encode([reaction_smiles])[0]
        drfp_list.append(drfp)

    # Add computed column to the original dataframe
    all_data['drfp'] = drfp_list

    # Save data with DRFP fingerprints
    all_data.to_pickle(save_path)


numbers = [1, 4, 6, 16, 9]
for i in range(1, 2):
    for j in range(1, 2):
        test_file_path = f"/mnt/usb/code/gfy/MyModel/HIS7_esm.pkl"
        test_save_path = f"/mnt/usb/code/gfy/MyModel/HIS7_esm_with_drfp.pkl"
        process_data(test_file_path, test_save_path)