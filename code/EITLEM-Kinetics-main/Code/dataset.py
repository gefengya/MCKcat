import torch
import os
import math
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np

# NOTE: do not initialize RDKit objects in the global scope!


class EitlemDataSet(Dataset):
    def __init__(self, Pairinfo, ProteinsPath, Smiles, nbits, radius, log10, Type='ECFP'):
        super(EitlemDataSet, self).__init__()
        self.pairinfo = Pairinfo
        self.smiles = torch.load(Smiles) if isinstance(Smiles, str) else Smiles
        self.seq_path = os.path.join(ProteinsPath, '{}.pt')
        self.nbits = nbits
        self.radius = radius
        self.log10 = log10
        self.Type = Type
        # Key: do not initialize RDKit objects in __init__; defer them to __getitem__ (inside a child process)
        
    def __getitem__(self, idx):
        # Import and initialize RDKit only inside the child process (avoids serialization in the main process)
        from rdkit.Chem import AllChem, MACCSkeys
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        # Initialize fpgen inside the child process (it is not passed across processes)
        fpgen = AllChem.GetRDKitFPGenerator(fpSize=self.nbits) if self.Type in ["RDKIT", "MACCSKeys_RDKIT"] else None
        
        # Original logic
        pro_id = self.pairinfo[idx][0]
        smi_id = self.pairinfo[idx][1]
        value = self.pairinfo[idx][2]
        protein_emb = torch.load(self.seq_path.format(pro_id))
        
        # Create the Mol object (created inside the child process, not serialized)
        mol = Chem.MolFromSmiles(self.smiles[smi_id].strip())
        if mol is None:
            # Replace invalid SMILES with a zero vector
            if self.Type == "MACCSKeys":
                fp = [0]*167
            elif self.Type == "MACCSKeys_RDKIT":
                fp1, fp2 = [0]*167, [0]*self.nbits
            else:
                fp = [0]*self.nbits
        else:
            # Generate fingerprints (handled inside the child process; Mol objects are not passed)
            if self.Type == "ECFP":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.nbits).ToList()
            elif self.Type == "MACCSKeys":
                fp = MACCSkeys.GenMACCSKeys(mol).ToList()
            elif self.Type == "RDKIT":
                fp = fpgen.GetFingerprint(mol).ToList()
            elif self.Type == "MACCSKeys_RDKIT":
                fp1 = MACCSkeys.GenMACCSKeys(mol).ToList()
                fp2 = fpgen.GetFingerprint(mol).ToList()
        
        # Build the Data object (contains only tensors, no RDKit objects)
        if self.Type == "MACCSKeys_RDKIT":
            return Data(
                x=torch.FloatTensor(fp1).unsqueeze(0),
                y=torch.FloatTensor(fp2).unsqueeze(0),
                pro_emb=protein_emb,
                value=torch.FloatTensor([value])
            )
        else:
            return Data(
                x=torch.FloatTensor(fp).unsqueeze(0),
                pro_emb=protein_emb,
                value=torch.FloatTensor([value])
            )
    
    def collate_fn(self, batch):
        return Batch.from_data_list(batch, follow_batch=['pro_emb'])
    
    def __len__(self):
        return len(self.pairinfo)


class EitlemDataLoader(DataLoader):
    def __init__(self, data,** kwargs):
        # Force the spawn start method to avoid inheritance issues caused by fork
        super().__init__(data, collate_fn=data.collate_fn, multiprocessing_context='spawn', **kwargs)


# Other functions remain unchanged
def shuffle_dataset(dataset):
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]