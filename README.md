# MCKcat: Predicting enzyme turnover numbers and enabling rational enzyme evolution

## Project Overview
MCKcat is a deep learning-based tool for predicting enzyme catalytic efficiency (kcat). By integrating **reaction fingerprints (RXNFP)** and **protein sequence features (UniRef50)**, it enables accurate kcat prediction for enzyme-reaction pairs. The tool includes a complete workflow for model training, prediction, and performance evaluation. It is suitable for applications in enzyme engineering, metabolic pathway optimization, and related fields.

The core files of the project include model definition (`MCKcat_model.py`), training script (`train.py`), and prediction script (`predict.py`). It also supports result comparison with other state-of-the-art methods such as CatPred, CatPro, and TurNup.

## Directory Structure

```
MCKcat_github/
├── code/
│   ├── MCKcat/
│   │   ├── MCKcat_model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── requirements.txt
│   ├── CatPred/    # Comparative Methods
│   ├── CatPro/     # Comparative Methods
│   ├── EITLEM/     # Comparative Methods
│   └── TurNup/     # Comparative Methods
└── data/
    ├── catpred_catpro/
    ├── mutation/
    └── turnup/
```

## Core Dependencies
Before running MCKcat, ensure the following dependencies are installed (Python 3.8+ is recommended):

```bash
# Basic data processing libraries
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
scipy==1.10.1

# Deep learning frameworks
torch==2.0.1
transformers==4.28.1

# Cheminformatics tools
rdkit-pypi==2023.3.3
rxnfp==1.2.1

# Utility libraries
tqdm==4.65.0
gzip==3.3.4
pickle-mixin==1.0
```

Quick installation command:
```bash
pip install -r requirements.txt
```

---

## 1. Model Training

### Prerequisites
Install dependencies first:
```bash
pip install -r requirements.txt
```

### Configure Training Data Paths
Before training, set your training and testing data paths in `train.py`:

```python
# Configure file paths
TRAIN_CSV = r"../data/TurNup_DB/cold_enzyme/1/train.csv"   # Training data CSV
TEST_CSV = r"../data/TurNup_DB/cold_enzyme/1/test.csv"     # Test data CSV
CACHE_FILE = "features_cache.pkl.gz"                       # Feature cache file
MODEL_SAVE_DIR = "models_MCKcat"                           # Model save directory
```

### Parameter Adjustment (Optional)
Modify the `CFG` class in `train.py` based on your hardware:

```python
class CFG:
    DEVICE = torch.device('cuda:0')  # Training device (cuda/cpu)
    EPOCHES = 40                     # Number of epochs
    lr = 5e-5                        # Learning rate
    batch_size = 1                   # Batch size (1-8 recommended for GPU)
    dtype = torch.float32            # Data precision
```
### Start Training
1. Update data paths and output directory in `train.py`.
2. Run:

```bash
python train.py
```

## 2. Model Prediction (Testing)

### Prerequisites
- Pre-trained models (from training, stored in `models_MCKcat/`).
- Prepare prediction CSV file with **2 mandatory columns** (1 optional):
  - Mandatory: `sequence`, `reaction_smiles`
  - Optional: `real_log10kcat_max` (true value for performance evaluation)

### Parameter Configuration
Modify prediction parameters in `predict.py` (under `if __name__ == "__main__":`):

```python
PREDICT_CSV = "prediction_data.csv"          # Path to your prediction data
CACHE_FILE = "features_cache.pkl.gz"         # Feature cache path
MODEL_SAVE_DIR = "models_MCKcat"             # Pre-trained model directory
OUTPUT_CSV = "final_predict_results.csv"     # Path to save results

# Load 10 pre-trained models (adjust seeds if needed)
MODEL_PATHS = [os.path.join(MODEL_SAVE_DIR, f"modelname.pth") for seed in range(10)]
```
### Start Prediction
Run:
```bash
python predict.py
```
