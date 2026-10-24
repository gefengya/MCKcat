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
│   ├── EITLEM-Kinetics-main/     # Comparative Methods
│   └── TurNup/     # Comparative Methods
└── data/
    ├── catpred_catpro/
    ├── mutation/
    └── turnup/
```

## Core Dependencies
Before running MCKcat, we recommend creating a dedicated conda environment with **Python 3.9**.
```bash
# Create and activate conda environment
conda create -n mckcat python=3.9 -y
conda activate mckcat
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
