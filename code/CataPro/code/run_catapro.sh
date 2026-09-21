#!/bin/bash
# Integrated Dataset Training Script
# Supports: Warm (fold 1-5), Cold Enzyme (fold 1-5), Cold Reaction (fold 1-5), CatPred base dataset
# Data paths: ./data/turnup/... and ./data/catpred_catpro/...
# ========================= Global Training Configurations =========================
# Adjust parameters below as needed
BATCH_SIZE=8               # Batch size
DEVICE="cuda:0"            # Training device (cuda:0 or cpu)
TRAIN_SCRIPT="train.py"    # Path to training script (ensure it matches actual location)
MODEL_ROOT_DIR="models"    # Root directory for saving models (separate subfolders for each dataset)
DATA_ROOT_DIR="./data"     # Root data directory (matches your ./data folder structure)
# ========================= Dataset Configurations =========================
# Define all datasets to be trained (format: dataset_name:data_prefix:fold_range:train_file_template:test_file_template)
# Format explanation: "dataset name:data path prefix:fold range(e.g.1-5):train file template:test file template"
# Datasets without folds (e.g. CatPred base): leave fold range empty, file templates are fixed filenames
DATASETS=(
    # 1. Warm dataset (5 folds, file template: train_fold_6_with_sub_.csv / test_fold_6_with_sub_.csv)
    "warm:${DATA_ROOT_DIR}/turnup/warm:1-5:train_fold_6_with_sub_.csv:test_fold_6_with_sub_.csv"
    # 2. Cold Enzyme dataset (5 folds, same file templates above)
    "cold_enzyme:${DATA_ROOT_DIR}/turnup/cold_enzyme:1-5:train_fold_6_with_sub_.csv:test_fold_6_with_sub_.csv"
    # 3. Cold Reaction dataset (5 folds, same file templates above)
    "cold_reaction:${DATA_ROOT_DIR}/turnup/cold_reaction:1-5:train_fold_6_with_sub_.csv:test_fold_6_with_sub_.csv"
    # 4. CatPred base dataset (no folds, fixed filenames)
    "catpred_basic:${DATA_ROOT_DIR}/catpred_catpro::train.csv:Held-out test.csv"
)
# ========================= Utility Functions =========================
# Function: Train single fold of one dataset (or dataset without folds)
# Arguments: dataset name, fold number (none for datasets without folds), train file path, test file path
train_single_fold() {
    local dataset_name="$1"
    local fold="$2"
    local train_path="$3"
    local test_path="$4"
    # 1. Create model save directory (hierarchy: dataset/fold to avoid file conflicts)
    local model_subdir="${dataset_name}"
    if [ "$fold" != "none" ]; then
        model_subdir="${dataset_name}/fold_${fold}"
    fi
    local model_dir="${MODEL_ROOT_DIR}/${model_subdir}"
    mkdir -p "$model_dir"  # Ensure directory exists
    # 2. Print training info
    echo -e "\n=================================================="
    if [ "$fold" != "none" ]; then
        echo "【Start Training】Dataset: ${dataset_name} | Fold: ${fold}"
    else
        echo "【Start Training】Dataset: ${dataset_name} (no folds)"
    fi
    echo "Train file: $(realpath "$train_path")"
    echo "Test file: $(realpath "$test_path")"
    echo "Model save path: $(realpath "$model_dir")"
    echo "=================================================="
    # 3. Execute training command (arguments consistent with original script: -inp_fpath/-test_fpath etc.)
    python "$TRAIN_SCRIPT" \
        -inp_fpath "$train_path" \
        -model_dpath "$model_dir" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE"
    # 4. Check training exit status
    if [ $? -eq 0 ]; then
        echo -e "【Training Succeeded】Dataset: ${dataset_name} | Fold: ${fold} → Model saved at: ${model_dir}\n"
    else
        echo -e "【Training Failed】Dataset: ${dataset_name} | Fold: ${fold} → Please check logs for troubleshooting\n"
    fi
}
# ========================= Main Training Pipeline =========================
echo "=== Integrated Dataset Training Script Started ==="
echo "Global Params: Batch size=${BATCH_SIZE} | Device=${DEVICE} | Data root=${DATA_ROOT_DIR}"
echo "Model root dir: ${MODEL_ROOT_DIR} | Training script: ${TRAIN_SCRIPT}"
echo "Total datasets to train: ${#DATASETS[@]}\n"
# Iterate over all datasets and run training
for dataset_config in "${DATASETS[@]}"; do
    # Parse dataset configuration (split by colon ":")
    IFS=':' read -r ds_name ds_prefix fold_range train_template test_template <<< "$dataset_config"
    
    echo -e "================ Start processing dataset: ${ds_name} ================"
    
    # Two branches: datasets with folds (warm 1-5 folds), datasets without folds (catpred_basic)
    if [ -n "$fold_range" ]; then
        # 1. Datasets with folds: parse fold range (e.g.1-5), loop over each fold
        IFS='-' read -r start_fold end_fold <<< "$fold_range"
        echo "Dataset ${ds_name}: fold range ${start_fold}-${end_fold}, starting fold-wise training..."
        
        for ((i=start_fold; i<=end_fold; i++)); do
            # Build train/test file paths for current fold (replace fold placeholder in path)
            train_file="${ds_prefix}/${i}/${train_template}"
            test_file="${ds_prefix}/${i}/${test_template}"
            
            # Check existence of files; skip this fold if missing
            if [ ! -f "$train_file" ]; then
                echo "WARNING: Train file for dataset ${ds_name} fold ${i} not found → ${train_file}, skip"
                continue
            fi
            if [ ! -f "$test_file" ]; then
                echo "WARNING: Test file for dataset ${ds_name} fold ${i} not found → ${test_file}, skip"
                continue
            fi
            
            # Call function to train single fold
            train_single_fold "$ds_name" "$i" "$train_file" "$test_file"
        done
    else
        # 2. Dataset without folds (CatPred base): directly use fixed filenames
        train_file="${ds_prefix}/${train_template}"
        test_file="${ds_prefix}/${test_template}"
        
        # Check existence of files
        if [ ! -f "$train_file" ]; then
            echo "ERROR: Train file for dataset ${ds_name} not found → ${train_file}, skip this dataset"
            continue
        fi
        if [ ! -f "$test_file" ]; then
            echo "ERROR: Test file for dataset ${ds_name} not found → ${test_file}, skip this dataset"
            continue
        fi
        
        # Call training function (pass "none" as fold argument)
        train_single_fold "$ds_name" "none" "$train_file" "$test_file"
    fi
    
    echo -e "================ Dataset ${ds_name} processing finished ================\n"
done
# ========================= Training Summary =========================
echo "=== All dataset training tasks completed ==="
echo "Model root directory: $(realpath "${MODEL_ROOT_DIR}")"
echo "Hint: You may check training outputs in the above directory"
