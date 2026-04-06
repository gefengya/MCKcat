#!/bin/bash
# Integrated Multi-Dataset Full Pipeline (CSV→PKL→ESM Features→Model Training)
# Supported: Warm (Folds 1-5), Cold Enzyme (Folds 1-5), Cold Reaction (Folds 1-5), MCKcat Basic, mutation_DB single-enzyme datasets
# Data Paths: ./data/TurNup_DB/..., ./data/MCKcat_DB/..., ./data/mutation_DB/...
# Core Pipeline: Raw CSV → Convert to PKL (cache) → Compute ESM Features (cache) → Model Training

# ========================= Global Parameters =========================
# Adjustable for all datasets
BATCH_SIZE=64                # Training batch size
EPOCHS=100                   # Training epochs
LR=0.001                     # Learning rate
DEVICE=0                     # Device (0=GPU, cpu=CPU)
MOL_TYPE="MACCSKeys"         # Molecular feature type (matches train.py)
ESM_MODEL="esm2_t33_650M_UR50D"  # ESM model version
TRAIN_SCRIPT="train.py"      # Training script
ESM_SCRIPT="esm2.py"         # ESM feature extraction script
DATA_ROOT_DIR="./data"       # Root data directory
PKL_ROOT_DIR="./data/pkl_files"  # PKL cache directory
MODEL_ROOT_DIR="./models"     # Model save directory

# ========================= Dataset Configuration =========================
# Format: "dataset_name:data_prefix:fold_range:train_file:test_file"
# Leave fold_range empty for non-fold datasets
DATASETS=(
    # 1. Warm Dataset (Folds 1-5)
    "warm:${DATA_ROOT_DIR}/TurNup_DB/warm:1-5:train.csv:test.csv"
    # 2. Cold Enzyme Dataset (Folds 1-5)
    "cold_enzyme:${DATA_ROOT_DIR}/TurNup_DB/cold_enzyme:1-5:train.csv:test.csv"
    # 3. Cold Reaction Dataset (Folds 1-5)
    "cold_reaction:${DATA_ROOT_DIR}/TurNup_DB/cold_reaction:1-5:train.csv:test.csv"
    # 4. MCKcat Basic Dataset (no folds)
    "MCKcat_DB_basic:${DATA_ROOT_DIR}/MCKcat_DB::train.csv:Held-out test.csv"

)

# ========================= Utility Functions =========================
# 1. Convert CSV to PKL (cached)
csv_to_pkl() {
    local csv_path="$1"
    local pkl_path="$2"
    local dataset_name="$3"
    local fold="$4"

    if [ ! -f "$csv_path" ]; then
        echo "❌ ${dataset_name} (Fold ${fold}): CSV not found → ${csv_path}"
        return 1
    fi

    if [ -f "$pkl_path" ]; then
        echo "✅ ${dataset_name} (Fold ${fold}): PKL already exists → ${pkl_path##*/}"
        return 0
    fi

    echo "🔄 ${dataset_name} (Fold ${fold}): Converting CSV to PKL..."
    python -c "
import pandas as pd
import os

csv_path = '${csv_path}'
pkl_path = '${pkl_path}'

try:
    df = pd.read_csv(csv_path, sep=',')
except Exception as e:
    print(f'❌ Failed to read CSV: {e}')
    exit(1)

try:
    df.to_pickle(pkl_path, compression='gzip')
    print(f'✅ ${dataset_name} (Fold ${fold}): CSV to PKL success → {pkl_path##*/}')
except Exception as e:
    print(f'❌ Failed to save PKL: {e}')
    exit(1)
    "

    if [ $? -eq 0 ] && [ -f "$pkl_path" ]; then
        return 0
    else
        echo "❌ ${dataset_name} (Fold ${fold}): CSV to PKL failed"
        return 1
    fi
}

# 2. Compute ESM Features (cached)
compute_esm_features() {
    local input_pkl="$1"
    local output_pkl="$2"
    local dataset_name="$3"
    local fold="$4"

    if [ ! -f "$input_pkl" ]; then
        echo "❌ ${dataset_name} (Fold ${fold}): Input PKL not found → ${input_pkl##*/}"
        return 1
    fi

    if [ -f "$output_pkl" ]; then
        echo "✅ ${dataset_name} (Fold ${fold}): ESM PKL exists → ${output_pkl##*/}"
        return 0
    fi

    echo "🔄 ${dataset_name} (Fold ${fold}): Computing ESM features..."
    python "${ESM_SCRIPT}" \
        "${ESM_MODEL}" \
        "${input_pkl}" \
        "${output_pkl}"

    if [ $? -eq 0 ] && [ -f "$output_pkl" ]; then
        echo "✅ ${dataset_name} (Fold ${fold}): ESM features computed → ${output_pkl##*/}"
        return 0
    else
        echo "❌ ${dataset_name} (Fold ${fold}): ESM feature extraction failed"
        return 1
    fi
}

# 3. Train Model
train_model() {
    local train_pkl="$1"
    local test_pkl="$2"
    local output_dir="$3"
    local dataset_name="$4"
    local fold="$5"

    if [ ! -f "$train_pkl" ]; then
        echo "❌ ${dataset_name} (Fold ${fold}): Train ESM PKL not found → ${train_pkl##*/}"
        return 1
    fi
    if [ ! -f "$test_pkl" ]; then
        echo "❌ ${dataset_name} (Fold ${fold}): Test ESM PKL not found → ${test_pkl##*/}"
        return 1
    fi

    mkdir -p "$output_dir"
    echo -e "\n🚀 ${dataset_name} (Fold ${fold}): Starting model training → ${output_dir##*/}"

    python "${TRAIN_SCRIPT}" \
        --train1_pkl "${train_pkl}" \
        --test_pkl "${test_pkl}" \
        --output_dir "${output_dir}" \
        --mol_type "${MOL_TYPE}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --device "${DEVICE}"

    if [ $? -eq 0 ]; then
        echo -e "🎉 ${dataset_name} (Fold ${fold}): Training completed! Model saved → ${output_dir}\n"
        return 0
    else
        echo -e "❌ ${dataset_name} (Fold ${fold}): Training failed\n"
        return 1
    fi
}

# 4. Process One Full Dataset
process_dataset() {
    local dataset_config="$1"
    IFS=':' read -r ds_name ds_prefix fold_range train_template test_template <<< "$dataset_config"

    echo -e "\n=================================================="
    echo "Processing Dataset: ${ds_name}"
    echo "Data Path: ${ds_prefix}"
    echo "Folds: ${fold_range:-No folds}"
    echo "Train File: ${train_template}"
    echo "Test File: ${test_template}"
    echo "=================================================="

    local ds_pkl_dir="${PKL_ROOT_DIR}/${ds_name}"
    mkdir -p "$ds_pkl_dir"

    if [ -n "$fold_range" ]; then
        IFS='-' read -r start_fold end_fold <<< "$fold_range"
        echo -e "\n🔄 Processing ${ds_name} folds ${start_fold}-${end_fold}..."

        for ((fold=start_fold; fold<=end_fold; fold++)); do
            echo -e "\n--------------------------------------------------"
            echo "Processing ${ds_name} Fold ${fold}"
            echo "--------------------------------------------------"

            local csv_train="${ds_prefix}/${fold}/${train_template}"
            local csv_test="${ds_prefix}/${fold}/${test_template}"
            local pkl_train="${ds_pkl_dir}/train_fold_${fold}.pkl.gz"
            local pkl_test="${ds_pkl_dir}/test_fold_${fold}.pkl.gz"
            local esm_pkl_train="${ds_pkl_dir}/train_fold_${fold}_with_esm.pkl"
            local esm_pkl_test="${ds_pkl_dir}/test_fold_${fold}_with_esm.pkl"
            local model_dir="${MODEL_ROOT_DIR}/${ds_name}/fold_${fold}"

            csv_to_pkl "$csv_train" "$pkl_train" "$ds_name" "$fold" || continue
            csv_to_pkl "$csv_test" "$pkl_test" "$ds_name" "$fold" || continue

            compute_esm_features "$pkl_train" "$esm_pkl_train" "$ds_name" "$fold" || continue
            compute_esm_features "$pkl_test" "$esm_pkl_test" "$ds_name" "$fold" || continue

            train_model "$esm_pkl_train" "$esm_pkl_test" "$model_dir" "$ds_name" "$fold"
        done
    else
        echo -e "\n🔄 Processing ${ds_name} (No folds)..."

        local csv_train="${ds_prefix}/${train_template}"
        local csv_test="${ds_prefix}/${test_template}"
        local pkl_train="${ds_pkl_dir}/train.pkl.gz"
        local pkl_test="${ds_pkl_dir}/test.pkl.gz"
        local esm_pkl_train="${ds_pkl_dir}/train_with_esm.pkl"
        local esm_pkl_test="${ds_pkl_dir}/test_with_esm.pkl"
        local model_dir="${MODEL_ROOT_DIR}/${ds_name}"

        csv_to_pkl "$csv_train" "$pkl_train" "$ds_name" "none" || return 1
        csv_to_pkl "$csv_test" "$pkl_test" "$ds_name" "none" || return 1

        compute_esm_features "$pkl_train" "$esm_pkl_train" "$ds_name" "none" || return 1
        compute_esm_features "$pkl_test" "$esm_pkl_test" "$ds_name" "none" || return 1

        train_model "$esm_pkl_train" "$esm_pkl_test" "$model_dir" "$ds_name" "none"
    fi

    echo -e "=================================================="
    echo "${ds_name} dataset processing completed!"
    echo "=================================================="
}

# ========================= Main =========================
echo "=== Integrated Multi-Dataset Training Pipeline Started ==="
echo "Global Parameters:"
echo "  Batch: ${BATCH_SIZE} | Epochs: ${EPOCHS} | LR: ${LR}"
echo "  ESM: ${ESM_MODEL} | Feature: ${MOL_TYPE} | Device: ${DEVICE}"
echo "  Data Root: ${DATA_ROOT_DIR}"
echo "  PKL Cache: ${PKL_ROOT_DIR}"
echo "  Models: ${MODEL_ROOT_DIR}"
echo -e "Datasets to process: ${#DATASETS[@]}\n"

mkdir -p "${PKL_ROOT_DIR}"
mkdir -p "${MODEL_ROOT_DIR}"

for ds_config in "${DATASETS[@]}"; do
    process_dataset "$ds_config"
done

echo -e "\n=== All datasets processed successfully ==="
echo "📊 Summary:"
echo "  - PKL files: ${PKL_ROOT_DIR}"
echo "  - Trained models: ${MODEL_ROOT_DIR}"
echo "  - Cached steps are skipped automatically on re-run"