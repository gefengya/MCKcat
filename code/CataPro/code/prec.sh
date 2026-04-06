#!/bin/bash
# Integrated Batch Prediction Script for Multiple Datasets
# Supports: Warm (5 folds), Cold Enzyme (5 folds), Cold Reaction (5 folds), CatPred Base Dataset
# Data Paths: ./data/turnup/... and ./data/catpred_catpro/...
# Model Paths: Auto-match hierarchical directories from training (models/dataset/fold)

# ========================= Global Prediction Parameters =========================
# Adjustable parameters
BATCH_SIZE=64              # Prediction batch size (reduce if out of memory, e.g. 32/16)
DEVICE="cuda:0"            # Prediction device (cuda:0 or cpu)
PREDICT_SCRIPT="predict.py"# Path to prediction script (ensure matches actual location)
OUTPUT_ROOT="./predictions"# Root directory for prediction results (all results saved by dataset)
MODEL_ROOT_DIR="models"    # Root directory for saved models (must match MODEL_ROOT_DIR in training script)
DATA_ROOT_DIR="./data"     # Root data directory (matches your ./data path structure)

# ========================= Dataset Configuration =========================
# Define all datasets to predict (name:data_prefix:fold_range:test_file_template:model_subdir_template)
# Format: "Dataset Name:Data Path Prefix:Fold Range:Test File Template:Model Subdir Template"
# Non-fold datasets: leave fold range empty, model subdir is fixed name
DATASETS=(
    # 1. Warm Dataset (1-5 folds, test file: test_fold_6_with_sub_.csv)
    "warm:${DATA_ROOT_DIR}/turnup/warm:1-5:test_fold_6_with_sub_.csv:warm/fold_{fold}"
    # 2. Cold Enzyme Dataset (1-5 folds, same test file)
    "cold_enzyme:${DATA_ROOT_DIR}/turnup/cold_enzyme:1-5:test_fold_6_with_sub_.csv:cold_enzyme/fold_{fold}"
    # 3. Cold Reaction Dataset (1-5 folds, same test file)
    "cold_reaction:${DATA_ROOT_DIR}/turnup/cold_reaction:1-5:test_fold_6_with_sub_.csv:cold_reaction/fold_{fold}"
    # 4. CatPred Basic Dataset (no folds, test file: Held-out test.csv)
    "catpred_basic:${DATA_ROOT_DIR}/catpred_catpro::Held-out test.csv:catpred_basic"
)

# ========================= Utility Functions =========================
# Function: Predict single fold for a dataset (or non-fold dataset)
# Parameters: dataset_name, fold (none for no folds), test_file_path, model_dir_path
predict_single_fold() {
    local dataset_name="$1"
    local fold="$2"
    local test_path="$3"
    local model_dir="$4"

    # 1. Build prediction output directory (hierarchical by dataset/fold, matching training dir)
    local output_subdir="${dataset_name}"
    if [ "$fold" != "none" ]; then
        output_subdir="${dataset_name}/fold_${fold}"
    fi
    local output_dir="${OUTPUT_ROOT}/${output_subdir}"
    local output_file="${output_dir}/pred_results.csv"  # Prediction result filename
    mkdir -p "$output_dir"  # Ensure directory exists

    # 2. Print prediction info
    echo -e "\n=================================================="
    if [ "$fold" != "none" ]; then
        echo "Start Prediction | Dataset: ${dataset_name} | Fold: ${fold}"
    else
        echo "Start Prediction | Dataset: ${dataset_name} (No Folds)"
    fi
    echo "Test File: $(realpath "$test_path")"
    echo "Model Directory: $(realpath "$model_dir")"
    echo "Results Save Path: $(realpath "$output_file")"
    echo "=================================================="

    # 3. Execute prediction command (parameters match original script: -inp_fpath/-model_dpath etc.)
    python "$PREDICT_SCRIPT" \
        -inp_fpath "$test_path" \
        -model_dpath "$model_dir" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE" \
        -out_fpath "$output_file"

    # 4. Check prediction result
    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        echo -e "Prediction Success → Results saved to: ${output_file}\n"
    else
        echo -e "Prediction Failed → No result file generated (check logs for issues)\n"
    fi
}

# ========================= Main Prediction Workflow =========================
echo "=== Integrated Multi-Dataset Batch Prediction Script Started ==="
echo "Global Params: Batch Size=${BATCH_SIZE} | Device=${DEVICE} | Results Root=${OUTPUT_ROOT}"
echo "Model Root: ${MODEL_ROOT_DIR} | Prediction Script: ${PREDICT_SCRIPT}"
echo "Number of Datasets to Predict: ${#DATASETS[@]}\n"

# Iterate all datasets and run prediction
for dataset_config in "${DATASETS[@]}"; do
    # Parse dataset config (split by :)
    IFS=':' read -r ds_name ds_prefix fold_range test_template model_dir_template <<< "$dataset_config"

    echo -e "================ Processing Dataset: ${ds_name} ================"

    # Two cases: folded datasets (warm 1-5), non-folded datasets (catpred_basic)
    if [ -n "$fold_range" ]; then
        # 1. Folded dataset: parse fold range (e.g. 1-5), loop predict each fold
        IFS='-' read -r start_fold end_fold <<< "$fold_range"
        echo "Dataset ${ds_name}: Fold range ${start_fold}-${end_fold}, starting batch prediction..."

        for ((fold=start_fold; fold<=end_fold; fold++)); do
            # Build test file path for current fold
            test_file="${ds_prefix}/${fold}/${test_template}"
            # Build model directory path (replace {fold} with actual fold number)
            model_dir=$(echo "${MODEL_ROOT_DIR}/${model_dir_template}" | sed "s/{fold}/${fold}/g")

            # Check if test file and model dir exist
            if [ ! -f "$test_file" ]; then
                echo "Warning: Test file missing for ${ds_name} fold ${fold} → ${test_file}, skipping"
                continue
            fi
            if [ ! -d "$model_dir" ]; then
                echo "Warning: Model directory missing for ${ds_name} fold ${fold} → ${model_dir}, skipping"
                continue
            fi

            # Call single fold prediction function
            predict_single_fold "$ds_name" "$fold" "$test_file" "$model_dir"
        done
    else
        # 2. Non-fold dataset (CatPred Basic): use fixed paths directly
        test_file="${ds_prefix}/${test_template}"
        model_dir="${MODEL_ROOT_DIR}/${model_dir_template}"

        # Check test file and model directory
        if [ ! -f "$test_file" ]; then
            echo "Error: Test file missing for ${ds_name} → ${test_file}, skipping dataset"
            continue
        fi
        if [ ! -d "$model_dir" ]; then
            echo "Error: Model directory missing for ${ds_name} → ${model_dir}, skipping dataset"
            continue
        fi

        # Call prediction function (pass "none" for fold)
        predict_single_fold "$ds_name" "none" "$test_file" "$model_dir"
    fi

    echo -e "================ Dataset ${ds_name} Prediction Completed ================\n"
done

# ========================= Prediction Summary =========================
echo "=== All Dataset Prediction Workflows Completed ==="
echo "Total Prediction Results Directory: $(realpath "${OUTPUT_ROOT}")"
echo "Tip: Results are saved hierarchically by dataset/fold, check corresponding CSV files directly"