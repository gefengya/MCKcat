#!/bin/bash

# ========================= Global Test Configuration =========================
TEST_SCRIPT="test.py"
DEVICE=0
MOL_TYPE="MACCSKeys"
MODEL_BEST_NAME="kcat_model_best_best.pt"

DATA_ROOT="./data"
PKL_ROOT="${DATA_ROOT}/pkl_files"
MODEL_ROOT="./models"
RESULT_ROOT="./test_results"

# ========================= Dataset Configuration =========================
# Format: "dataset_name:data_subpath:fold_range:pkl_filename_template:model_subdirectory"
DATASETS=(
    # Warm Dataset (Folds 1-5)
    "warm:TurNup_DB/warm:1-5:test_fold_6_with_sub__with_esm.pkl:warm"
    # Cold Enzyme Dataset (Folds 1-5)
    "cold_enzyme:TurNup_DB/cold_enzyme:1-5:test_fold_6_with_sub__with_esm.pkl:cold_enzyme"
    # Cold Reaction Dataset (Folds 1-5)
    "cold_reaction:TurNup_DB/cold_reaction:1-5:test_fold_6_with_sub__with_esm.pkl:cold_reaction"
    # CatPred Base Dataset (no folds)
    "MCKcat_DB_basic:MCKcat_DB::Held-out_test_with_esm.pkl:catpred_basic"
)

# ========================= Utility Functions =========================
# Test single fold data
test_single() {
    local dataset=$1
    local fold=$2
    local esm_pkl=$3
    local model_path=$4
    local output_dir=$5

    # Check file existence
    if [ ! -f "$esm_pkl" ]; then
        echo "❌ Missing test PKL with ESM features: $esm_pkl"
        return 1
    fi
    if [ ! -f "$model_path" ]; then
        echo "❌ Missing model file: $model_path"
        return 1
    fi

    # Create output directory
    mkdir -p "$output_dir"
    echo "📌 Results will be saved to: $output_dir"

    # Run test
    echo "▶️ Starting test for ${dataset} (Fold ${fold})..."
    python "$TEST_SCRIPT" \
        --test_pkl "$esm_pkl" \
        --model_path "$model_path" \
        --output_dir "$output_dir" \
        --mol_type "$MOL_TYPE" \
        --device "$DEVICE"

    # Check test result
    if [ $? -eq 0 ]; then
        echo "✅ ${dataset} (Fold ${fold}) test completed"
        return 0
    else
        echo "❌ ${dataset} (Fold ${fold}) test failed"
        return 1
    fi
}

# Process full dataset testing
process_dataset() {
    local config=$1
    IFS=':' read -r name subpath folds pkl_template model_subdir <<< "$config"

    echo -e "\n=================================================="
    echo "Starting test for dataset: $name"
    echo "Data path: ${DATA_ROOT}/${subpath}"
    echo "Fold range: ${folds:-No folds}"
    echo "=================================================="

    # Process datasets with folds
    if [ -n "$folds" ]; then
        IFS='-' read -r start end <<< "$folds"
        for ((fold=start; fold<=end; fold++)); do
            echo -e "\n--------------------------------------------------"
            echo "Testing Fold ${fold}"
            echo "--------------------------------------------------"

            # Build paths
            esm_pkl="${PKL_ROOT}/${name}/${fold}/${pkl_template}"
            model_path="${MODEL_ROOT}/${model_subdir}/fold_${fold}/${MODEL_BEST_NAME}"
            output_dir="${RESULT_ROOT}/${name}/fold_${fold}"

            # Run test
            test_single "$name" "$fold" "$esm_pkl" "$model_path" "$output_dir"
        done
    else
        # Process datasets without folds
        echo -e "\n--------------------------------------------------"
        echo "Testing non-fold dataset"
        echo "--------------------------------------------------"

        # Build paths
        esm_pkl="${PKL_ROOT}/${name}/${pkl_template}"
        model_path="${MODEL_ROOT}/${model_subdir}/${MODEL_BEST_NAME}"
        output_dir="${RESULT_ROOT}/${name}"

        # Run test
        test_single "$name" "none" "$esm_pkl" "$model_path" "$output_dir"
    fi

    echo -e "\n=================================================="
    echo "$name dataset testing completed"
    echo "Result summary: ${RESULT_ROOT}/${name}"
    echo "=================================================="
}

# ========================= Main Workflow =========================
echo "=== Multi-Dataset Model Testing Script Started ==="
echo "Test parameters:"
echo "  Device: cuda:${DEVICE}"
echo "  Molecular feature type: ${MOL_TYPE}"
echo "  Test script: ${TEST_SCRIPT}"
echo "  Result root directory: ${RESULT_ROOT}"

# Initialize result directory
mkdir -p "$RESULT_ROOT"

# Iterate all datasets and run tests
for dataset in "${DATASETS[@]}"; do
    process_dataset "$dataset"
done

echo -e "\n=== All datasets testing completed ==="
echo "Final results stored at: ${RESULT_ROOT}"