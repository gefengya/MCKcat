#!/bin/bash

# Script to reproduce training, prediction, and analysis runs for
# CatPred models on CatPred-DB datasets for kcat (supports multiple data splits)
# Modified to support custom data paths: warm/cold_enzyme/cold_reaction/catpred_catpro

# Author: Veda Sheersh Boorla (Modified by Your Name)
# Date: 12-09-2024 (Modified Date)

# Exit script on error
set -e

# --------------------------
# 1. 基础配置（修改这里适配你的环境路径）
# --------------------------
# 根数据目录（所有数据集的父目录，根据实际路径调整）
ROOT_DATA_DIR="./data"  
# 模型 checkpoint、日志、结果输出目录（保持原逻辑）
CKPT_DIR="../data/pretrained/reproduce_checkpoints/"
LOG_DIR="../data/results/reproduce_logs"
OUTPUT_DIR="../data/results/reproduce_results"
# 脚本路径（确保 train.py/predict.py/analyze_ablation.py 路径正确）
TRAINING_SCRIPT="train.py"
PREDICTION_SCRIPT="predict.py"
ANALYSIS_SCRIPT="./scripts/analyze_ablation.py"

# 定义要训练的实验类型（仅保留 seqemb36_attn6_esm_ens10，可按需添加其他实验）
EXPERIMENTS=(
    seqemb36_attn6_esm_ens10
)

# 定义所有数据集配置（名称 + 训练/测试路径模板）
# 格式："数据集名称:训练路径模板:测试路径模板:折数范围"
# 路径模板中 {i} 表示折数（1-5），无折数时留空
DATASETS=(
    # 1. warm 数据集（i=1-5）
    "warm:${ROOT_DATA_DIR}/turnup/warm/{i}/train_fold_6_with_sub_.csv:${ROOT_DATA_DIR}/turnup/warm/{i}/test_fold_6_with_sub_.csv:1-5"
    # 2. cold_enzyme 数据集（i=1-5）
    "cold_enzyme:${ROOT_DATA_DIR}/turnup/cold_enzyme/{i}/train_fold_6_with_sub_.csv:${ROOT_DATA_DIR}/turnup/cold_enzyme/{i}/test_fold_6_with_sub_.csv:1-5"
    # 3. cold_reaction 数据集（i=1-5）
    "cold_reaction:${ROOT_DATA_DIR}/turnup/cold_reaction/{i}/train_fold_6_with_sub_.csv:${ROOT_DATA_DIR}/turnup/cold_reaction/{i}/test_fold_6_with_sub_.csv:1-5"
    # 4. catpred_catpro 数据集（无折数，单训练/测试文件）
    "catpred_catpro:${ROOT_DATA_DIR}/catpred_catpro/train.csv:${ROOT_DATA_DIR}/catpred_catpro/Held-out test.csv:"
)

# --------------------------
# 2. 初始化目录（确保输出目录存在）
# --------------------------
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$OUTPUT_DIR"
echo "=== 目录配置 ==="
echo "Checkpoint 目录: $(realpath "$CKPT_DIR")"
echo "日志目录: $(realpath "$LOG_DIR")"
echo "结果输出目录: $(realpath "$OUTPUT_DIR")"
echo "根数据目录: $(realpath "$ROOT_DATA_DIR")"
echo "================\n"


# --------------------------
# 3. 工具函数（解析数据集配置、生成实验参数）
# --------------------------
# 解析数据集配置：根据名称获取训练/测试路径和折数
parse_dataset_config() {
    local dataset_name="$1"
    # 遍历数据集配置，匹配名称
    for config in "${DATASETS[@]}"; do
        IFS=':' read -r name train_template test_template fold_range <<< "$config"
        if [ "$name" = "$dataset_name" ]; then
            echo "$train_template:$test_template:$fold_range"
            return 0
        fi
    done
    echo "错误：未找到数据集 $dataset_name 的配置"
    exit 1
}

# 生成实验额外参数（根据实验类型匹配）
get_experiment_args() {
    local exp_name="$1"
    case "$exp_name" in
        substrate_only)
            echo "--skip_protein"
            ;;
        seqemb36_attn6_ens10)
            echo "--ensemble_size 10"
            ;;
        seqemb36_attn6_esm_ens10)
            echo "--add_esm_feats"  # ESM 特征启用参数（原脚本逻辑）
            ;;
        seqemb36_attn6_esm_ens10_Pretrained_egnnFeats)
            echo "--add_esm_feats --add_pretrained_egnn_feats --pretrained_egnn_feats_path ${ROOT_DATA_DIR}/CatPred-DB/catpred_progres_embeds_dict.pt"
            ;;
        *)
            echo ""  # 默认无额外参数
            ;;
    esac
}


# --------------------------
# 4. 核心训练函数（支持多数据集、多折）
# --------------------------
run_training() {
    local target_parameter="kcat"  # 仅训练 kcat（原脚本逻辑）
    local smiles_col="reactant_smiles"  # kcat 对应的 SMILES 列
    local target_col="log10kcat_max"   # kcat 目标列
    # 蛋白质记录文件（原脚本路径，根据实际位置调整）
    local protein_records="${ROOT_DATA_DIR}/CatPred-DB/data/${target_parameter}/${target_parameter}_max_wt_singleSeqs_wpdbs_pdbrecords.json.gz"

    # 检查蛋白质记录文件是否存在
    if [ ! -f "$protein_records" ]; then
        echo "警告：蛋白质记录文件不存在 $protein_records，跳过蛋白质相关特征"
        protein_records=""
    fi

    echo "=== 开始训练：参数=$target_parameter ==="
    echo "实验类型: ${EXPERIMENTS[*]}"
    echo "数据集: ${!DATASETS[@]}\n"

    # 遍历每个数据集
    for dataset_config in "${DATASETS[@]}"; do
        # 解析数据集信息（名称、路径模板、折数）
        IFS=':' read -r dataset_name train_template test_template fold_range <<< "$dataset_config"
        echo "=================================================="
        echo "正在处理数据集：$dataset_name"
        echo "训练路径模板：$train_template"
        echo "测试路径模板：$test_template"
        echo "折数范围：${fold_range:-无折数}"
        echo "=================================================="

        # 处理折数（有折数则遍历 1-5，无折数则单轮训练）
        if [ -n "$fold_range" ]; then
            # 折数范围为 1-5（解析为 start=1, end=5）
            IFS='-' read -r start_fold end_fold <<< "$fold_range"
            for ((i=start_fold; i<=end_fold; i++)); do
                # 替换路径模板中的 {i} 为实际折数
                train_path=$(echo "$train_template" | sed "s/{i}/$i/g")
                test_path=$(echo "$test_template" | sed "s/{i}/$i/g")
                # 训练单折数据
                train_single_fold "$dataset_name" "$i" "$train_path" "$test_path" "$target_parameter" "$smiles_col" "$target_col" "$protein_records"
            done
        else
            # 无折数（如 catpred_catpro），直接使用原始路径
            train_path="$train_template"
            test_path="$test_template"
            train_single_fold "$dataset_name" "none" "$train_path" "$test_path" "$target_parameter" "$smiles_col" "$target_col" "$protein_records"
        fi
    done

    echo -e "\n=== 所有数据集训练完成 ==="
}

# 单折训练函数（复用逻辑，减少冗余）
train_single_fold() {
    local dataset_name="$1"    # 数据集名称（如 warm）
    local fold="$2"            # 折数（none 表示无折数）
    local train_path="$3"      # 实际训练文件路径
    local test_path="$4"       # 实际测试文件路径
    local param="$5"           # 目标参数（kcat）
    local smiles_col="$6"      # SMILES 列名
    local target_col="$7"      # 目标列名
    local protein_records="$8" # 蛋白质记录文件

    # 检查训练/测试文件是否存在
    if [ ! -f "$train_path" ]; then
        echo "警告：训练文件不存在 $train_path，跳过该折"
        return 1
    fi
    if [ ! -f "$test_path" ]; then
        echo "警告：测试文件不存在 $test_path，跳过该折"
        return 1
    fi

    # 遍历每个实验类型（如 seqemb36_attn6_esm_ens10）
    for exp in "${EXPERIMENTS[@]}"; do
        # 1. 配置当前实验的保存目录和日志文件
        local exp_suffix=""
        if [ "$fold" != "none" ]; then
            exp_suffix="fold_${fold}"  # 有折数则添加后缀（如 fold_1）
        fi
        local save_dir="${CKPT_DIR}/${param}_ablation_retrain/${dataset_name}/${exp}/${exp_suffix}"
        local log_file="${LOG_DIR}/${param}_training_${dataset_name}_${exp}_${exp_suffix}.log"

        # 2. 创建保存目录（避免目录不存在报错）
        mkdir -p "$save_dir"

        # 3. 获取实验额外参数（如 ESM 特征）
        local extra_args=$(get_experiment_args "$exp")

        # 4. 构建蛋白质记录参数（文件存在才传入）
        local protein_arg=""
        if [ -n "$protein_records" ] && [ -f "$protein_records" ]; then
            protein_arg="--protein_records_path $protein_records"
        fi

        # 5. 打印训练信息
        echo -e "\n【开始训练】"
        echo "数据集: $dataset_name | 折数: $fold | 实验: $exp"
        echo "训练文件: $(realpath "$train_path")"
        echo "测试文件: $(realpath "$test_path")"
        echo "保存目录: $(realpath "$save_dir")"
        echo "日志文件: $(realpath "$log_file")"
        echo "额外参数: $extra_args"

        # 6. 执行训练脚本（核心命令，保留原脚本的超参数）
        python "$TRAINING_SCRIPT" \
            $protein_arg \
            --data_path "$train_path" \
            --dataset_type regression \
            --separate_test_path "$test_path" \
            --separate_val_path "$test_path" \  # 用测试集作为验证集（原脚本逻辑）
            --smiles_columns "$smiles_col" \
            --target_columns "$target_col" \
            --extra_metrics mae mse r2 \
            --ensemble_size 10 \
            --seq_embed_dim 36 \
            --seq_self_attn_nheads 6 \
            --loss_function mve \
            --batch_size 16 \
            --save_dir "$save_dir" \
            --epochs 20 \
            $extra_args \
            > "$log_file" 2>&1

        # 7. 检查训练结果
        if [ $? -eq 0 ]; then
            echo "【训练成功】数据集: $dataset_name | 折数: $fold | 实验: $exp（日志：$log_file）"
        else
            echo "【训练失败】数据集: $dataset_name | 折数: $fold | 实验: $exp（查看日志：$log_file）"
        fi
    done
}


# --------------------------
# 5. 核心预测函数（匹配训练的数据集和实验）
# --------------------------
run_prediction() {
    local target_parameter="kcat"
    local smiles_col="reactant_smiles"
    # 蛋白质记录文件（与训练一致）
    local protein_records="${ROOT_DATA_DIR}/CatPred-DB/data/${target_parameter}/${target_parameter}_max_wt_singleSeqs_wpdbs_pdbrecords.json.gz"

    echo -e "\n=== 开始预测：参数=$target_parameter ==="

    # 遍历每个数据集
    for dataset_config in "${DATASETS[@]}"; do
        IFS=':' read -r dataset_name train_template test_template fold_range <<< "$dataset_config"
        echo "=================================================="
        echo "正在处理数据集：$dataset_name"
        echo "测试路径模板：$test_template"
        echo "折数范围：${fold_range:-无折数}"
        echo "=================================================="

        # 处理折数（与训练逻辑一致）
        if [ -n "$fold_range" ]; then
            IFS='-' read -r start_fold end_fold <<< "$fold_range"
            for ((i=start_fold; i<=end_fold; i++)); do
                test_path=$(echo "$test_template" | sed "s/{i}/$i/g")
                predict_single_fold "$dataset_name" "$i" "$test_path" "$target_parameter" "$smiles_col" "$protein_records"
            done
        else
            test_path="$test_template"
            predict_single_fold "$dataset_name" "none" "$test_path" "$target_parameter" "$smiles_col" "$protein_records"
        fi
    done

    echo -e "\n=== 所有数据集预测完成 ==="
}

# 单折预测函数
predict_single_fold() {
    local dataset_name="$1"
    local fold="$2"
    local test_path="$3"
    local param="$4"
    local smiles_col="$5"
    local protein_records="$6"

    # 检查测试文件是否存在
    if [ ! -f "$test_path" ]; then
        echo "警告：测试文件不存在 $test_path，跳过该折"
        return 1
    fi

    # 遍历每个实验类型
    for exp in "${EXPERIMENTS[@]}"; do
        # 1. 配置输出路径和 checkpoint 目录（与训练目录对应）
        local exp_suffix=""
        if [ "$fold" != "none" ]; then
            exp_suffix="fold_${fold}"
        fi
        local checkpoint_dir="${CKPT_DIR}/${param}_ablation_retrain/${dataset_name}/${exp}/${exp_suffix}"
        local preds_output="${OUTPUT_DIR}/${param}/${dataset_name}/${exp}/${exp_suffix}"
        local preds_file="${preds_output}/${param}_predictions.csv"
        local log_file="${LOG_DIR}/${param}_prediction_${dataset_name}_${exp}_${exp_suffix}.log"

        # 2. 创建输出目录
        mkdir -p "$preds_output"

        # 3. 检查 checkpoint 目录是否存在（训练过才预测）
        if [ ! -d "$checkpoint_dir" ]; then
            echo "警告：Checkpoint 目录不存在 $checkpoint_dir（未训练该实验），跳过预测"
            continue
        fi

        # 4. 获取实验额外参数（如预训练 EGNN 特征）
        local extra_args=""
        if [ "$exp" = "seqemb36_attn6_esm_ens10_Pretrained_egnnFeats" ]; then
            extra_args="--pretrained_egnn_feats_path ${ROOT_DATA_DIR}/CatPred-DB/catpred_progres_embeds_dict.pt"
        fi

        # 5. 构建蛋白质记录参数
        local protein_arg=""
        if [ -n "$protein_records" ] && [ -f "$protein_records" ]; then
            protein_arg="--protein_records_path $protein_records"
        fi

        # 6. 打印预测信息
        echo -e "\n【开始预测】"
        echo "数据集: $dataset_name | 折数: $fold | 实验: $exp"
        echo "测试文件: $(realpath "$test_path")"
        echo "Checkpoint 目录: $(realpath "$checkpoint_dir")"
        echo "预测结果: $(realpath "$preds_file")"
        echo "日志文件: $(realpath "$log_file")"

        # 7. 执行预测脚本（核心命令）
        python "$PREDICTION_SCRIPT" \
            $protein_arg \
            --test_path "$test_path" \
            --smiles_columns "$smiles_col" \
            --preds_path "$preds_file" \
            --checkpoint_dir "$checkpoint_dir" \
            --individual_ensemble_predictions \  # 输出每个集成模型的预测结果
            --batch_size 32 \
            $extra_args \
            > "$log_file" 2>&1

        # 8. 检查预测结果
        if [ $? -eq 0 ]; then
            echo "【预测成功】数据集: $dataset_name | 折数: $fold | 实验: $exp（结果：$preds_file）"
        else
            echo "【预测失败】数据集: $dataset_name | 折数: $fold | 实验: $exp（查看日志：$log_file）"
        fi
    done
}


# --------------------------
# 6. 分析函数（适配多数据集结果）
# --------------------------
run_analysis() {
    local target_parameter="kcat"
    echo -e "\n=== 开始分析：参数=$target_parameter ==="

    # 收集所有预测结果文件（匹配多数据集格式）
    local all_preds_files=()
    for dataset_config in "${DATASETS[@]}"; do
        IFS=':' read -r dataset_name _ test_template fold_range <<< "$dataset_config"
        # 构建该数据集的预测结果路径模板
        local preds_template="${OUTPUT_DIR}/${target_parameter}/${dataset_name}/${EXPERIMENTS[0]}"
        if [ -n "$fold_range" ]; then
            # 多折数据集：匹配所有折的结果
            for ((i=1; i<=5; i++)); do
                local preds_path="${preds_template}/fold_${i}/${target_parameter}_predictions.csv"
                if [ -f "$preds_path" ]; then
                    all_preds_files+=("$preds_path")
                fi
            done
        else
            # 单文件数据集：直接匹配
            local preds_path="${preds_template}/none/${target_parameter}_predictions.csv"
            if [ -f "$preds_path" ]; then
                all_preds_files+=("$preds_path")
            fi
        fi
    done

    # 检查是否有预测结果
    if [ ${#all_preds_files[@]} -eq 0 ]; then
        echo "错误：未找到任何预测结果文件，跳过分析"
        return 1
    fi

    # 定义分析输入文件（根据实际训练数据路径调整）
    local train_data="${ROOT_DATA_DIR}/CatPred-DB/data/${target_parameter}/${target_parameter}_train_reaction.csv"
    local test_data="${ROOT_DATA_DIR}/CatPred-DB/data/${target_parameter}/${target_parameter}_test_reaction.csv"
    local analysis_summary="${OUTPUT_DIR}/${target_parameter}_ablation_analysis-summary.csv"
    local log_file="${LOG_DIR}/${target_parameter}_ablation_analysis.log"

    # 执行分析脚本
    echo "分析输入：训练数据=$train_data | 测试数据=$test_data"
    echo "分析结果：$analysis_summary"
    echo "预测结果文件数量：${#all_preds_files[@]}"

    python "$ANALYSIS_SCRIPT" \
        "$target_parameter" \
        "$train_data" \
        "$test_data" \
        "$analysis_summary" \
        "${all_preds_files[@]}" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "\n【分析成功】"
        echo "分析总结：$(realpath "$analysis_summary")"
        echo "R2/MAE/P1mag 结果：$(realpath "${OUTPUT_DIR}")"
        echo "分析日志：$(realpath "$log_file")"
    else
        echo "【分析失败】查看日志：$log_file"
    fi

    echo -e "\n=== 分析完成 ==="
}


# --------------------------
# 7. 主流程入口（支持 training/prediction/analysis 命令）
# --------------------------
usage() {
    echo "用法：$0 [training|prediction|analysis]"
    echo "  training   - 执行所有数据集的训练 + 预测 + 分析"
    echo "  prediction - 仅执行所有数据集的预测 + 分析"
    echo "  analysis   - 仅执行分析（需先完成预测）"
    exit 1
}

# 检查参数数量
if [ "$#" -ne 1 ]; then
    usage
fi

# 执行对应流程
case "$1" in
    training)
        echo "=== 启动完整流程：训练 → 预测 → 分析 ==="
        run_training
        run_prediction
        run_analysis
        ;;
    prediction)
        echo "=== 启动流程：预测 → 分析 ==="
        run_prediction
        run_analysis
        ;;
    analysis)
        echo "=== 启动流程：仅分析 ==="
        run_analysis
        ;;
    *)
        usage
        ;;
esac

echo -e "\n=== 流程 $1 执行完成！==="