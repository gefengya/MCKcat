#!/bin/bash
# 整合版数据集训练脚本
# 支持：Warm(1-5折)、Cold Enzyme(1-5折)、Cold Reaction(1-5折)、CatPred基础数据集
# 数据路径：./data/turnup/... 和 ./data/catpred_catpro/...

# ========================= 全局训练参数配置 =========================
# 可根据需求调整以下参数
BATCH_SIZE=8               # 批次大小
DEVICE="cuda:0"            # 训练设备（cuda:0 或 cpu）
TRAIN_SCRIPT="train.py"    # 训练脚本路径（确保与脚本实际位置一致）
MODEL_ROOT_DIR="models"    # 模型保存根目录（所有数据集模型会按子目录区分）
DATA_ROOT_DIR="./data"     # 数据根目录（匹配你的 ./data 路径结构）

# ========================= 数据集配置 =========================
# 定义所有待训练的数据集（名称:数据前缀:折数范围:训练文件名模板:测试文件名模板）
# 格式说明："数据集名:数据路径前缀:折数范围(如1-5):训练文件模板:测试文件模板"
# 无折数数据集（如CatPred基础）：折数范围留空，文件模板为固定文件名
DATASETS=(
    # 1. Warm 数据集（1-5折，文件模板：train_fold_6_with_sub_.csv/test_fold_6_with_sub_.csv）
    "warm:${DATA_ROOT_DIR}/turnup/warm:1-5:train_fold_6_with_sub_.csv:test_fold_6_with_sub_.csv"
    # 2. Cold Enzyme 数据集（1-5折，同上文件模板）
    "cold_enzyme:${DATA_ROOT_DIR}/turnup/cold_enzyme:1-5:train_fold_6_with_sub_.csv:test_fold_6_with_sub_.csv"
    # 3. Cold Reaction 数据集（1-5折，同上文件模板）
    "cold_reaction:${DATA_ROOT_DIR}/turnup/cold_reaction:1-5:train_fold_6_with_sub_.csv:test_fold_6_with_sub_.csv"
    # 4. CatPred基础数据集（无折数，固定文件名）
    "catpred_basic:${DATA_ROOT_DIR}/catpred_catpro::train.csv:Held-out test.csv"
)

# ========================= 工具函数 =========================
# 功能：训练单个数据集的单折（或无折数据集）
# 参数：数据集名、折数（none表示无折）、训练文件路径、测试文件路径
train_single_fold() {
    local dataset_name="$1"
    local fold="$2"
    local train_path="$3"
    local test_path="$4"

    # 1. 构建模型保存目录（按“数据集/折数”分层，避免文件冲突）
    local model_subdir="${dataset_name}"
    if [ "$fold" != "none" ]; then
        model_subdir="${dataset_name}/fold_${fold}"
    fi
    local model_dir="${MODEL_ROOT_DIR}/${model_subdir}"
    mkdir -p "$model_dir"  # 确保目录存在

    # 2. 打印训练信息
    echo -e "\n=================================================="
    if [ "$fold" != "none" ]; then
        echo "【开始训练】数据集：${dataset_name} | 折数：${fold}"
    else
        echo "【开始训练】数据集：${dataset_name}（无折数）"
    fi
    echo "训练文件：$(realpath "$train_path")"
    echo "测试文件：$(realpath "$test_path")"
    echo "模型保存：$(realpath "$model_dir")"
    echo "=================================================="

    # 3. 执行训练命令（参数与原脚本保持一致：-inp_fpath/-test_fpath等）
    python "$TRAIN_SCRIPT" \
        -inp_fpath "$train_path" \
        -test_fpath "$test_path" \
        -model_dpath "$model_dir" \
        -batch_size "$BATCH_SIZE" \
        -device "$DEVICE"

    # 4. 检查训练结果
    if [ $? -eq 0 ]; then
        echo -e "【训练成功】数据集：${dataset_name} | 折数：${fold} → 模型保存于：${model_dir}\n"
    else
        echo -e "【训练失败】数据集：${dataset_name} | 折数：${fold} → 请查看日志排查问题\n"
    fi
}

# ========================= 主训练流程 =========================
echo "=== 整合版数据集训练脚本启动 ==="
echo "全局参数：批次大小=${BATCH_SIZE} | 设备=${DEVICE} | 数据根目录=${DATA_ROOT_DIR}"
echo "模型根目录：${MODEL_ROOT_DIR} | 训练脚本：${TRAIN_SCRIPT}"
echo "待训练数据集数量：${#DATASETS[@]}\n"

# 遍历所有数据集，执行训练
for dataset_config in "${DATASETS[@]}"; do
    # 解析数据集配置（按“:”分割）
    IFS=':' read -r ds_name ds_prefix fold_range train_template test_template <<< "$dataset_config"
    
    echo -e "================ 开始处理数据集：${ds_name} ================"
    
    # 分两种情况：有折数数据集（如warm 1-5折）、无折数数据集（如catpred_basic）
    if [ -n "$fold_range" ]; then
        # 1. 有折数数据集：解析折数范围（如1-5），循环训练每折
        IFS='-' read -r start_fold end_fold <<< "$fold_range"
        echo "数据集${ds_name}：折数范围 ${start_fold}-${end_fold}，开始循环训练..."
        
        for ((i=start_fold; i<=end_fold; i++)); do
            # 构建当前折的训练/测试文件路径（替换路径中的折数占位）
            train_file="${ds_prefix}/${i}/${train_template}"
            test_file="${ds_prefix}/${i}/${test_template}"
            
            # 检查文件是否存在，不存在则跳过该折
            if [ ! -f "$train_file" ]; then
                echo "警告：数据集${ds_name}第${i}折训练文件不存在 → ${train_file}，跳过"
                continue
            fi
            if [ ! -f "$test_file" ]; then
                echo "警告：数据集${ds_name}第${i}折测试文件不存在 → ${test_file}，跳过"
                continue
            fi
            
            # 调用单折训练函数
            train_single_fold "$ds_name" "$i" "$train_file" "$test_file"
        done
    else
        # 2. 无折数数据集（如CatPred基础数据集）：直接使用固定文件名
        train_file="${ds_prefix}/${train_template}"
        test_file="${ds_prefix}/${test_template}"
        
        # 检查文件是否存在
        if [ ! -f "$train_file" ]; then
            echo "错误：数据集${ds_name}训练文件不存在 → ${train_file}，跳过该数据集"
            continue
        fi
        if [ ! -f "$test_file" ]; then
            echo "错误：数据集${ds_name}测试文件不存在 → ${test_file}，跳过该数据集"
            continue
        fi
        
        # 调用单折训练函数（折数参数传"none"）
        train_single_fold "$ds_name" "none" "$train_file" "$test_file"
    fi
    
    echo -e "================ 数据集${ds_name}处理完成 ================\n"
done

# ========================= 训练总结 =========================
echo "=== 所有数据集训练流程结束 ==="
echo "模型总保存目录：$(realpath "${MODEL_ROOT_DIR}")"
echo "提示：可在上述目录中查看各数据集的训练结果"