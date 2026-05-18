#!/bin/bash
set -e  # 任何命令出错时立即退出

# -------------------- 参数配置 --------------------
input_file="/home/work_nfs19/hkxie/modified_scp/part_2.scp"   # 输入的 scp 文件路径
data_root="/home"                                               # 数据根目录参数
out_data_root="/home/work_nfs19/hkxie/tar_bag/10wh/"         # 输出文件存储目录（各个任务写入此目录）
num_gpus=16                                                      # GPU 数量
log_dir="./logs"                                                # 日志输出目录
tmp_dir="./tmp_split"                                             # 分割后临时文件存放目录

# -------------------- 创建必要目录 --------------------
mkdir -p "$log_dir" "$tmp_dir"

# -------------------- 步骤1: 分割 scp 文件 --------------------
# 将输入文件按行均分为 8 个部分，生成的文件名形如 part_00.scp, part_01.scp, ...
split -d -n l/$num_gpus --additional-suffix ".scp" "$input_file" "${tmp_dir}/part_"

# -------------------- 步骤2: 多 GPU 并行处理 --------------------
pids=()
for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
    # 根据 GPU id 构造分割后的 scp 文件名
    part_scp=$(printf "${tmp_dir}/part_%02d.scp" $gpu_id)
    
    (
        # 激活指定 conda 环境
        source /home/environment2/hkxie/anaconda3/bin/activate /home/environment/zhguo/anaconda3/envs/torch112_dualvc
        # 设置环境变量限制当前进程使用对应 GPU，启动处理任务
	CUDA_VISIBLE_DEVICES=$(( gpu_id % 8 )) python extract.py \
            --data_root "$data_root" \
            --out_data_root "$out_data_root" \
            --filelist "$part_scp"
    ) > "${log_dir}/gpu${gpu_id}.log" 2>&1 &
    
    pids+=($!)
    echo "GPU $gpu_id 正在处理文件: $part_scp (PID: ${pids[-1]})"
done

# -------------------- 步骤3: 等待所有任务完成 --------------------
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        echo "进程 $pid 失败！"
        exit 1
    fi
done

echo "全部任务完成！处理结果已存储到: $out_data_root"
