#!/bin/bash

# ----------------------------
# Step 1: 激活conda环境（按需修改路径）
# ----------------------------
# source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/F5-TTS
# source /home/work_nfs19/hkxie/environment/anaconda3/bin/activate /home/work_nfs19/hkxie/environment/anaconda3/envs/f5-tts
# source /home/work_nfs19/hkxie/environment/anaconda3/bin/activate /home/work_nfs19/hkxie/environment/anaconda3/envs/F5-TTS
source /home/work_nfs19/hkxie/environment/anaconda3/bin/activate /home/work_nfs19/hkxie/environment/anaconda3/envs/covomix

# /home/node57_data/hkxie/4O/streaming_fm/data

# ----------------------------
# Step 2: 定位到F5-TTS根目录
# ----------------------------
# 获取脚本绝对路径（兼容软链接）
SCRIPT_PATH=$(readlink -f "$0")
# 定位到项目根目录：从脚本路径向上回退5级（src/f5_tts/train -> 根目录）
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")")
cd "$PROJECT_ROOT" || { echo "Failed to enter project root"; exit 1; }

# ----------------------------
# Step 3: 验证当前路径
# ----------------------------
echo "当前工作目录：$(pwd)"
echo "预期根目录：/home/work_nfs14/code/hkxie/TTS/F5-TTS"  # 请核对路径是否一致
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ----------------------------
# Step 4: 执行训练命令
# ----------------------------
#2.17
# accelerate launch --main_process_port 64434 src/f5_tts/train/train.py --config-name fm_Base_train.yaml
#2.21
# accelerate launch --main_process_port 64434 src/f5_tts/train/train.py --config-name fm_10ms_cosyvoice1.yaml
#2.25
# accelerate launch --main_process_port 64434 src/f5_tts/train/train.py --config-name fm_10ms_cosyvoice1_ecaptdnn.yaml
#2.28 #加入了cosyvoice spk embeding 
# accelerate launch --main_process_port 64434 src/f5_tts/train/train.py --config-name fm_10ms_cosyvoice1_ecaptdnn_spk.yaml

#3.6 1800w数据，转移到了work_nfs19
# accelerate launch --main_process_port 64434 src/f5_tts/train/train.py --config-name fm_10ms_cosyvoice1_ecaptdnn_spk.yaml

#3.12 4卡梯度累计2
export CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes=4 --main_process_port 64434 src/f5_tts/train/train.py --config-name fm_10ms_cosyvoice1_ecaptdnn_spk2.yaml

# python3 /home/node57_data/hkxie/xxtool/send.py "node60 streamingfm_s3token1_ecaptdnn_spk 训练中断？显存？"