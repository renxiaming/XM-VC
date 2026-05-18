
# 激活指定 conda 环境
source /home/environment2/hkxie/anaconda3/bin/activate /home/environment/zhguo/anaconda3/envs/torch112_dualvc

# cd /home/work_nfs19/hkxie/extract/preprocess

# 设置环境变量限制当前进程使用对应 GPU，启动处理任务
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python preprocess/extract_bn_10ms.py \
    --data_root /home/work_nfs19/hkxie/tar_bag/data/hq_cn_seedvcref_lance_data_2 \
    --out_data_root /home/work_nfs19/hkxie/tar_bag/data/hq_cn_seedvcref_lance_data_2bn \
    --filelist /home/work_nfs19/hkxie/modified_scp/part_2.scp \
    --use_lance True 

