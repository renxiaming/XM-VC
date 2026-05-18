source /home/environment2/hkxie/anaconda3/bin/activate /home/environment/zhguo/anaconda3/envs/torch112_dualvc


# python preprocess/extract_bn.py \
#     /home/work_nfs14/code/hkxie/X-Codec-2.0_causal_kernel1/test_audio/output_test_kernel1_ft30w \
#     /home/work_nfs7/hkxie/data/10wh/bnf

# python preprocess/extract_bn.py \
#     --data_root /home \
#     --out_data_root /home/work_nfs19/hkxie/tar_bag/10wh/bnf \
#     --filelist /home/work_nfs19/hkxie/modified_scp/part_1.scp

python preprocess/extract_bn.py \
    --data_root /home \
    --out_data_root /home/work_nfs19/hkxie/tar_bag/10wh/bnf2 \
    --filelist /home/work_nfs19/hkxie/modified_scp/part_2.scp


# python preprocess/extract_bn.py \
#     --data_root /home \
#     --out_data_root /home/work_nfs7/hkxie/data/10wh/bnf4 \
#     --filelist /home/work_nfs19/hkxie/modified_scp/part_4.scp &

# python preprocess/extract_bn.py \
#     --data_root /home \
#     --out_data_root /home/work_nfs7/hkxie/data/10wh/bnf5 \
#     --filelist /home/work_nfs19/hkxie/modified_scp/part_5.scp &

# python preprocess/extract_bn.py \
#     --data_root /home \
#     --out_data_root /home/work_nfs7/hkxie/data/10wh/bnf6 \
#     --filelist /home/work_nfs19/hkxie/modified_scp/part_6.scp &

# python preprocess/extract_bn.py \
#     --data_root /home \
#     --out_data_root /home/work_nfs7/hkxie/data/10wh/bnf7 \
#     --filelist /home/work_nfs19/hkxie/modified_scp/part_7.scp &

# python preprocess/extract_bn.py \
#     --data_root /home \
#     --out_data_root /home/work_nfs7/hkxie/data/10wh/bnf8 \
#     --filelist /home/work_nfs19/hkxie/modified_scp/part_8.scp &