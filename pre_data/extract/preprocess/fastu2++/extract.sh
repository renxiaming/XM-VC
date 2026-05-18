#!/bin/bash
source /home/environment2/hkxie/anaconda3/bin/activate /home/environment/zhguo/anaconda3/envs/torch112_dualvc
export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

dir="exp/huawei_noadd"
config="exp/huawei_noadd/train.yaml"
#test_data="/home/work_nfs6/ypjiang/data/cbhgar/data.list"
#test_data="/home/work_nfs4_ssd/ypjiang/data/big_expressive/data.list"
#test_data="/home/work_nfs6/ypjiang/data/cbhgar/testdata/medium/data.list"
#test_data="/home/work_nfs6/ypjiang/data/for_zqning/data.list"
#test_data="/home/work_nfs6/ypjiang/data/24k_qcxie/data.list"
#test_data="/home/work_nfs5_ssd/ypjiang/data/cbhgar_aug/data.list"
test_data="/home/work_nfs5_ssd/ypjiang/data/svc/data.list"
# test_data="/home/work_nfs7/ypjiang/data/aishell3/xad"
# test_data="/home/work_nfs7/hkxie/data/10wh/bnf"

#checkpoint="exp/huawei_noadd/6.pt"
checkpoint=15.pt
dict="data/dict/lang_char.txt.cn"
mode="extract"
batch_size=1
s_decoding_chunk_size=-1
s_num_decoding_left_chunks=-1
#result_file="/home/work_nfs6/ypjiang/data/cbhgar/fastu2pp/bigcbc4c"
#result_file="/home/work_nfs6/ypjiang/data/cbhgar/testdata/medium/streaming_bn/fastu2pp/bigcbc4c10lc"
#result_file="/home/work_nfs4_ssd/ypjiang/data/big_expressive/fastu2pp/bigcbc2c2lc"
#result_file="/home/work_nfs6/ypjiang/data/for_zqning/bn/vfcbc4c2lc"
#result_file="/home/work_nfs6/ypjiang/data/24k_qcxie/fastu2pp/bigfull"
#result_file="/home/work_nfs5_ssd/ypjiang/data/cbhgar_aug/fastu2pp/bigcbc4c"
#result_file="/home/work_nfs5_ssd/ypjiang/data/svc/fastu2pp/bigfull"
result_file="/home/work_nfs7/ypjiang/data/aishell3/bn/bigfull"
result_file="/home/work_nfs7/hkxie/data/10wh/bnf"

#python3 wenet/bin/export_jit.py \
#	--config ${config} \
#	--checkpoint ${checkpoint} \
#	--output_file "./asrbig2stg.pt"

#exit
python3 wenet/bin/extract_encoder.py \
      --gpu $CUDA_VISIBLE_DEVICES \
      --mode ${mode} \
      --config ${config} \
      --test_data ${test_data} \
      --checkpoint ${checkpoint} \
      --batch_size 1 \
      --dict data/dict/lang_char.txt.cn \
      --result_file  ${result_file} \
      --s_decoding_chunk_size $s_decoding_chunk_size \
      --s_num_decoding_left_chunks $s_num_decoding_left_chunks \
      --simulate_streaming
