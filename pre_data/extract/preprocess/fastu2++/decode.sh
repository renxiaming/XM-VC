# hdfs_dir=hdfs://hobot-bigdata/user/chengdong.liang/aishell_fast_u2++_conformer
hdfs_dir=
gpuid=-1
# dir=exp/conformer_u2pp_distill_e2etrain_noadd
dir=exp/huawei_noadd
# decode_modes="ctc_greedy_search ctc_prefix_beam_search attention_rescoring"
decode_modes="attention_rescoring"
test_sets="aishell1 SPEECHIO_ASR_ZH00002 SPEECHIO_ASR_ZH00003 SPEECHIO_ASR_ZH00004 SPEECHIO_ASR_ZH00005"

average_checkpoint=false
average_num=20

epoch=6

# decoding related parameter
s_decoding_chunk_size=4
b_decoding_chunk_size=24
s_num_decoding_left_chunks=-1
b_num_decoding_left_chunks=-1
ctc_weight=0.3
reverse_weight=0.5
beam_size=10

. path.sh
. tools/parse_options.sh

decode_checkpoint=$dir/6.pt

# Optional, download model dir from HDFS
if [ ! -z $hdfs_dir ]; then
  echo "download ckpt from k8s docker..."
  hdfs dfs -get $hdfs_dir exp
  echo "download ckpt from k8s docker... done."
fi


# Optional, do model average
if ${average_checkpoint}; then
  decode_checkpoint=$dir/avg_${average_num}.pt
  echo "do model average and final checkpoint is $decode_checkpoint"
  python3 wenet/bin/average_model.py \
    --dst_model $decode_checkpoint \
    --src_path $dir  \
    --num ${average_num} \
    --val_best
fi


echo "start decoding"
for mode in ${decode_modes}; do
{
  for test_set in $test_sets;do
  {
    # test_dir=$dir/test_${mode}_chunk${s_decoding_chunk_size}_${b_decoding_chunk_size}_left${s_num_decoding_left_chunks}_${b_num_decoding_left_chunks}_ctc${ctc_weight}_r${reverse_weight}_beam${beam_size}_ep${epoch}
    test_dir=$dir/test_${mode}_chunk${s_decoding_chunk_size}_${b_decoding_chunk_size}_left${s_num_decoding_left_chunks}_${b_num_decoding_left_chunks}_ctc${ctc_weight}_r${reverse_weight}_beam${beam_size}_ep${epoch}/${test_set}
    mkdir -p $test_dir
    python3 wenet/bin/recognize.py --gpu "$gpuid" \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type "raw" \
      --test_data data/test/$test_set/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size $beam_size \
      --batch_size 1 \
      --penalty 0.0 \
      --dict data/dict/lang_char.txt.cn \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
      ${s_num_decoding_left_chunks:+--s_num_decoding_left_chunks $s_num_decoding_left_chunks} \
      ${s_decoding_chunk_size:+--s_decoding_chunk_size $s_decoding_chunk_size} \
      ${b_num_decoding_left_chunks:+--b_num_decoding_left_chunks $b_num_decoding_left_chunks} \
      ${b_decoding_chunk_size:+--b_decoding_chunk_size $b_decoding_chunk_size} \
      --is_output_b_chunk \
      --is_lh_output_chunk
    python3 tools/compute-wer.py --char=1 --v=1 \
        data/test/text $test_dir/text > $test_dir/wer
    python3 tools/compute-wer.py --char=1 --v=1 \
        data/test/text $test_dir/text_b_chunk > $test_dir/wer_b_chunk
    python3 tools/compute-wer.py --char=1 --v=1 \
        data/test/text $test_dir/text_lh_chunk > $test_dir/wer_lh_chunk
  } done
} &
done
wait
