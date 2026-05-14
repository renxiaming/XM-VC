source /home/work_nfs19/hkxie/environment/anaconda3/bin/activate /home/work_nfs19/hkxie/environment/anaconda3/envs/covomix

cd /home/node60_tmpdata/hkxie/workspace/streamingfm/src/f5_tts/infer


# python infer_streaming_official.py \
#     --wav_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test \
#     --token_path /home/node60_tmpdata/hkxie/workspace/streamingfm/test \
#     --output_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testout/output_200k_official

# python /home/node60_tmpdata/hkxie/workspace/streamingfm/src/f5_tts/infer/infer_streaming_official_novc.py \
#     --wav_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testold \
#     --token_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testold \
#     --output_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testout/output_200k_novc &


python /home/node60_tmpdata/hkxie/workspace/streamingfm/src/f5_tts/infer/infer_streaming_official_novc.py \
    --wav_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testold \
    --token_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testold \
    --output_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testout/output_300k_novc \
    --ckpt_file /home/node60_tmpdata/hkxie/workspace/streamingfm/ckpts/F5TTS_fm_10ms_ecapa_tdnn_spk_hifigancosyvoice1/model_300000.pt \
    --device 0 &

python infer_streaming_official.py \
    --wav_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test \
    --token_path /home/node60_tmpdata/hkxie/workspace/streamingfm/test \
    --output_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testout/output_300k_official \
    --ckpt_file /home/node60_tmpdata/hkxie/workspace/streamingfm/ckpts/F5TTS_fm_10ms_ecapa_tdnn_spk_hifigancosyvoice1/model_300000.pt \
    --device 1 &

python /home/node60_tmpdata/hkxie/workspace/streamingfm/src/f5_tts/infer/infer_streaming_official_novc.py \
    --wav_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testold \
    --token_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testold \
    --output_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testout/output_400k_novc \
    --ckpt_file /home/node60_tmpdata/hkxie/workspace/streamingfm/ckpts/F5TTS_fm_10ms_ecapa_tdnn_spk_hifigancosyvoice1/model_400000.pt \
    --device 2 &

python infer_streaming_official.py \
    --wav_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test \
    --token_path /home/node60_tmpdata/hkxie/workspace/streamingfm/test \
    --output_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testout/output_400k_official \
    --ckpt_file /home/node60_tmpdata/hkxie/workspace/streamingfm/ckpts/F5TTS_fm_10ms_ecapa_tdnn_spk_hifigancosyvoice1/model_400000.pt \
    --device 3 &

wait 