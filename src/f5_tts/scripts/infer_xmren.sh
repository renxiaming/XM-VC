source /home/work_nfs16/xmren/miniconda3/bin/activate /home/work_nfs16/xmren/miniconda3/envs/vc

# cd /home/node60_tmpdata/xmren/streamingfm/src/f5_tts/infer
# python infer_streaming_official.py \
#     --wav_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testold \
#     --token_path /home/node60_tmpdata/hkxie/workspace/streamingfm/testold \
#     --output_path /home/node60_tmpdata/xmren/streamingfm/xmren_inferout/demo1 \
#     --ckpt_file /home/node60_tmpdata/xmren/streamingfm/ckpts/F5TTS_fm_10ms_ecapa_tdnn_spk_hifigancosyvoice1/model_350000.pt \
#     --device 2 &


cd /home/node60_tmpdata/xmren/streamingfm/src/f5_tts/infer
python infer_streaming_official.py \
    --wav_path /home/node60_tmpdata/xmren/streamingfm/xmren_inferout/test_data/ref_wav \
    --token_path /home/node60_tmpdata/xmren/streamingfm/xmren_inferout/test_data/s1token \
    --output_path /home/node60_tmpdata/xmren/streamingfm/xmren_inferout/demo2 \
    --ckpt_file /home/node60_tmpdata/xmren/streamingfm/ckpts/F5TTS_fm_10ms_ecapa_tdnn_spk_hifigancosyvoice1/model_350000.pt \
    --device 2 &