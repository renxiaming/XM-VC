export PYTHONPATH=./:$PYTHONPATH
python ./wenet/bin/export_jit.py --config ./exp/huawei_noadd/train.yaml --checkpoint /home/work_nfs6/yzli/workshop/wenet_fastu2pp_distill_full/examples/aishell/exp/huawei_new/15.pt --output_file ./big.pt
