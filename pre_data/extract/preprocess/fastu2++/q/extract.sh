#!/bin/bash
[ -f ./path.sh ] && . ./path.sh
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
echo "Number of GPUs you need: 1"
 cd /home/work_nfs6/ypjiang/code/aishell
CUDA_VISIBLE_DEVICES=`nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print NR-1,$0}' | sort -k 2 -n -r | cut -d ' ' -f 1 | head -1 | perl -pe 'chop if eof' | tr '\n' ','`
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" ./extract.sh 
EOF
) >extract.log
time1=`date +"%s"`
 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ./extract.sh  ) 2>>extract.log >>extract.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>extract.log
echo '#' Finished at `date` with status $ret >>extract.log
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.19204
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/extract.log -l gpu=1 -q tts.q   /home/work_nfs6/ypjiang/code/aishell/./q/extract.sh >>./q/extract.log 2>&1
