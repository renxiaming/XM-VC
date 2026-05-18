#!/bin/bash
[ -f ./path.sh ] && . ./path.sh
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
echo "Number of GPUs you need: 1"
 cd /home/work_nfs6/ypjiang/code/aishell
CUDA_VISIBLE_DEVICES=`nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print NR-1,$0}' | sort -k 2 -n -r | cut -d ' ' -f 1 | head -1 | perl -pe 'chop if eof' | tr '\n' ','`
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" extract.sh 
EOF
) >./fastfull
time1=`date +"%s"`
 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES extract.sh  ) 2>>./fastfull >>./fastfull
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>./fastfull
echo '#' Finished at `date` with status $ret >>./fastfull
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.37965
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/fastfull -l gpu=1 -q tts.q   /home/work_nfs6/ypjiang/code/aishell/./q/fastfull.sh >>./q/fastfull 2>&1
