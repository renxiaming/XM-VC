export WENET_DIR=$PWD/../..
MAIN_ROOT=/home/work_nfs4_ssd/yhliang/workspace/espnet_multitalker
KALDI_ROOT=$MAIN_ROOT/tools/kaldi
ANACONDA_ROOT=/home/environment/pkchen/anaconda3
export PATH=$PWD:${BUILD_DIR}:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

[ ! -d utils ] && ln -s $KALDI_ROOT/egs/wsj/s5/utils
[ ! -d steps ] && ln -s $KALDI_ROOT/egs/wsj/s5/steps

export LC_ALL=C

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
source $ANACONDA_ROOT/bin/activate wenet_27
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=./:$PYTHONPATH

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export USE_SYSTEM_NCCL=1


