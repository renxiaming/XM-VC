## How to run ?

``` bash sh
# Step 1: prepare data from HDFS
bash prepare_data.sh

# Step 2: develop and run on local machine, make sure it works okay
conda activate wenet
bash train_local.sh

# Step 3. train on aidi cluster, this script calls train_cluster.sh
conda activate aidi
bash tools/submit.sh --train_config conf/train_xxx.yaml --dir exp/xxx --hdfs_dir aishell_xxx --job_name xxx

# Step 4. test WER on local machine(TODO)
conda activate wenet
bash decode.sh
```
