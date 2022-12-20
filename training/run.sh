#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a40:2
#SBATCH --account=shrikann_35

eval "$(conda shell.bash hook)"
conda activate /home1/rajatheb/.conda/envs/ear

dataset=combined
model_type=vit
model_size=base

dataset_mean=-7.625
dataset_std=2.36
n_class=2

data_dir=./data
tr_data=$data_dir/${dataset}_train.json
val_data=$data_dir/${dataset}_val.json
test_data=$data_dir/${dataset}_test.json

label_csv=$data_dir/classes.csv

freqm=0
timem=0
lr=1e-4
batch_size=64
epoch=15
patch_freq=128
patch_time=6

exp_dir=${dataset}_exp/${model_type}-${model_size}-f${patch_freq}-t${patch_time}/b${batch_size}-lr${lr}

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir/models

CUDA_CACHE_DISABLE=1 python -W ignore ./run.py \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${test_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} --lr ${lr} --n-epochs ${epoch} --batch-size ${batch_size} \
--patch_freq ${patch_freq} --patch_time ${patch_time} \
--dataset-mean ${dataset_mean} --dataset-std ${dataset_std} --freqm ${freqm} --timem ${timem} \
--model_type ${model_type} --model_size ${model_size} --save_model True >> ${exp_dir}/log.txt
