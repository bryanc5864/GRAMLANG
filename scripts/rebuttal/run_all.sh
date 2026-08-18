#!/usr/bin/env bash
# train the readout ladder for every (model, dataset) pair, sharded over GPUs
set -u
cd /home/bcheng/grammar
export PYTHONPATH=/home/bcheng/grammar

MODELS=(hyenadna dnabert2 nt)
DATASETS=(jores agarwal klein vaishnav inoue)
GPUS=(1 3 0)

i=0
for mi in "${!MODELS[@]}"; do
  m=${MODELS[$mi]}
  g=${GPUS[$mi]}
  (
    for d in "${DATASETS[@]}"; do
      echo "$m / $d (gpu $g)"
      CUDA_VISIBLE_DEVICES=$g python -u scripts/rebuttal/train_readouts.py \
        --model "$m" --dataset "$d" --epochs 60 2>&1 | grep -viE "warn|newly initialized|TRAIN this model"
    done
  ) > results/rebuttal/train_${m}.log 2>&1 &
  i=$((i+1))
done
wait
echo ALL_TRAINING_DONE
