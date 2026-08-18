#!/usr/bin/env bash
# corrected census with the full readout ladder, sharded over GPUs
set -u
cd /home/bcheng/grammar
export PYTHONPATH=/home/bcheng/grammar

MODELS=(hyenadna dnabert2 nt)
DATASETS=(jores agarwal klein vaishnav inoue)
GPUS=(1 3 0)

for mi in "${!MODELS[@]}"; do
  m=${MODELS[$mi]}
  g=${GPUS[$mi]}
  (
    for d in "${DATASETS[@]}"; do
      echo "census $m / $d (gpu $g)"
      CUDA_VISIBLE_DEVICES=$g python -u scripts/rebuttal/run_census_v2.py \
        --model "$m" --dataset "$d" --n-enhancers 300 --n-draws 40 --n-perm 2000 \
        2>&1 | grep --line-buffered -viE "warn|newly initialized|TRAIN this model"
    done
  ) > results/rebuttal/census_${m}.log 2>&1 &
done
wait
echo ALL_CENSUS_DONE
