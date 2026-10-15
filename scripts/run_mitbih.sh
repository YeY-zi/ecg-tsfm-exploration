#!/usr/bin/env bash
set -euo pipefail
# Run from repository root, after activating the Python environment.
python -m ecgts.run --prepare --data data/mitdb
for model in persistence train_mean cnn patchtst; do
  python -m ecgts.run --data data/mitdb --model "$model" \
    --context 256 --horizon 64 --stride 512 --epochs 5 --seed 42 \
    --batch-size 64 --threads 2 --out "runs/mitbih-${model}-seed42"
done
