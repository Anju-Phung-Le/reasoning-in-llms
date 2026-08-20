#!/bin/bash
#SBATCH --job-name=deepseek-cot1024
#SBATCH --account=ec403
#SBATCH --time=10:00:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --exclude=gpu-14,gpu-15,gpu-16
#SBATCH --output=logs/deepseek_cot1024_%j.out
#SBATCH --error=logs/deepseek_cot1024_%j.err

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8

# WORKAROUND for protobuf / sentencepiece mismatch
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Optional first arg: seed. If provided, use sampled decoding and embed the
# seed in output filenames so parallel seed runs don't overwrite each other.
#   sbatch predict_deepseek_cot1024.slurm          # greedy, default filenames
#   sbatch predict_deepseek_cot1024.slurm 0        # sampled, seed 0
#   for s in 0 1 2; do sbatch predict_deepseek_cot1024.slurm $s; done
SEED="${1:-}"
if [ -n "$SEED" ]; then
  SAMPLE_FLAGS="--do-sample --seed $SEED"
  TAG="_seed${SEED}"
else
  SAMPLE_FLAGS=""
  TAG=""
fi

PRED_FP="data/predictions/deepseek_8b_cot1024_sub64${TAG}_predictions.jsonl"
EVAL_FP="outputs/deepseek_8b_cot1024_sub64${TAG}_outputs_log.csv"

# DeepSeek CoT on 64-item slice (512 predictions) with a REAL token budget.
# The 384-token run truncated ~85% of chains; 1024 should let it actually conclude.
python3 -m src.cli predict \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --data data/processed/dataset_sub64.jsonl \
  --out "$PRED_FP" \
  --cot --cot-max-new-tokens 2048 \
  $SAMPLE_FLAGS

# Eval runs on CPU quickly; safe to include here.
python3 -m src.cli eval \
  --gold data/processed/dataset_sub64.jsonl \
  --pred "$PRED_FP" \
  --out "$EVAL_FP"