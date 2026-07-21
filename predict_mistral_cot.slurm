#!/bin/bash
#SBATCH --job-name=mistral-cot
#SBATCH --account=ec403
#SBATCH --time=02:00:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --exclude=gpu-14,gpu-15,gpu-16
#SBATCH --output=logs/mistral_cot_%j.out
#SBATCH --error=logs/mistral_cot_%j.err

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

# Mistral CoT on SUBSAMPLED dataset (256 items × 8 = 2048 predictions).
# Instruct model — 128 tokens is plenty for short reasoning + letter.
python3 -m src.cli predict \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --data data/processed/dataset_sub256.jsonl \
  --out data/predictions/mistral_7b_cot_sub256_predictions.jsonl \
  --cot --cot-max-new-tokens 128

# Eval runs on CPU quickly; safe to include here.
python3 -m src.cli eval \
  --gold data/processed/dataset_sub256.jsonl \
  --pred data/predictions/mistral_7b_cot_sub256_predictions.jsonl \
  --out outputs/mistral_7b_cot_sub256_outputs_log.csv
