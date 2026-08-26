#!/bin/bash
#SBATCH --job-name=deepseek-nocot-v2
#SBATCH --account=ec403
#SBATCH --time=04:00:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --exclude=gpu-14,gpu-15,gpu-16,gpu-17
#SBATCH --output=logs/deepseek_nocot_v2_%j.out
#SBATCH --error=logs/deepseek_nocot_v2_%j.err

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset
cd "$SLURM_SUBMIT_DIR"
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8

# WORKAROUND for protobuf / sentencepiece mismatch
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# DeepSeek no-CoT rerun with v2 parser (case-sensitive A/B/C, honest unparsed handling)
# Greedy decoding on full all_forms dataset for Table 4.1.
python3 -m src.cli predict \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --data data/processed/dataset_all_forms.jsonl \
  --out data/predictions/deepseek_8b_all_forms_v2_predictions.jsonl