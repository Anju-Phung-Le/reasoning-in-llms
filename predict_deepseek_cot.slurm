#!/bin/bash
#SBATCH --job-name=deepseek-cot
#SBATCH --account=ec403
#SBATCH --time=12:00:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --exclude=gpu-14,gpu-15,gpu-16
#SBATCH --output=logs/deepseek_cot_%j.out
#SBATCH --error=logs/deepseek_cot_%j.err

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

# DeepSeek CoT (reasoning model — larger token budget for <think> traces)
python3 -m src.cli predict \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --data data/processed/dataset_all_forms.jsonl \
  --out data/predictions/deepseek_8b_cot_all_forms_predictions.jsonl \
  --cot --cot-max-new-tokens 512

# Eval runs on CPU quickly; safe to include here.
python3 -m src.cli eval \
  --gold data/processed/dataset_all_forms.jsonl \
  --pred data/predictions/deepseek_8b_cot_all_forms_predictions.jsonl \
  --out outputs/deepseek_8b_cot_all_forms_outputs_log.csv
