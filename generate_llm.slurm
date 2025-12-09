#!/bin/bash
#SBATCH --job-name=reason-llm
#SBATCH --account=ec403
#SBATCH --time=04:00:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --exclude=gpu-14,gpu-15,gpu-16

source ~/.bashrc || true

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

# 1) Generate: seed dataset
python3 -m src.cli gen \
  --n 150 \
  --templates configs/syllogism_templates.json \
  --domains configs/domains.json \
  --out data/raw/seed.jsonl

# 2) Expand: create full dataset
python3 -m src.cli expand \
  --in data/raw/seed.jsonl \
  --out data/processed/dataset_v1.jsonl

# 3) Predict: run Mistral-7B-Instruct-v0.3 on dataset
python3 -m src.cli predict \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --data data/processed/dataset_v1.jsonl \
  --out data/predictions/mistral_7b_instruct_v0_3_predictions.jsonl

# 4) Eval: evaluate Mistral predictions
python3 -m src.cli eval \
  --gold data/processed/dataset_v1.jsonl \
  --pred data/predictions/mistral_7b_instruct_v0_3_predictions.jsonl \
  --out outputs/mistral_7b_instruct_v0_3_outputs_log.csv
