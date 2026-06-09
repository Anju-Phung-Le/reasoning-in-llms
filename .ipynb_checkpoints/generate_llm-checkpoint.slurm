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

# 1) Generate: seed dataset
python3 -m src.cli gen \
  --all-forms \
  --domains configs/domains.json \
  --out data/raw/seed_all_forms.jsonl

# 2) Expand: create full dataset
python3 -m src.cli expand \
  --in data/raw/seed_all_forms.jsonl \
  --out data/processed/dataset_all_forms.jsonl

echo "Checking files..."
wc -l data/raw/seed_all_forms.jsonl
wc -l data/processed/dataset_all_forms.jsonl

python3 - <<'PY'
import json

for fp in [
    "data/raw/seed_all_forms.jsonl",
    "data/processed/dataset_all_forms.jsonl",
]:
    lines = 0
    conclusions = 0
    with open(fp) as f:
        for line in f:
            item = json.loads(line)
            lines += 1
            conclusions += len(item["conclusions"])
    print(f"{fp}: {lines} JSONL lines, {conclusions} conclusion-level items")
PY
