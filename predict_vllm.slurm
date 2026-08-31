#!/bin/bash
#SBATCH --job-name=vllm-predict
#SBATCH --account=ec30
#SBATCH --time=02:00:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/vllm_predict_%j.out
#SBATCH --error=logs/vllm_predict_%j.err

# Usage:
#   sbatch predict_vllm.slurm <model> <dataset> <tag> [extra predict flags...]
#
#   sbatch predict_vllm.slurm DeepSeek-R1-Distill-Llama-8B \
#          data/processed/dataset_sub64.jsonl smoke --cot --cot-max-new-tokens 2048
#
# Variance sweep, five draws in ONE job (one model load, all sequences batched
# together). Writes <tag>_predictions_sample0.jsonl ... _sample4.jsonl:
#   sbatch predict_vllm.slurm DeepSeek-R1-Distill-Llama-8B \
#     data/processed/dataset_sub64.jsonl cot1024_sub64_vllm \
#     --cot --cot-max-new-tokens 2048 --max-model-len 4096 \
#     --do-sample --seed 0 --n-samples 5
#
# Independently seeded runs instead (five jobs, five model loads) -- only if a
# table needs separate seeds rather than draws from one sampler state:
#   for s in 0 1 2 3 4; do
#     sbatch predict_vllm.slurm DeepSeek-R1-Distill-Llama-8B \
#       data/processed/dataset_sub64.jsonl "cot1024_sub64_vllm_seed${s}" \
#       --cot --cot-max-new-tokens 2048 --max-model-len 4096 --do-sample --seed "$s"
#   done

set -o errexit
set -o nounset
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL="${1:?usage: sbatch predict_vllm.slurm <model> <dataset> <tag> [flags...]}"
DATA="${2:?missing dataset}"
TAG="${3:?missing tag}"
shift 3

# Where the local checkpoints live. Override by exporting MODEL_ROOT before
# sbatch, or edit here if they move.
MODEL_ROOT="${MODEL_ROOT:-/cluster/work/projects/ec30/emilpo/models}"

# <model> may be a bare directory name under MODEL_ROOT ("DeepSeek-R1-Distill-Llama-8B"),
# an absolute path, or a hub id. Resolve the first form; leave the others alone.
if [ ! -d "$MODEL" ] && [ -d "$MODEL_ROOT/$MODEL" ]; then
  MODEL="$MODEL_ROOT/$MODEL"
fi

if [ -d "$MODEL" ]; then
  echo "Model (local dir): $MODEL"
else
  echo "Model (hub id): $MODEL"
  echo "  Resolved from the HF cache under HF_HOME, offline. If it is not"
  echo "  cached, the job fails immediately rather than reaching for the"
  echo "  network -- download it first on a login node with:"
  echo "      huggingface-cli download $MODEL"
fi

module purge

# Ignore ~/.local: a stray `pip install --user` there silently outranks the
# module-provided torch/transformers/vllm, for every run, invisibly.
export PYTHONNOUSERSITE=1

module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-accelerate/1.9.0-foss-2024a-Python-3.12.3
module load Transformers/4.57.1-gfbf-2024a
module load nlpl-vllm/0.8.2-foss-2024a-Python-3.12.3

# vLLM forks its engine worker; if the parent already touched CUDA the fork
# cannot re-initialize it. Force spawn so the worker starts fresh.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Shard-loading and CUDA-graph-capture progress bars are ~150 lines of carriage
# returns per startup; the INFO summary after each phase carries the same signal.
export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

# The existing HF cache: this directory already holds hub/ and xet/, which is
# exactly the layout HF_HOME expects. Keeps hub metadata off $HOME, which is
# small and shared.
export HF_HOME="${HF_HOME:-/cluster/work/projects/ec30/emilpo}"

# Compute nodes have no route to huggingface.co, where a hub lookup hangs
# rather than failing. Offline mode resolves a model id against the cache
# above instead, and errors immediately when it is not there. Download new
# models on a login node first: huggingface-cli download <id>
export HF_HUB_OFFLINE=1

# Fail now, not twenty minutes into a model load.
python3 -c '
import sys, torch, transformers, vllm
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
      f"transformers {transformers.__version__} | vllm {vllm.__version__}")
assert torch.cuda.is_available(), "torch reports no CUDA: CPU-only build, or no GPU allocated"
'

PRED_FP="data/predictions/${TAG}_predictions.jsonl"

python3 -m src.cli predict \
  --backend vllm \
  --model "$MODEL" \
  --data "$DATA" \
  --out "$PRED_FP" \
  "$@"

# With --n-samples the predict step writes _sample0.._sampleN-1 in place of
# PRED_FP itself, so evaluate whatever it actually produced. Each sample gets
# its own CSV; eval order does not matter.
shopt -s nullglob
PRED_FILES=("$PRED_FP" data/predictions/"${TAG}"_predictions_sample*.jsonl)
shopt -u nullglob

for f in "${PRED_FILES[@]}"; do
  [ -f "$f" ] || continue
  base="$(basename "$f" .jsonl)"   # <tag>_predictions[_sampleN]
  base="${base/_predictions/}"     # -> <tag>[_sampleN]
  python3 -m src.cli eval \
    --gold "$DATA" \
    --pred "$f" \
    --out "outputs/${base}_outputs_log.csv"
done
