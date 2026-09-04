#!/bin/bash
#SBATCH --job-name=vllm-predict
#SBATCH --account=ec30
#SBATCH --time=24:00:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
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

# Tensor parallelism. Keep this at 1.
#
# A TP=2 run of Mistral-Small-24B deadlocked in the NCCL bootstrap (job
# 4094871): both workers printed "vLLM is using nccl==2.22.3" three minutes in
# and then produced nothing for the remaining 11h56m of a 12h wall -- no weight
# load, 3.1 GiB peak host RSS against a ~44 GiB checkpoint, two cores pinned
# spinning on a rendezvous that never completed. It cost 468 billing hours and
# zero predictions.
#
# On the 80 GiB A100 nothing here needs to shard. At --gpu-memory-utilization
# 0.90 (72 GiB of budget):
#   Mistral-Small-24B   23.6B x 2B = ~44 GiB weights -> ~28 GiB for KV cache
#   R1-Distill-Qwen-32B 32.8B x 2B = ~61 GiB weights -> ~11 GiB, and its KV is
#     0.25 MiB/token (64 layers x 8 KV heads x 128 head_dim x 2 x bf16), so
#     ~45k tokens: about 11 concurrent 4096-ctx sequences. Tight but workable;
#     raise utilization to 0.95 before reaching for a second GPU.
#
# Do not answer a fit problem by moving back to the 40 GiB partition with TP=2.
# That is the configuration that hung: of the A100 nodes, only gpu-[7,9] have
# NVLink and both are 80 GiB, so TP on the 40 GiB nodes is NCCL peer-to-peer
# over PCIe. If a 40 GiB run is ever unavoidable, debug it deliberately
# (NCCL_DEBUG=INFO, NCCL_P2P_DISABLE=1, NCCL_CUMEM_ENABLE=0,
# --disable-custom-all-reduce) rather than working around it with a longer wall.
#
# If a model really does outgrow one 80 GiB card, ask for TENSOR_PARALLEL=2 with
# --gpus-per-node=a100_80:2 and target gpu-[7,9] (--nodelist), where the shard
# traffic crosses NVLink instead of PCIe.
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

# Caller-supplied --tensor-parallel-size wins; otherwise inject the default.
TP_ARGS=(--tensor-parallel-size "$TENSOR_PARALLEL")
for arg in "$@"; do
  case "$arg" in
    --tensor-parallel-size|--tensor-parallel-size=*) TP_ARGS=() ;;
  esac
done

# Hard cap on the predict step, comfortably under the wall so a hang still
# leaves time for this script to say so. A stall in engine startup burns the
# whole allocation silently otherwise.
PREDICT_TIMEOUT="${PREDICT_TIMEOUT:-23h}"

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

# --kill-after sends SIGKILL if the engine ignores SIGTERM; a wedged NCCL
# worker often does. errexit would swallow the distinction between "timed out"
# and "crashed", so check the status by hand.
set +o errexit
timeout --kill-after=2m "$PREDICT_TIMEOUT" \
python3 -m src.cli predict \
  --backend vllm \
  --model "$MODEL" \
  --data "$DATA" \
  --out "$PRED_FP" \
  ${TP_ARGS[@]+"${TP_ARGS[@]}"} \
  "$@"
PREDICT_STATUS=$?
set -o errexit

if [ "$PREDICT_STATUS" -eq 124 ] || [ "$PREDICT_STATUS" -eq 137 ]; then
  echo "ERROR: predict exceeded PREDICT_TIMEOUT=$PREDICT_TIMEOUT and was killed." >&2
  echo "  If the last vLLM log line is an NCCL version banner, it deadlocked in" >&2
  echo "  engine startup rather than running slowly -- a longer wall will not help." >&2
  exit "$PREDICT_STATUS"
elif [ "$PREDICT_STATUS" -ne 0 ]; then
  exit "$PREDICT_STATUS"
fi

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
