#!/usr/bin/env bash
# Re-run eval for every predictions file with a unified metric schema.
# Gold dataset is chosen by filename convention:
#   *_all_forms_*  -> dataset_all_forms.jsonl
#   *_sub256_*     -> dataset_sub256.jsonl
#   *_sub64_*      -> dataset_sub64.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p outputs

for pred in data/predictions/*_predictions.jsonl; do
    # skip the *_conversations.jsonl trace files
    case "$pred" in *_conversations.jsonl) continue ;; esac

    base=$(basename "$pred" _predictions.jsonl)   # e.g. mistral_7b_cot_sub256

    case "$base" in
        *_all_forms) gold=data/processed/dataset_all_forms.jsonl ;;
        *_sub256)    gold=data/processed/dataset_sub256.jsonl ;;
        *_sub64)     gold=data/processed/dataset_sub64.jsonl ;;
        *) echo "!! skip $base (unknown split)"; continue ;;
    esac

    out=outputs/${base}_outputs_log.csv
    echo "==> $base"
    python -m src.cli eval --gold "$gold" --pred "$pred" --out "$out" >/dev/null
done

echo
echo "Done. CSVs in outputs/ (all share the same 19-column schema)."
ls -1 outputs/*_outputs_log.csv
