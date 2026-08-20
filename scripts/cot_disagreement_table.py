"""
CoT-mode variant of scripts/disagreement_table.py.

Reports how models resolve items where classical logic and WCS (mental-model
logic) disagree — but based on CoT (chain-of-thought) predictions instead of
the forced-choice (argmax over A/B/C logits) predictions.

Because the CoT prediction files cover different subsets (sub64 vs sub256),
we filter each model's predictions to the intersection of item IDs across
all included model files so cross-model comparison is fair.

Note: the DeepSeek CoT sub256 run used a 384-token budget which truncated
~85% of reasoning chains, so it is intentionally excluded. Only the cleanly
parsed CoT1024/sub64 DeepSeek run is used here.

Outputs: outputs/report/disagreement_none_cot.{csv,tex}
"""

import json
import csv
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

GOLD_FILE = Path("data/processed/dataset_all_forms.jsonl")

MODEL_FILES = {
    "DeepSeek-8B (CoT, 1024)": Path(
        "data/predictions/deepseek_8b_cot1024_sub64_predictions.jsonl"
    ),
    "Mistral-7B (CoT, 128)": Path(
        "data/predictions/mistral_7b_cot_sub256_predictions.jsonl"
    ),
}

# Main analysis uses the neutral role only.
ROLE = "none"

OUTPUT_DIR = Path("outputs/report")
CSV_OUTPUT = OUTPUT_DIR / "disagreement_none_cot.csv"
LATEX_OUTPUT = OUTPUT_DIR / "disagreement_none_cot.tex"


# -------------------------------------------------------------------
# Load gold dataset
# -------------------------------------------------------------------

def load_gold(path):
    """Load processed dataset and index records by item ID."""

    gold = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gold[item["id"]] = item

    return gold


# -------------------------------------------------------------------
# Compute intersection of item IDs across all model prediction files
# -------------------------------------------------------------------

def prediction_ids(prediction_file):
    with open(prediction_file, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f}


def common_item_ids(model_files):
    id_sets = [prediction_ids(p) for p in model_files.values()]
    return set.intersection(*id_sets)


# -------------------------------------------------------------------
# Analyse one model on a restricted set of item IDs
# -------------------------------------------------------------------

def analyse_model(prediction_file, gold, allowed_ids):
    """
    Analyse only question instances for which Classical Logic and WCS
    assign different gold labels, restricted to allowed_ids.
    """

    counts = {
        "n": 0,
        "classical": 0,
        "wcs": 0,
        "neither": 0,
        "unparsed": 0,
    }

    with open(prediction_file, encoding="utf-8") as f:

        for line in f:
            prediction = json.loads(line)

            if prediction["id"] not in allowed_ids:
                continue

            item = gold.get(prediction["id"])

            if item is None:
                continue

            # Neutral persona only
            if item["role"] != ROLE:
                continue

            for conclusion_code, pred_conclusion in prediction[
                "conclusions"
            ].items():

                gold_conclusion = item["conclusions"].get(
                    conclusion_code
                )

                if gold_conclusion is None:
                    continue

                classical = gold_conclusion["classical"]
                wcs = gold_conclusion["wcs"]

                # Keep ONLY disagreement cases
                if classical == wcs:
                    continue

                counts["n"] += 1

                pred = pred_conclusion.get("pred_index", -1)

                if pred == classical:
                    counts["classical"] += 1

                elif pred == wcs:
                    counts["wcs"] += 1

                elif pred in (0, 1, 2):
                    counts["neither"] += 1

                else:
                    counts["unparsed"] += 1

    return counts


# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

def percentage(count, total):
    if total == 0:
        return 0.0

    return 100 * count / total


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    gold = load_gold(GOLD_FILE)

    for model, prediction_file in MODEL_FILES.items():
        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {prediction_file}"
            )

    allowed_ids = common_item_ids(MODEL_FILES)
    print(
        f"Common item IDs across {len(MODEL_FILES)} model file(s): "
        f"{len(allowed_ids)}"
    )

    rows = []

    for model, prediction_file in MODEL_FILES.items():

        counts = analyse_model(
            prediction_file,
            gold,
            allowed_ids,
        )

        n = counts["n"]

        rows.append(
            {
                "model": model,
                "n_disagreement": n,
                "classical_match_pct": round(
                    percentage(counts["classical"], n), 2
                ),
                "wcs_match_pct": round(
                    percentage(counts["wcs"], n), 2
                ),
                "neither_pct": round(
                    percentage(counts["neither"], n), 2
                ),
                "unparsed_pct": round(
                    percentage(counts["unparsed"], n), 2
                ),
                "wcs_minus_classical_pp": round(
                    percentage(counts["wcs"], n)
                    - percentage(counts["classical"], n),
                    2,
                ),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    df.to_latex(
        LATEX_OUTPUT,
        index=False,
        float_format="%.2f",
        caption=(
            "Cross-model behaviour on disagreement items (CoT mode). "
            f"Filtered to {len(allowed_ids)} common items."
        ),
        label="tab:disagreement_none_cot",
    )

    print(f"Wrote {CSV_OUTPUT}")
    print(f"Wrote {LATEX_OUTPUT}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
