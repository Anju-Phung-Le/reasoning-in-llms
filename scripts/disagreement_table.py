import json
import csv
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

GOLD_FILE = Path("data/processed/dataset_all_forms.jsonl")

MODEL_FILES = {
    "Flan-T5 Small": Path(
        "data/predictions/flan_t5_small4_all_forms_predictions.jsonl"
    ),
    "Flan-T5 Base": Path(
        "data/predictions/flan_t5_base3_all_forms_predictions.jsonl"
    ),
    "Flan-T5 Large": Path(
        "data/predictions/flan_t5_large3_all_forms_predictions.jsonl"
    ),
    "Mistral-7B": Path(
        "data/predictions/mistral_7b_all_forms_predictions.jsonl"
    ),
    "DeepSeek-8B": Path(
        "data/predictions/deepseek_8b_all_forms_predictions.jsonl"
    ),
}

# Main analysis uses the neutral role only
ROLE = "none"

OUTPUT_DIR = Path("outputs/report")
CSV_OUTPUT = OUTPUT_DIR / "disagreement_none.csv"
LATEX_OUTPUT = OUTPUT_DIR / "disagreement_none.tex"


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
# Analyse one model
# -------------------------------------------------------------------

def analyse_model(prediction_file, gold):
    """
    Analyse only question instances for which Classical Logic and WCS
    assign different gold labels.
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

                # ---------------------------------------------------
                # Keep ONLY disagreement cases
                # ---------------------------------------------------
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

    rows = []

    for model, prediction_file in MODEL_FILES.items():

        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {prediction_file}"
            )

        counts = analyse_model(
            prediction_file,
            gold,
        )

        n = counts["n"]

        classical_pct = percentage(
            counts["classical"],
            n,
        )

        wcs_pct = percentage(
            counts["wcs"],
            n,
        )

        neither_pct = percentage(
            counts["neither"],
            n,
        )

        unparsed_pct = percentage(
            counts["unparsed"],
            n,
        )

        # Positive delta = more responses matched WCS
        delta = wcs_pct - classical_pct

        row = {
            "model": model,
            "n_disagreement": n,
            "classical_match_pct": round(classical_pct, 2),
            "wcs_match_pct": round(wcs_pct, 2),
            "neither_pct": round(neither_pct, 2),
            "unparsed_pct": round(unparsed_pct, 2),
            "wcs_minus_classical_pp": round(delta, 2),
        }

        rows.append(row)

    # ----------------------------------------------------------------
    # Terminal output
    # ----------------------------------------------------------------

    print()

    print(
        f"{'Model':<20}"
        f"{'N':>8}"
        f"{'Classical %':>15}"
        f"{'WCS %':>12}"
        f"{'Neither %':>14}"
        f"{'Unparsed %':>14}"
        f"{'Delta':>10}"
    )

    print("-" * 93)

    for row in rows:

        print(
            f"{row['model']:<20}"
            f"{row['n_disagreement']:>8}"
            f"{row['classical_match_pct']:>15.2f}"
            f"{row['wcs_match_pct']:>12.2f}"
            f"{row['neither_pct']:>14.2f}"
            f"{row['unparsed_pct']:>14.2f}"
            f"{row['wcs_minus_classical_pp']:>+10.2f}"
        )

    # ----------------------------------------------------------------
    # Save complete CSV
    # ----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved CSV table to: {CSV_OUTPUT}")

    # ----------------------------------------------------------------
    # Create thesis-ready LaTeX table
    # ----------------------------------------------------------------

    df = pd.DataFrame(rows)

    # Select only useful columns for the thesis.
    # Unparsed is kept in the CSV but omitted here because it is 0%.
    latex_df = df[
        [
            "model",
            "n_disagreement",
            "classical_match_pct",
            "wcs_match_pct",
            "neither_pct",
            "wcs_minus_classical_pp",
        ]
    ].copy()

    latex_df.columns = [
        "Model",
        "$N$",
        "Classical (\\%)",
        "WCS (\\%)",
        "Neither (\\%)",
        "$\\Delta$ (pp)",
    ]

    # Format numbers exactly as we want them to appear in the thesis
    latex_df["Classical (\\%)"] = latex_df[
        "Classical (\\%)"
    ].map(lambda x: f"{x:.2f}")

    latex_df["WCS (\\%)"] = latex_df[
        "WCS (\\%)"
    ].map(lambda x: f"{x:.2f}")

    latex_df["Neither (\\%)"] = latex_df[
        "Neither (\\%)"
    ].map(lambda x: f"{x:.2f}")

    latex_df["$\\Delta$ (pp)"] = latex_df[
        "$\\Delta$ (pp)"
    ].map(lambda x: f"{x:+.2f}")

    latex = latex_df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrr",
        position="htbp",
        caption=(
            "Model responses on question instances where Classical Logic "
            "and Weak Completion Semantics (WCS) assign different gold "
            "labels in the neutral role condition. "
            "$N$ denotes the number of disagreement instances and "
            "$\\Delta$ denotes the WCS match rate minus the Classical "
            "match rate in percentage points."
        ),
        label="tab:disagreement-neutral",
    )

    LATEX_OUTPUT.write_text(
        latex,
        encoding="utf-8",
    )

    print(f"Saved LaTeX table to: {LATEX_OUTPUT}")


if __name__ == "__main__":
    main()