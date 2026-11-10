"""Rebuild tables/final2_disagreement_table.tex from the full-dataset supervisor predictions.

Run from any directory with: python3 scripts/final2_disagreement_table.py
Uses the original scoring and sample-standard-deviation conventions.
"""

import csv
import json
import math
from pathlib import Path

from aggregate_seeds import mean_std, score_seed

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data/processed/dataset_all_forms.jsonl"
PRED_DIR = ROOT / "data/predictions"
TEX_PATH = ROOT / "tables/final2_disagreement_table.tex"
CSV_PATH = ROOT / "outputs/report/final2_per_sample.csv"
METRICS = ("classical_match_pct", "wcs_match_pct", "neither_pct", "delta")
BASELINES = {
    "Flan-T5 Small": "flan_t5_small4",
    "Flan-T5 Base": "flan_t5_base3",
    "Flan-T5 Large": "flan_t5_large3",
}
# Short model names used in the thesis tables and figures.
DISPLAY_NAMES = {
    "Flan-T5 Small": "Flan-T5 Small",
    "Flan-T5 Base": "Flan-T5 Base",
    "Flan-T5 Large": "Flan-T5 Large",
    "Mistral-7B": "Mistral-7B",
    "Llama-3.1-8B": "Llama-8B",
    "OLMo-2-13B": "OLMo-13B",
    "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-Qwen-14B",
    "Mistral-Small-24B": "Mistral-24B",
    "DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-Qwen-32B",
}
COT_MODELS = {
    "Mistral-7B": "mistral_7b",
    "Llama-3.1-8B": "llama31_8b",
    "OLMo-2-13B": "olmo2_13b",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek_8b",
    "DeepSeek-R1-Distill-Qwen-14B": "deepseek14b",
    "Mistral-Small-24B": "mistral24b",
    "DeepSeek-R1-Distill-Qwen-32B": "deepseek_qwen_32B",
}


def load_unique(path):
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"{path}: duplicate ID {row['id']}")
            rows[row["id"]] = row
    return rows


def validate_predictions(path, gold):
    predictions = load_unique(path)
    if predictions.keys() != gold.keys():
        missing = gold.keys() - predictions.keys()
        extra = predictions.keys() - gold.keys()
        raise ValueError(f"{path}: {len(missing)} missing and {len(extra)} extra IDs")
    for iid, row in predictions.items():
        item = gold[iid]
        for field in ("role", "form", "domain", "premises"):
            if row.get(field) != item.get(field):
                raise ValueError(f"{path}: {field} differs for {iid}")
        if row["conclusions"].keys() != item["conclusions"].keys():
            raise ValueError(f"{path}: conclusion coverage differs for {iid}")
        for code, conclusion in row["conclusions"].items():
            if conclusion.get("question") != item["conclusions"][code].get("question"):
                raise ValueError(f"{path}: question differs for {iid}/{code}")
            label = conclusion.get("pred_index")
            if type(label) is not int or label not in (-1, 0, 1, 2):
                raise ValueError(f"{path}: unexpected pred_index for {iid}/{code}: {label}")
            for framework in ("classical", "wcs"):
                if conclusion.get(framework) != item["conclusions"][code][framework]:
                    raise ValueError(f"{path}: {framework} gold differs for {iid}/{code}")
    return predictions


def prediction_files():
    files = {
        model: [PRED_DIR / "all_forms" / f"{prefix}_all_forms_predictions.jsonl"]
        for model, prefix in BASELINES.items()
    }
    files.update({
        model: [
            PRED_DIR / "final" / f"{prefix}_cot_allforms_predictions_sample{i}.jsonl"
            for i in range(5)
        ]
        for model, prefix in COT_MODELS.items()
    })
    return files


def load_runs(models=None):
    """Return (gold, {model: [(path, predictions_by_id), ...]}) after validation."""
    gold = load_unique(GOLD_PATH)
    runs = {
        model: [(path, validate_predictions(path, gold)) for path in paths]
        for model, paths in prediction_files().items()
        if models is None or model in models
    }
    return gold, runs


def main():
    gold = load_unique(GOLD_PATH)
    n_disagreement = sum(
        conclusion["classical"] != conclusion["wcs"]
        for item in gold.values() if item["role"] == "none"
        for conclusion in item["conclusions"].values()
    )
    if n_disagreement != 1280:
        raise ValueError(f"Expected 1280 neutral disagreements, found {n_disagreement}")

    model_files = prediction_files()
    per_sample = []
    model_rows = {}
    for model, paths in model_files.items():
        rows = []
        for sample, path in enumerate(paths):
            validate_predictions(path, gold)
            metrics = score_seed(path, gold)
            if metrics["n_disagreement"] != n_disagreement:
                raise ValueError(f"{path}: incomplete disagreement subset")
            total = sum(metrics[key] for key in (*METRICS[:3], "unparsed_pct"))
            if not math.isclose(total, 100.0, abs_tol=1e-10):
                raise ValueError(f"{path}: outcome percentages sum to {total}")
            rows.append(metrics)
            per_sample.append({
                "model": model, "sample": sample,
                "prediction_file": str(path.relative_to(ROOT)), **metrics,
            })
        model_rows[model] = rows

    lines = [
        "% Generated by scripts/final2_disagreement_table.py",
        r"\begin{table}[H]",
        r"\caption{Model responses on the 1,280 question instances in the full all-forms dataset where Classical Logic and Weak Completion Semantics (WCS) assign different gold labels in the neutral role condition. CoT entries report mean $\pm$ sample standard deviation across five sampling runs; FLAN-T5 entries are deterministic. All percentages use the full disagreement subset, including unparsed responses in the denominator. Neither counts only valid responses matching neither framework, so the three displayed rates may sum to less than 100\%. $\Delta$ denotes the WCS match rate minus the Classical match rate, in percentage points, with its standard deviation computed across per-run differences.}",
        r"\label{tab:disagreement-neutral-final2}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Classical (\%) & WCS (\%) & Neither (\%) & $\Delta$ \\",
        r"\midrule",
    ]
    for model, rows in model_rows.items():
        cells = []
        for key in METRICS:
            mean, std = mean_std([row[key] for row in rows])
            cell = f"{mean:+.2f}" if key == "delta" else f"{mean:.2f}"
            if len(rows) > 1:
                cell += f" $\\pm$ {std:.2f}"
            cells.append(cell)
        lines.append(model + " & " + " & ".join(cells) + r" \\")
        unparsed, _ = mean_std([row["unparsed_pct"] for row in rows])
        print(f"{model}: {' | '.join(cells)} | unparsed={unparsed:.4f}%")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validated {len(per_sample)} files; {n_disagreement} disagreements per run.")
    print(f"Wrote {TEX_PATH} and {CSV_PATH}")


if __name__ == "__main__":
    main()
