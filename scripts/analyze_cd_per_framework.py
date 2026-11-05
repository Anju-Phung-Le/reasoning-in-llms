#!/usr/bin/env python3
"""Evaluate model CD predictions separately against Classical and WCS gold labels."""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from analyse_cd_precision_recall import MODELS, NVC, ROLE

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data/processed/dataset_sub256.jsonl"
PREDICTION_ROOT = ROOT / "data/predictions"
REPORT_DIR = ROOT / "outputs/report"
TABLE_PATH = ROOT / "tables/cd_per_framework_revised.tex"
EXPECTED_OVERALL_CD = {
    "Mistral-7B": (16.66, 0.63),
    "Llama-3.1-8B": (29.71, 1.61),
    "OLMo-2-13B": (20.39, 1.10),
    "DeepSeek-R1-Distill-Llama-8B": (38.32, 1.20),
    "DeepSeek-R1-Distill-Qwen-14B": (66.48, 1.43),
    "Mistral-Small-24B": (68.67, 1.01),
}
LABELS = {0: "Yes", 1: "No", 2: "CD"}
INVALID = 3


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def load_gold():
    targets = {}
    for row in read_jsonl(GOLD_PATH):
        if row.get("role") != ROLE:
            continue
        for code, conclusion in row.get("conclusions", {}).items():
            key = (row["id"], code)
            if key in targets:
                raise ValueError(f"Duplicate neutral-role gold question: {key}")
            targets[key] = {
                "id": row["id"],
                "code": code,
                "role": row["role"],
                "form": row.get("form"),
                "domain": row.get("domain"),
                "premises": row.get("premises"),
                "question": conclusion.get("question"),
                "classical": conclusion.get("classical"),
                "wcs": conclusion.get("wcs"),
            }
    if len(targets) != 512:
        raise ValueError(f"Expected 512 neutral-role questions, found {len(targets)}")
    if any(target["classical"] not in LABELS or target["wcs"] not in LABELS for target in targets.values()):
        raise ValueError("Invalid Classical/WCS gold label found")
    return targets


def prediction_paths(model_config):
    return [PREDICTION_ROOT / relative_path for relative_path in model_config["files"]]


def load_prediction_run(path, targets):
    occurrences = Counter()
    predictions = {}
    mismatches = []
    for row in read_jsonl(path):
        for code, prediction in row.get("conclusions", {}).items():
            key = (row.get("id"), code)
            target = targets.get(key)
            if target is None:
                continue
            occurrences[key] += 1
            predictions[key] = prediction
            for field in ("role", "form", "domain", "premises"):
                if row.get(field) != target[field]:
                    mismatches.append((key, field, target[field], row.get(field)))
            for field in ("question", "classical", "wcs"):
                if prediction.get(field) != target[field]:
                    mismatches.append((key, field, target[field], prediction.get(field)))
    missing = set(targets) - set(occurrences)
    duplicates = {key: count for key, count in occurrences.items() if count > 1}
    report = {
        "file": path.name,
        "matched": len(set(occurrences) & set(targets)),
        "missing": len(missing),
        "duplicate_keys": len(duplicates),
        "duplicate_extra_rows": sum(count - 1 for count in duplicates.values()),
        "mismatches": mismatches,
        "missing_keys": missing,
    }
    return predictions, report


def prediction_label(prediction):
    value = prediction.get("pred_index")
    return value if type(value) is int and value in LABELS else INVALID


def score_cd(targets, predictions, framework):
    counts = Counter()
    for key, target in targets.items():
        gold_positive = target[framework] == NVC
        predicted = prediction_label(predictions[key])
        if predicted == INVALID:
            counts["invalid_total"] += 1
            counts["invalid_on_cd_gold" if gold_positive else "invalid_on_non_cd_gold"] += 1
        if gold_positive:
            if predicted == NVC:
                counts["tp"] += 1
            else:
                counts["fn"] += 1
        else:
            if predicted == NVC:
                counts["fp"] += 1
            elif predicted in (0, 1):
                counts["tn"] += 1

    positive_gold = sum(target[framework] == NVC for target in targets.values())
    negative_gold = len(targets) - positive_gold
    predicted_cd = counts["tp"] + counts["fp"]
    precision_denominator = predicted_cd
    recall_denominator = counts["tp"] + counts["fn"]
    f1_denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]

    precision = counts["tp"] / precision_denominator if precision_denominator else None
    recall = counts["tp"] / recall_denominator if recall_denominator else None
    f1 = (
        2 * counts["tp"] / f1_denominator
        if f1_denominator else None
    )
    counts.update({
        "questions": len(targets),
        "gold_cd_denominator": positive_gold,
        "gold_non_cd_denominator": negative_gold,
        "predicted_cd_denominator": predicted_cd,
        "precision_denominator": precision_denominator,
        "recall_denominator": recall_denominator,
        "f1_denominator": f1_denominator,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    return counts


def wcs_yes_no_rates(targets, predictions):
    yes_total = sum(target["wcs"] == 0 for target in targets.values())
    no_total = sum(target["wcs"] == 1 for target in targets.values())
    yes_correct = sum(
        prediction_label(predictions[key]) == 0
        for key, target in targets.items() if target["wcs"] == 0
    )
    no_correct = sum(
        prediction_label(predictions[key]) == 1
        for key, target in targets.items() if target["wcs"] == 1
    )
    return {
        "wcs_yes_gold_denominator": yes_total,
        "wcs_yes_correct": yes_correct,
        "wcs_yes_accuracy": yes_correct / yes_total if yes_total else None,
        "wcs_no_gold_denominator": no_total,
        "wcs_no_correct": no_correct,
        "wcs_no_accuracy": no_correct / no_total if no_total else None,
    }


def mean_sd(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None, None
    return statistics.fmean(valid), statistics.stdev(valid) if len(valid) > 1 else 0.0


def validate_table44(model_results):
    errors = []
    for model, runs in model_results.items():
        overall_rates = [run["overall_cd_rate"] for run in runs]
        mean = statistics.fmean(overall_rates)
        sd = statistics.stdev(overall_rates) if len(overall_rates) > 1 else 0.0
        expected_mean, expected_sd = EXPECTED_OVERALL_CD[model]
        if round(mean, 2) != round(expected_mean, 2) or round(sd, 2) != round(expected_sd, 2):
            errors.append(
                f"Table 4.4 CD rate mismatch for {model}: {mean:.2f} +/- {sd:.2f}; "
                f"expected {expected_mean:.2f} +/- {expected_sd:.2f}"
            )
    return errors


def analyze():
    targets = load_gold()
    target_keys = set(targets)
    framework_gold_counts = {
        framework: Counter(target[framework] for target in targets.values())
        for framework in ("classical", "wcs")
    }
    model_runs = {}
    validation_errors = []
    validation_report = []

    for model, config in MODELS.items():
        paths = prediction_paths(config)
        if len(paths) != 10:
            validation_errors.append(f"{model}: expected 10 saved generations, found {len(paths)}")
        runs = []
        per_model_key_sets = []
        for run_index, path in enumerate(paths, start=1):
            if not path.exists():
                validation_errors.append(f"Missing prediction file: {path}")
                continue
            predictions, report = load_prediction_run(path, targets)
            per_model_key_sets.append(set(predictions))
            if (
                report["matched"] != 512
                or report["missing"]
                or report["duplicate_keys"]
                or report["mismatches"]
            ):
                validation_errors.append(
                    f"{model} {path.name}: matched={report['matched']}, missing={report['missing']}, "
                    f"duplicate_keys={report['duplicate_keys']}, mismatches={len(report['mismatches'])}"
                )
            predicted_labels = [prediction_label(predictions[key]) for key in target_keys if key in predictions]
            overall_cd_rate = 100.0 * sum(label == NVC for label in predicted_labels) / len(targets)
            framework_metrics = {
                framework: score_cd(targets, predictions, framework)
                for framework in ("classical", "wcs")
            }
            wcs_binary = wcs_yes_no_rates(targets, predictions)
            runs.append({
                "run_index": run_index,
                "path": path,
                "predictions": predictions,
                "overall_cd_rate": overall_cd_rate,
                "predicted_counts": Counter(predicted_labels),
                "framework_metrics": framework_metrics,
                "wcs_binary": wcs_binary,
            })
            validation_report.append({
                "model": model,
                "generation": run_index,
                "file": path.name,
                "matched": report["matched"],
                "missing": report["missing"],
                "duplicate_keys": report["duplicate_keys"],
                "duplicate_extra_rows": report["duplicate_extra_rows"],
                "gold_content_mismatches": len(report["mismatches"]),
            })
        if per_model_key_sets and any(keys != target_keys for keys in per_model_key_sets):
            validation_errors.append(f"{model}: saved generations do not share the exact 512-question key set")
        if len(runs) == 10:
            overall_mean = statistics.fmean(run["overall_cd_rate"] for run in runs)
            overall_sd = statistics.stdev(run["overall_cd_rate"] for run in runs)
            expected_mean, expected_sd = EXPECTED_OVERALL_CD[model]
            if round(overall_mean, 2) != round(expected_mean, 2) or round(overall_sd, 2) != round(expected_sd, 2):
                validation_errors.append(
                    f"{model}: overall CD rate does not reproduce Table 4.4; "
                    f"{overall_mean:.2f} +/- {overall_sd:.2f} vs {expected_mean:.2f} +/- {expected_sd:.2f}"
                )
        model_runs[model] = runs

    if len(target_keys) != 512:
        validation_errors.append(f"Expected 512 unique target pairs, found {len(target_keys)}")
    if framework_gold_counts["classical"].get(NVC, 0) == 0 or framework_gold_counts["wcs"].get(NVC, 0) == 0:
        validation_errors.append("One framework has no CD gold questions")

    if validation_errors:
        print("VALIDATION FAILED; no revised outputs written:")
        for error in validation_errors:
            print(f"- {error}")
        raise SystemExit(1)
    return targets, framework_gold_counts, model_runs, validation_report


def aggregate_model(model, runs, targets, framework_gold_counts):
    row = {
        "model": model,
        "generations": len(runs),
        "questions_per_generation": len(targets),
        "overall_cd_rate_denominator": len(targets),
        "overall_cd_rate_mean_pct": statistics.fmean(run["overall_cd_rate"] for run in runs),
        "overall_cd_rate_sd_pct": statistics.stdev(run["overall_cd_rate"] for run in runs),
    }
    for framework in ("classical", "wcs"):
        gold_cd_n = framework_gold_counts[framework][NVC]
        gold_non_cd_n = len(targets) - gold_cd_n
        metric_runs = [run["framework_metrics"][framework] for run in runs]
        for count_name in ("tp", "fp", "fn", "tn", "invalid_total", "invalid_on_cd_gold", "invalid_on_non_cd_gold"):
            mean, sd = mean_sd([float(metric[count_name]) for metric in metric_runs])
            row[f"{framework}_{count_name}_mean"] = mean
            row[f"{framework}_{count_name}_sd"] = sd
        row[f"{framework}_gold_cd_denominator"] = gold_cd_n
        row[f"{framework}_gold_non_cd_denominator"] = gold_non_cd_n
        row[f"{framework}_precision_denominator_mean"] = statistics.fmean(metric["precision_denominator"] for metric in metric_runs)
        row[f"{framework}_recall_denominator"] = gold_cd_n
        row[f"{framework}_f1_denominator_mean"] = statistics.fmean(metric["f1_denominator"] for metric in metric_runs)
        for metric_name in ("precision", "recall", "f1"):
            mean, sd = mean_sd([metric[metric_name] * 100 if metric[metric_name] is not None else None for metric in metric_runs])
            row[f"{framework}_{metric_name}_mean_pct"] = mean
            row[f"{framework}_{metric_name}_sd_pct"] = sd
    for key in (
        "wcs_yes_gold_denominator", "wcs_yes_correct",
        "wcs_no_gold_denominator", "wcs_no_correct",
    ):
        mean, sd = mean_sd([float(run["wcs_binary"][key]) for run in runs])
        row[f"{key}_mean"] = mean
        row[f"{key}_sd"] = sd
    for key in ("wcs_yes_accuracy", "wcs_no_accuracy"):
        mean, sd = mean_sd([run["wcs_binary"][key] * 100 if run["wcs_binary"][key] is not None else None for run in runs])
        row[f"{key}_mean_pct"] = mean
        row[f"{key}_sd_pct"] = sd
    return row


def csv_rows(targets, framework_gold_counts, model_runs):
    rows = []
    for model, runs in model_runs.items():
        for run in runs:
            for framework in ("classical", "wcs"):
                metrics = run["framework_metrics"][framework]
                rows.append({
                    "row_type": "generation",
                    "model": model,
                    "framework": framework.title(),
                    "generation": run["run_index"],
                    "file": run["path"].name,
                    "questions_total": len(targets),
                    "gold_cd_denominator": framework_gold_counts[framework][NVC],
                    "gold_non_cd_denominator": len(targets) - framework_gold_counts[framework][NVC],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                    "invalid_total": metrics["invalid_total"],
                    "invalid_on_cd_gold": metrics["invalid_on_cd_gold"],
                    "invalid_on_non_cd_gold": metrics["invalid_on_non_cd_gold"],
                    "precision_denominator": metrics["precision_denominator"],
                    "recall_denominator": metrics["recall_denominator"],
                    "f1_denominator": metrics["f1_denominator"],
                    "precision_pct": metrics["precision"] * 100 if metrics["precision"] is not None else "NA",
                    "recall_pct": metrics["recall"] * 100 if metrics["recall"] is not None else "NA",
                    "f1_pct": metrics["f1"] * 100 if metrics["f1"] is not None else "NA",
                    "overall_cd_predictions": sum(label == NVC for label in run["predicted_counts"].elements()),
                    "overall_cd_rate_denominator": len(targets),
                    "overall_cd_rate_pct": run["overall_cd_rate"],
                    "wcs_yes_gold_denominator": run["wcs_binary"]["wcs_yes_gold_denominator"],
                    "wcs_yes_correct": run["wcs_binary"]["wcs_yes_correct"],
                    "wcs_yes_accuracy_pct": run["wcs_binary"]["wcs_yes_accuracy"] * 100,
                    "wcs_no_gold_denominator": run["wcs_binary"]["wcs_no_gold_denominator"],
                    "wcs_no_correct": run["wcs_binary"]["wcs_no_correct"],
                    "wcs_no_accuracy_pct": run["wcs_binary"]["wcs_no_accuracy"] * 100,
                })
        summary = aggregate_model(model, runs, targets, framework_gold_counts)
        for framework in ("Classical", "WCS"):
            row = {"row_type": "mean_sd_summary", "framework": framework, **summary}
            rows.append(row)
    return rows


def fmt_metric(runs, framework, metric):
    values = [run["framework_metrics"][framework][metric] for run in runs]
    if any(value is None for value in values):
        return "N/A"
    mean = statistics.fmean(values) * 100
    sd = statistics.stdev(values) * 100 if len(values) > 1 else 0.0
    return f"{mean:.2f} $\\pm$ {sd:.2f}\\%"


def fmt_overall_cd(runs):
    values = [run["overall_cd_rate"] for run in runs]
    return f"{statistics.fmean(values):.2f} $\\pm$ {statistics.stdev(values):.2f}\\%"


def write_latex(targets, framework_gold_counts, model_runs):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Cannot-Determine (CD) performance against each framework's gold labels on 512 neutral-role questions. Values are mean $\pm$ standard deviation over ten saved generations.}",
        r"\label{tab:cd-per-framework-revised}",
        r"\begin{tabular}{p{3.5cm}rrrr}",
        r"\toprule",
        r"\multicolumn{5}{c}{\textbf{Panel A: Classical Logic}} \\",
        r"\midrule",
        r"Model & Overall CD (\%) & Precision & Recall & F1 \\",
        r"\midrule",
    ]
    for model, runs in model_runs.items():
        precision = fmt_metric(runs, "classical", "precision")
        recall = fmt_metric(runs, "classical", "recall")
        f1 = fmt_metric(runs, "classical", "f1")
        lines.append(f"{model} & {fmt_overall_cd(runs)} & {precision} & {recall} & {f1} \\\\")
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{c}{\textbf{Panel B: WCS}} \\",
        r"\midrule",
        r"Model & Overall CD (\%) & Precision & Recall & F1 \\",
        r"\midrule",
    ])
    for model, runs in model_runs.items():
        precision = fmt_metric(runs, "wcs", "precision")
        recall = fmt_metric(runs, "wcs", "recall")
        f1 = fmt_metric(runs, "wcs", "f1")
        lines.append(f"{model} & {fmt_overall_cd(runs)} & {precision} & {recall} & {f1} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}",
        r"\scriptsize Overall CD rates use all 512 outputs. CD-gold denominators are $n=454$ for Classical and $n=433$ for WCS. Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). Invalid outputs on CD-gold questions are FN; invalid outputs on non-CD-gold questions are reported separately and excluded from TN. The CSV reports WCS Yes/No accuracy, confusion counts, and invalid outputs. Table 4.4 overall CD rates were reproduced as a validation check.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    targets, framework_gold_counts, model_runs, validation = analyze()
    report_file = REPORT_DIR / "cd_per_framework_revised.csv"
    rows = csv_rows(targets, framework_gold_counts, model_runs)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8", newline="") as destination:
        fieldnames = list(dict.fromkeys(field for row in rows for field in row))
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_latex(targets, framework_gold_counts, model_runs)

    print(f"Validated {len(targets)} common neutral-role questions for all 6 models × 10 generations.")
    print("Gold CD denominators:", {name.title(): counts[NVC] for name, counts in framework_gold_counts.items()})
    print("Question matching, duplicates, gold labels/content, and ten-generation counts: PASS")
    print("Overall Table 4.4 CD rate reproduction: PASS")
    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {report_file}")
    print("Invalid outputs are separate; an invalid response on a CD-gold question is included in FN, while an invalid response on non-CD gold is excluded from TN.")


if __name__ == "__main__":
    main()
