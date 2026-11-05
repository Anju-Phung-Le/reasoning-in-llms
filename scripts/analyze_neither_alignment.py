#!/usr/bin/env python3
"""Analyze label-type contributions and direct contradictions on neutral disagreements."""

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data/processed/dataset_sub256.jsonl"
PREDICTION_ROOT = ROOT / "data/predictions"
TABLES_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "outputs/report"

LABELS = {0: "Yes", 1: "No", 2: "CD"}
CATEGORY_INFO = {
    "A": {"classical": 2, "wcs": 0, "description": "Classical CD; WCS Yes"},
    "B": {"classical": 2, "wcs": 1, "description": "Classical CD; WCS No"},
    "C": {"classical": 0, "wcs": 2, "description": "Classical Yes; WCS CD"},
    "D": {"classical": 1, "wcs": 2, "description": "Classical No; WCS CD"},
}
OUTCOMES = ("classical_only", "wcs_only", "neither", "invalid")
DECOMPOSITIONS = (
    "classical_cd",
    "classical_definite",
    "wcs_cd",
    "wcs_definite",
)

MODEL_SPECS = [
    {
        "name": "Flan-T5 Small",
        "short": "Flan-T5 Small",
        "setting": "deterministic FLAN-T5",
        "paths": [PREDICTION_ROOT / "all_forms/flan_t5_small4_all_forms_predictions.jsonl"],
    },
    {
        "name": "Flan-T5 Base",
        "short": "Flan-T5 Base",
        "setting": "deterministic FLAN-T5",
        "paths": [PREDICTION_ROOT / "all_forms/flan_t5_base3_all_forms_predictions.jsonl"],
    },
    {
        "name": "Flan-T5 Large",
        "short": "Flan-T5 Large",
        "setting": "deterministic FLAN-T5",
        "paths": [PREDICTION_ROOT / "all_forms/flan_t5_large3_all_forms_predictions.jsonl"],
    },
    {
        "name": "Mistral-7B",
        "short": "Mistral-7B",
        "setting": "sampled vLLM CoT",
        "directory": "mistral7b_vllm",
    },
    {
        "name": "Llama-3.1-8B",
        "short": "Llama-3.1-8B",
        "setting": "sampled vLLM CoT",
        "directory": "llama8b_vllm",
    },
    {
        "name": "OLMo-2-13B",
        "short": "OLMo-2-13B",
        "setting": "sampled vLLM CoT",
        "directory": "olmo2_13b_vllm",
    },
    {
        "name": "DeepSeek-R1-Distill-Llama-8B",
        "short": "DeepSeek-Llama-8B",
        "setting": "sampled vLLM CoT",
        "directory": "deepseek_llama_8b_vllm",
    },
    {
        "name": "DeepSeek-R1-Distill-Qwen-14B",
        "short": "DeepSeek-Qwen-14B",
        "setting": "sampled vLLM CoT",
        "directory": "deepseek_qwen_14b_vllm",
    },
    {
        "name": "Mistral-Small-24B",
        "short": "Mistral-Small-24B",
        "setting": "sampled vLLM CoT",
        "directory": "mistral24B_vllm",
    },
]

# Table 4.3 values are used as a regression check, not as analysis inputs.
TABLE_43_EXPECTED = {
    "Flan-T5 Small": {"classical_only": 17.39, "wcs_only": 34.78, "neither": 47.83, "delta": 17.39},
    "Flan-T5 Base": {"classical_only": 24.64, "wcs_only": 52.17, "neither": 23.19, "delta": 27.54},
    "Flan-T5 Large": {"classical_only": 30.43, "wcs_only": 46.38, "neither": 23.19, "delta": 15.94},
    "Mistral-7B": {
        "classical_only": (36.09, 2.77), "wcs_only": (45.80, 2.39),
        "neither": (17.54, 2.51), "delta": (9.71, 4.48),
    },
    "Llama-3.1-8B": {
        "classical_only": (39.86, 3.82), "wcs_only": (51.88, 3.79),
        "neither": (6.52, 1.71), "delta": (12.03, 7.13),
    },
    "OLMo-2-13B": {
        "classical_only": (36.52, 1.91), "wcs_only": (43.62, 3.01),
        "neither": (19.86, 2.74), "delta": (7.10, 4.24),
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "classical_only": (53.33, 3.60), "wcs_only": (40.87, 2.80),
        "neither": (5.80, 1.81), "delta": (-12.46, 6.19),
    },
    "DeepSeek-R1-Distill-Qwen-14B": {
        "classical_only": (64.49, 4.17), "wcs_only": (33.77, 3.99),
        "neither": (1.74, 1.33), "delta": (-30.72, 8.05),
    },
    "Mistral-Small-24B": {
        "classical_only": (56.81, 1.91), "wcs_only": (42.32, 2.25),
        "neither": (0.87, 0.75), "delta": (-14.49, 4.10),
    },
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def category_for(classical, wcs):
    for category, labels in CATEGORY_INFO.items():
        if classical == labels["classical"] and wcs == labels["wcs"]:
            return category
    raise ValueError(f"Unexpected disagreement labels: Classical={classical}, WCS={wcs}")


def load_targets():
    targets = {}
    occurrences = Counter()
    for row in read_jsonl(GOLD_PATH):
        if row.get("role") != "none":
            continue
        for code, conclusion in row.get("conclusions", {}).items():
            classical = conclusion.get("classical")
            wcs = conclusion.get("wcs")
            if classical == wcs:
                continue
            key = (row["id"], code)
            occurrences[key] += 1
            targets[key] = {
                "id": row["id"],
                "code": code,
                "form": row.get("form"),
                "domain": row.get("domain"),
                "role": row.get("role"),
                "premises": row.get("premises"),
                "question": conclusion.get("question"),
                "classical": classical,
                "wcs": wcs,
                "category": category_for(classical, wcs),
            }
    duplicate_keys = [key for key, count in occurrences.items() if count > 1]
    if duplicate_keys:
        raise ValueError(f"Duplicate target keys in gold data: {duplicate_keys[:5]}")
    if len(targets) != 69:
        raise ValueError(f"Expected 69 neutral-role disagreements, found {len(targets)}")
    category_counts = Counter(target["category"] for target in targets.values())
    if sum(category_counts.values()) != 69 or set(category_counts) != set(CATEGORY_INFO):
        raise ValueError(f"Disagreement category coverage is invalid: {dict(category_counts)}")
    return targets, category_counts


def prediction_paths(spec):
    if "paths" in spec:
        return spec["paths"]
    directory = PREDICTION_ROOT / "vllm" / spec["directory"]
    return sorted(
        path
        for path in directory.glob("*sample*.jsonl")
        if "_conversations" not in path.name
    )


def load_prediction_run(path, targets):
    occurrences = Counter()
    predictions = {}
    mismatches = []
    for row in read_jsonl(path):
        conclusions = row.get("conclusions")
        if not isinstance(conclusions, dict):
            continue
        for code, prediction in conclusions.items():
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
        "duplicates": len(duplicates),
        "duplicate_extra_rows": sum(count - 1 for count in duplicates.values()),
        "mismatches": mismatches,
        "missing_keys": missing,
    }
    return predictions, report


def initialize_category_counts():
    return {
        category: {
            "outcomes": Counter(),
            "contradicts_classical": 0,
            "contradicts_wcs": 0,
        }
        for category in CATEGORY_INFO
    }


def analyze_run(targets, predictions):
    result = {
        "outcomes": Counter(),
        "decomposition": Counter(),
        "contradicts_classical": 0,
        "contradicts_wcs": 0,
        "categories": initialize_category_counts(),
    }
    for key, target in targets.items():
        prediction = predictions[key]
        pred_index = prediction.get("pred_index")
        if type(pred_index) is not int or pred_index not in LABELS:
            outcome = "invalid"
        elif pred_index == target["classical"]:
            outcome = "classical_only"
        elif pred_index == target["wcs"]:
            outcome = "wcs_only"
        else:
            outcome = "neither"

        result["outcomes"][outcome] += 1
        category_result = result["categories"][target["category"]]
        category_result["outcomes"][outcome] += 1

        if outcome == "invalid":
            continue
        if target["classical"] == 2 and pred_index == 2:
            result["decomposition"]["classical_cd"] += 1
        if target["classical"] in (0, 1) and pred_index == target["classical"]:
            result["decomposition"]["classical_definite"] += 1
        if target["wcs"] == 2 and pred_index == 2:
            result["decomposition"]["wcs_cd"] += 1
        if target["wcs"] in (0, 1) and pred_index == target["wcs"]:
            result["decomposition"]["wcs_definite"] += 1

        if outcome == "neither":
            if target["classical"] in (0, 1) and pred_index == 1 - target["classical"]:
                result["contradicts_classical"] += 1
                category_result["contradicts_classical"] += 1
            if target["wcs"] in (0, 1) and pred_index == 1 - target["wcs"]:
                result["contradicts_wcs"] += 1
                category_result["contradicts_wcs"] += 1

    if sum(result["outcomes"].values()) != len(targets):
        raise ValueError("Classified outcome counts do not sum to 69")
    if result["contradicts_classical"] + result["contradicts_wcs"] != result["outcomes"]["neither"]:
        raise ValueError("Each valid Neither response should contradict exactly one definite gold label")
    return result


def denominator_counts(targets):
    return {
        "classical_cd": sum(target["classical"] == 2 for target in targets.values()),
        "classical_definite": sum(target["classical"] in (0, 1) for target in targets.values()),
        "wcs_cd": sum(target["wcs"] == 2 for target in targets.values()),
        "wcs_definite": sum(target["wcs"] in (0, 1) for target in targets.values()),
    }


def metric_values(run, targets, denominators):
    total = len(targets)
    values = {}
    for outcome in OUTCOMES:
        values[f"{outcome}_count"] = run["outcomes"][outcome]
        values[f"{outcome}_pct"] = 100.0 * run["outcomes"][outcome] / total
    values["delta_pp"] = values["wcs_only_pct"] - values["classical_only_pct"]
    for name in DECOMPOSITIONS:
        count = run["decomposition"][name]
        denominator = denominators[name]
        values[f"{name}_count"] = count
        values[f"{name}_pct_all69"] = 100.0 * count / total
        values[f"{name}_conditional_pct"] = 100.0 * count / denominator if denominator else 0.0
    values["contradicts_classical_count"] = run["contradicts_classical"]
    values["contradicts_wcs_count"] = run["contradicts_wcs"]
    values["contradicts_classical_pct_all69"] = 100.0 * run["contradicts_classical"] / total
    values["contradicts_wcs_pct_all69"] = 100.0 * run["contradicts_wcs"] / total
    neither = run["outcomes"]["neither"]
    values["contradicts_classical_pct_neither"] = (
        100.0 * run["contradicts_classical"] / neither if neither else None
    )
    values["contradicts_wcs_pct_neither"] = (
        100.0 * run["contradicts_wcs"] / neither if neither else None
    )
    values["contradicts_classical_conditional_pct"] = (
        100.0 * run["contradicts_classical"] / denominators["classical_definite"]
        if denominators["classical_definite"] else 0.0
    )
    values["contradicts_wcs_conditional_pct"] = (
        100.0 * run["contradicts_wcs"] / denominators["wcs_definite"]
        if denominators["wcs_definite"] else 0.0
    )
    return values


def mean_sd(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_metric(runs, key):
    return mean_sd([run[key] for run in runs])


def validate_table43(model_results):
    errors = []
    checked = {}
    for spec in MODEL_SPECS:
        name = spec["name"]
        runs = model_results[name]
        checked[name] = {}
        for metric in ("classical_only_pct", "wcs_only_pct", "neither_pct", "delta_pp"):
            mean, sd = aggregate_metric([metric_values(run, model_results["targets"], model_results["denominators"]) for run in runs], metric)
            expected_key = {
                "classical_only_pct": "classical_only",
                "wcs_only_pct": "wcs_only",
                "neither_pct": "neither",
                "delta_pp": "delta",
            }[metric]
            expected = TABLE_43_EXPECTED[name][expected_key]
            expected_mean = expected[0] if isinstance(expected, tuple) else expected
            expected_sd = expected[1] if isinstance(expected, tuple) else None
            passed = round(mean, 2) == round(expected_mean, 2)
            if expected_sd is not None:
                passed = passed and round(sd, 2) == round(expected_sd, 2)
            checked[name][metric] = {"mean": mean, "sd": sd, "expected": expected, "passed": passed}
            if not passed:
                errors.append(f"Table 4.3 mismatch for {name} {metric}: observed {mean:.2f} +/- {sd:.2f}, expected {expected}")
    return checked, errors


def load_all_models(targets, denominators):
    model_results = {"targets": targets, "denominators": denominators}
    validation_errors = []
    validation_reports = {}
    for spec in MODEL_SPECS:
        paths = prediction_paths(spec)
        if not paths:
            validation_errors.append(f"No prediction files found for {spec['name']}")
            continue
        if spec["setting"] == "sampled vLLM CoT" and len(paths) != 10:
            validation_errors.append(f"Expected 10 vLLM sample files for {spec['name']}, found {len(paths)}")
        results = []
        reports = []
        for path in paths:
            if not path.exists():
                validation_errors.append(f"Missing prediction file: {path}")
                continue
            predictions, report = load_prediction_run(path, targets)
            reports.append(report)
            if report["matched"] != 69 or report["missing"] or report["duplicates"] or report["mismatches"]:
                validation_errors.append(
                    f"{spec['name']} {path.name}: matched={report['matched']}, "
                    f"missing={report['missing']}, duplicate_keys={report['duplicates']}, "
                    f"gold/content_mismatches={len(report['mismatches'])}"
                )
            else:
                results.append(analyze_run(targets, predictions))
        validation_reports[spec["name"]] = reports
        if results:
            model_results[spec["name"]] = results
    if len(model_results) != len(MODEL_SPECS) + 2:
        validation_errors.append("One or more models did not produce analyzable prediction runs")
    return model_results, validation_reports, validation_errors


def validate_category_sample(category_counts):
    return sum(category_counts.values()) == 69 and set(category_counts) == set(CATEGORY_INFO)


def safe_value(value, digits=4):
    return "" if value is None else f"{value:.{digits}f}"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_alignment_csv(model_results, targets, denominators):
    rows = []
    category_counts = Counter(target["category"] for target in targets.values())
    for spec in MODEL_SPECS:
        name = spec["name"]
        runs = model_results[name]
        row = {
            "model": name,
            "inference_setting": spec["setting"],
            "prediction_runs": len(runs),
            "questions_per_run": len(targets),
        }
        for category in CATEGORY_INFO:
            row[f"category_{category}_question_denominator"] = category_counts[category]
        for outcome in OUTCOMES:
            for metric_suffix in ("count", "pct"):
                mean, sd = aggregate_metric(
                    [metric_values(run, targets, denominators) for run in runs],
                    f"{outcome}_{metric_suffix}",
                )
                row[f"{outcome}_{metric_suffix}_mean"] = safe_value(mean)
                row[f"{outcome}_{metric_suffix}_sd"] = safe_value(sd)
        delta_mean, delta_sd = aggregate_metric(
            [metric_values(run, targets, denominators) for run in runs], "delta_pp"
        )
        row["delta_pp_mean"] = safe_value(delta_mean)
        row["delta_pp_sd"] = safe_value(delta_sd)
        for name_key in DECOMPOSITIONS:
            row[f"{name_key}_question_denominator"] = denominators[name_key]
            for suffix in ("count", "pct_all69", "conditional_pct"):
                mean, sd = aggregate_metric(
                    [metric_values(run, targets, denominators) for run in runs],
                    f"{name_key}_{suffix}",
                )
                row[f"{name_key}_{suffix}_mean"] = safe_value(mean)
                row[f"{name_key}_{suffix}_sd"] = safe_value(sd)
        rows.append(row)
    return rows


def build_neither_csv(model_results, targets, denominators):
    rows = []
    total = len(targets)
    for spec in MODEL_SPECS:
        model = spec["name"]
        runs = model_results[model]
        category_order = ["ALL", *CATEGORY_INFO]
        for category in category_order:
            category_targets = (
                [target for target in targets.values() if target["category"] == category]
                if category != "ALL" else list(targets.values())
            )
            category_denominator = len(category_targets)
            c_definite_in_category = sum(target["classical"] in (0, 1) for target in category_targets)
            w_definite_in_category = sum(target["wcs"] in (0, 1) for target in category_targets)
            c_denom = denominators["classical_definite"] if category == "ALL" else c_definite_in_category
            w_denom = denominators["wcs_definite"] if category == "ALL" else w_definite_in_category
            metric_samples = defaultdict(list)
            for run in runs:
                outcomes = run["outcomes"] if category == "ALL" else run["categories"][category]["outcomes"]
                neither = outcomes["neither"]
                contra_classical = run["contradicts_classical"] if category == "ALL" else run["categories"][category]["contradicts_classical"]
                contra_wcs = run["contradicts_wcs"] if category == "ALL" else run["categories"][category]["contradicts_wcs"]
                for outcome in OUTCOMES:
                    metric_samples[f"{outcome}_count"].append(outcomes[outcome])
                    metric_samples[f"{outcome}_pct_category"].append(
                        100.0 * outcomes[outcome] / category_denominator if category_denominator else None
                    )
                    metric_samples[f"{outcome}_pct_all69"].append(100.0 * outcomes[outcome] / total)
                metric_samples["contradicts_classical_count"].append(contra_classical)
                metric_samples["contradicts_wcs_count"].append(contra_wcs)
                metric_samples["contradicts_classical_pct_all69"].append(100.0 * contra_classical / total)
                metric_samples["contradicts_wcs_pct_all69"].append(100.0 * contra_wcs / total)
                metric_samples["contradicts_classical_pct_neither"].append(100.0 * contra_classical / neither if neither else None)
                metric_samples["contradicts_wcs_pct_neither"].append(100.0 * contra_wcs / neither if neither else None)
                metric_samples["contradicts_classical_conditional_pct"].append(100.0 * contra_classical / c_denom if c_denom else None)
                metric_samples["contradicts_wcs_conditional_pct"].append(100.0 * contra_wcs / w_denom if w_denom else None)

            row = {
                "model": model,
                "inference_setting": spec["setting"],
                "category": category,
                "category_description": "All categories" if category == "ALL" else CATEGORY_INFO[category]["description"],
                "category_question_denominator": category_denominator,
                "all_questions_denominator": total,
                "classical_definite_question_denominator": denominators["classical_definite"],
                "wcs_definite_question_denominator": denominators["wcs_definite"],
                "conditional_classical_definite_denominator": c_denom,
                "conditional_wcs_definite_denominator": w_denom,
                "prediction_runs": len(runs),
            }
            for metric_name, samples in metric_samples.items():
                mean, sd = mean_sd(samples)
                row[f"{metric_name}_mean"] = safe_value(mean)
                row[f"{metric_name}_sd"] = safe_value(sd)
            rows.append(row)
    return rows


def write_categories_csv(path, targets):
    counts = Counter(target["category"] for target in targets.values())
    rows = []
    for category, info in CATEGORY_INFO.items():
        rows.append({
            "category": category,
            "description": info["description"],
            "classical_gold": LABELS[info["classical"]],
            "wcs_gold": LABELS[info["wcs"]],
            "question_count": counts[category],
            "total_questions_denominator": len(targets),
            "percent_of_69": f"{100.0 * counts[category] / len(targets):.4f}",
        })
    write_csv(path, rows, list(rows[0]))


def percentage_cell(runs, targets, denominators, metric, latex=True):
    mean, sd = aggregate_metric(
        [metric_values(run, targets, denominators) for run in runs], metric
    )
    if len(runs) == 1:
        return f"{mean:.2f}\\%" if latex else f"{mean:.2f}%"
    return f"{mean:.2f} $\\pm$ {sd:.2f}\\%" if latex else f"{mean:.2f} +/- {sd:.2f}%"


def write_alignment_latex(path, model_results, targets, denominators):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Decomposition of Classical and WCS alignment on neutral-role disagreement questions. Entries are percentages of all 69 questions; vLLM entries report mean $\pm$ standard deviation over ten generations, while FLAN-T5 entries are deterministic.}",
        r"\label{tab:alignment-decomposition}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Classical CD & Classical Yes/No & WCS CD & WCS Yes/No \\",
        r"\midrule",
    ]
    for spec in MODEL_SPECS:
        runs = model_results[spec["name"]]
        cells = [spec["short"]]
        for metric in DECOMPOSITIONS:
            cells.append(percentage_cell(runs, targets, denominators, f"{metric}_pct_all69"))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}",
        r"\scriptsize CD means Cannot Determine. Classical CD/Yes-No columns divide matching responses by 69; the conditional rates by the applicable framework label counts are reported in the CSV. These are response-alignment rates, not evidence that a model internally implements either semantics.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_neither_latex(path, model_results, targets, denominators):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Neither response rates and conditional direct-contradiction rates on neutral-role disagreement questions.}",
        r"\label{tab:neither-direct-contradictions}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Neither / 69 (\%) & Classical definite / 24 (\%) & WCS definite / 45 (\%) \\",
        r"\midrule",
    ]
    for spec in MODEL_SPECS:
        runs = model_results[spec["name"]]
        cells = [spec["short"]]
        for metric in (
            "neither_pct",
            "contradicts_classical_conditional_pct",
            "contradicts_wcs_conditional_pct",
        ):
            cells.append(percentage_cell(runs, targets, denominators, metric))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}",
        r"\scriptsize Neither rate uses all 69 questions. The Classical and WCS columns are conditional rates: direct contradictions divided by questions where that framework gives a definite Yes/No label ($n=24$ for Classical; $n=45$ for WCS). CD is not a direct contradiction. vLLM entries are mean $\pm$ SD over ten generations; FLAN-T5 entries are deterministic. Category-specific counts are in the CSV report.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path, targets, category_counts, model_results, validation_reports, table43_checks, denominators):
    lines = [
        "NEITHER ALIGNMENT ANALYSIS",
        "==========================",
        f"Target set: {len(targets)} neutral-role (id, conclusion_code) questions from {GOLD_PATH.relative_to(ROOT)}.",
        "Category counts: " + ", ".join(f"{key}={category_counts[key]}" for key in CATEGORY_INFO),
        f"Category total: {sum(category_counts.values())}/69.",
        "Classical definite denominator: " + str(denominators["classical_definite"]),
        "WCS definite denominator: " + str(denominators["wcs_definite"]),
        "",
        "VALIDATION",
        "----------",
    ]
    for spec in MODEL_SPECS:
        reports = validation_reports[spec["name"]]
        lines.append(
            f"{spec['name']}: {len(reports)} run(s); each matched 69, missing 0, duplicate keys 0, "
            f"duplicate extra rows 0, gold/content mismatches 0."
        )
    lines.append("All classified outcomes sum to 69 per run: Classical-only + WCS-only + Neither + invalid.")
    lines.append("Recomputed Table 4.3 means/SDs: " + ("PASS" if all(item["passed"] for values in table43_checks.values() for item in values.values()) else "FAIL"))
    lines.extend(["", "TABLE 4.3 REPRODUCTION", "----------------------"])
    for spec in MODEL_SPECS:
        checks = table43_checks[spec["name"]]
        entries = []
        for metric in ("classical_only_pct", "wcs_only_pct", "neither_pct", "delta_pp"):
            check = checks[metric]
            entry = f"{metric}={check['mean']:.2f}"
            if check["sd"]:
                entry += f" +/- {check['sd']:.2f}"
            entries.append(entry)
        invalid_mean, invalid_sd = aggregate_metric(
            [metric_values(run, targets, denominators) for run in model_results[spec["name"]]],
            "invalid_pct",
        )
        entries.append(f"invalid={invalid_mean:.2f}%" + (f" +/- {invalid_sd:.2f}" if invalid_sd else ""))
        lines.append(f"{spec['name']}: " + "; ".join(entries))

    lines.extend(["", "INTERPRETATION", "---------------"])
    model_summaries = []
    for spec in MODEL_SPECS:
        runs = model_results[spec["name"]]
        metric_runs = [metric_values(run, targets, denominators) for run in runs]
        delta = statistics.mean(run["delta_pp"] for run in metric_runs)
        delta_cd = statistics.mean(
            run["wcs_cd_pct_all69"] - run["classical_cd_pct_all69"] for run in metric_runs
        )
        delta_definite = statistics.mean(
            run["wcs_definite_pct_all69"] - run["classical_definite_pct_all69"] for run in metric_runs
        )
        direction = "WCS-leaning" if delta > 0 else "Classical-leaning" if delta < 0 else "balanced"
        driver = "CD matches" if (delta_cd >= delta_definite if delta > 0 else delta_cd <= delta_definite) else "definite Yes/No matches"
        c_def_contra = statistics.mean(run["contradicts_classical_conditional_pct"] for run in metric_runs)
        w_def_contra = statistics.mean(run["contradicts_wcs_conditional_pct"] for run in metric_runs)
        model_summaries.append({
            "name": spec["name"], "direction": direction, "delta": delta,
            "delta_cd": delta_cd, "delta_definite": delta_definite, "driver": driver,
            "c_cond": c_def_contra, "w_cond": w_def_contra,
            "c_contra_all": statistics.mean(run["contradicts_classical_pct_all69"] for run in metric_runs),
            "w_contra_all": statistics.mean(run["contradicts_wcs_pct_all69"] for run in metric_runs),
        })
        lines.append(
            f"{spec['name']}: {direction} (Table 4.3 Δ={delta:+.2f} pp); "
            f"CD component Δ={delta_cd:+.2f} pp, definite-answer component Δ={delta_definite:+.2f} pp; "
            f"larger directional contribution: {driver}."
        )

    mean_c_contra = statistics.mean(item["c_contra_all"] for item in model_summaries)
    mean_w_contra = statistics.mean(item["w_contra_all"] for item in model_summaries)
    mean_c_cond = statistics.mean(item["c_cond"] for item in model_summaries)
    mean_w_cond = statistics.mean(item["w_cond"] for item in model_summaries)
    lines.append(
        f"Equal-model average direct contradiction rates: Classical {mean_c_contra:.2f}% of all questions; "
        f"WCS {mean_w_contra:.2f}% of all questions. Conditional on the respective framework's definite-label questions: "
        f"Classical {mean_c_cond:.2f}%, WCS {mean_w_cond:.2f}%."
    )
    leaning = [item for item in model_summaries if item["delta"] > 0]
    classical_leaning = [item for item in model_summaries if item["delta"] < 0]
    lines.append(
        f"Overall direction: {len(leaning)} of 9 models have positive mean Δ and {len(classical_leaning)} have negative mean Δ. "
        "These describe label alignment, not internal implementation of either framework. CD is a gold answer category, not evidence of subjective uncertainty."
    )
    lines.append(
        "The response settings differ: FLAN-T5 contributes one deterministic forced-choice output per question, "
        "whereas vLLM contributes ten sampled Chain-of-Thought generations per question."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main():
    targets, category_counts = load_targets()
    if not validate_category_sample(category_counts):
        raise SystemExit(f"Gold disagreement categories do not sum to 69: {dict(category_counts)}")
    denominators = denominator_counts(targets)
    model_results, validation_reports, errors = load_all_models(targets, denominators)
    if errors:
        print("VALIDATION FAILED; no analysis outputs were written:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    table43_checks, table43_errors = validate_table43(model_results)
    if table43_errors:
        print("TABLE 4.3 REPRODUCTION FAILED; no analysis outputs were written:")
        for error in table43_errors:
            print(f"- {error}")
        raise SystemExit(1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    alignment_rows = build_alignment_csv(model_results, targets, denominators)
    alignment_fields = list(alignment_rows[0])
    write_csv(REPORT_DIR / "alignment_decomposition.csv", alignment_rows, alignment_fields)

    neither_rows = build_neither_csv(model_results, targets, denominators)
    write_csv(REPORT_DIR / "neither_contradictions.csv", neither_rows, list(neither_rows[0]))
    write_categories_csv(REPORT_DIR / "gold_disagreement_categories.csv", targets)
    write_alignment_latex(TABLES_DIR / "alignment_decomposition.tex", model_results, targets, denominators)
    write_neither_latex(TABLES_DIR / "neither_contradictions.tex", model_results, targets, denominators)
    summary_lines = write_summary(
        REPORT_DIR / "neither_alignment_summary.txt", targets, category_counts,
        model_results, validation_reports, table43_checks, denominators,
    )

    print(f"Validated {len(targets)} unique neutral-role disagreement questions.")
    print("Gold categories:", ", ".join(f"{key}={category_counts[key]}" for key in CATEGORY_INFO))
    print("All 9 models passed exact-key, label/content, duplicate/missing, and Table 4.3 reproduction checks.")
    print(f"Wrote {TABLES_DIR / 'alignment_decomposition.tex'}")
    print(f"Wrote {TABLES_DIR / 'neither_contradictions.tex'}")
    print(f"Wrote analysis CSVs and summary under {REPORT_DIR}")
    print("\n".join(summary_lines[12:]))


if __name__ == "__main__":
    main()
