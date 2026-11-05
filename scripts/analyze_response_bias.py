#!/usr/bin/env python3
"""Test whether Table 4.3 alignment exceeds answer-frequency-only baselines."""

import csv
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analyze_neither_alignment import (
    CATEGORY_INFO,
    LABELS,
    MODEL_SPECS,
    TABLE_43_EXPECTED,
    analyze_run,
    denominator_counts,
    load_prediction_run,
    load_targets,
    prediction_paths,
    validate_table43,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs/report"
TABLES_DIR = ROOT / "tables"
N_SHUFFLES = 5000
RANDOM_SEED = 20261104
INVALID = 3


def valid_index(prediction):
    value = prediction.get("pred_index")
    return value if type(value) is int and value in LABELS else INVALID


def pct(count, denominator):
    return 100.0 * count / denominator if denominator else 0.0


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def expected_random_match_pct(predictions, targets, gold_key):
    """Exact expectation over all permutations of the fixed answer multiset."""
    n = len(targets)
    answer_counts = Counter(predictions)
    gold_counts = Counter(target[gold_key] for target in targets)
    expected_matches = sum(answer_counts[label] * gold_counts[label] for label in LABELS) / n
    return pct(expected_matches, n)


def shuffle_baseline(predictions, targets, rng):
    n = len(targets)
    samples = {"classical_pct": [], "wcs_pct": [], "delta_pp": []}
    target_pairs = [(target["classical"], target["wcs"]) for target in targets]

    for _ in range(N_SHUFFLES):
        shuffled = rng.sample(predictions, n)
        classical_matches = sum(answer == classical for answer, (classical, _) in zip(shuffled, target_pairs))
        wcs_matches = sum(answer == wcs for answer, (_, wcs) in zip(shuffled, target_pairs))
        classical_pct = pct(classical_matches, n)
        wcs_pct = pct(wcs_matches, n)
        samples["classical_pct"].append(classical_pct)
        samples["wcs_pct"].append(wcs_pct)
        samples["delta_pp"].append(wcs_pct - classical_pct)

    baseline = {}
    for metric, shuffled_values in samples.items():
        baseline[f"{metric}_mc_mean"], baseline[f"{metric}_null_sd"] = mean_sd(shuffled_values)

    baseline["classical_pct_exact"] = expected_random_match_pct(predictions, targets, "classical")
    baseline["wcs_pct_exact"] = expected_random_match_pct(predictions, targets, "wcs")
    baseline["delta_pp_exact"] = baseline["wcs_pct_exact"] - baseline["classical_pct_exact"]

    for metric in ("classical_pct", "wcs_pct", "delta_pp"):
        monte_carlo_mean = baseline[f"{metric}_mc_mean"]
        null_sd = baseline[f"{metric}_null_sd"]
        tolerance = 5 * null_sd / (N_SHUFFLES ** 0.5) + 0.01
        if abs(monte_carlo_mean - baseline[f"{metric}_exact"]) > tolerance:
            raise ValueError(f"Shuffled {metric} mean does not match exact permutation expectation")
    return baseline


def summarize_run(targets, predictions, rng):
    n = len(targets)
    ordered_keys = list(targets)
    ordered_targets = [targets[key] for key in ordered_keys]
    answers = [valid_index(predictions[key]) for key in ordered_keys]
    core = analyze_run(targets, predictions)

    observed = {
        "classical_matches": sum(answer == target["classical"] for answer, target in zip(answers, ordered_targets)),
        "wcs_matches": sum(answer == target["wcs"] for answer, target in zip(answers, ordered_targets)),
        "classical_only": core["outcomes"]["classical_only"],
        "wcs_only": core["outcomes"]["wcs_only"],
        "neither": core["outcomes"]["neither"],
        "invalid": core["outcomes"]["invalid"],
    }
    observed["classical_pct"] = pct(observed["classical_matches"], n)
    observed["wcs_pct"] = pct(observed["wcs_matches"], n)
    observed["delta_pp"] = observed["wcs_pct"] - observed["classical_pct"]
    observed["outcome_pct"] = {key: pct(observed[key], n) for key in ("classical_only", "wcs_only", "neither", "invalid")}
    observed["answer_counts"] = Counter(answers)
    observed["answer_pct"] = {label: pct(observed["answer_counts"][label], n) for label in (*LABELS, INVALID)}

    categories = {}
    for category in CATEGORY_INFO:
        category_targets = [target for target in ordered_targets if target["category"] == category]
        keys = [key for key, target in zip(ordered_keys, ordered_targets) if target["category"] == category]
        category_answers = [valid_index(predictions[key]) for key in keys]
        category_n = len(category_targets)
        c_matches = sum(answer == target["classical"] for answer, target in zip(category_answers, category_targets))
        w_matches = sum(answer == target["wcs"] for answer, target in zip(category_answers, category_targets))
        categories[category] = {
            "n": category_n,
            "classical_matches": c_matches,
            "classical_pct": pct(c_matches, category_n),
            "wcs_matches": w_matches,
            "wcs_pct": pct(w_matches, category_n),
            "wcs_yes_matches": sum(answer == 0 for answer, target in zip(category_answers, category_targets) if target["wcs"] == 0),
            "wcs_yes_n": sum(target["wcs"] == 0 for target in category_targets),
            "wcs_no_matches": sum(answer == 1 for answer, target in zip(category_answers, category_targets) if target["wcs"] == 1),
            "wcs_no_n": sum(target["wcs"] == 1 for target in category_targets),
            "answer_counts": Counter(category_answers),
        }
        categories[category]["wcs_yes_pct"] = (
            pct(categories[category]["wcs_yes_matches"], categories[category]["wcs_yes_n"])
            if categories[category]["wcs_yes_n"] else None
        )
        categories[category]["wcs_no_pct"] = (
            pct(categories[category]["wcs_no_matches"], categories[category]["wcs_no_n"])
            if categories[category]["wcs_no_n"] else None
        )

    baseline = shuffle_baseline(answers, ordered_targets, rng)
    return {"observed": observed, "categories": categories, "baseline": baseline}


def validate_all(targets, model_runs):
    failures = []
    if len(targets) != 69:
        failures.append(f"Expected 69 target questions, found {len(targets)}")
    if sum(1 for target in targets.values() if target["wcs"] in (0, 1)) != 45:
        failures.append("Expected 45 WCS-definite questions")
    if sum(1 for target in targets.values() if target["classical"] in (0, 1)) != 24:
        failures.append("Expected 24 Classical-definite questions")
    category_counts = Counter(target["category"] for target in targets.values())
    if category_counts != Counter({"A": 21, "B": 24, "C": 12, "D": 12}):
        failures.append(f"Unexpected gold category counts: {dict(category_counts)}")
    if sum(category_counts.values()) != 69:
        failures.append("Gold category counts do not sum to 69")

    table43_input = {"targets": targets, "denominators": denominator_counts(targets)}
    for spec in MODEL_SPECS:
        runs = model_runs.get(spec["name"], [])
        if not runs:
            failures.append(f"No usable runs for {spec['name']}")
            continue
        core_runs = [run["core"] for run in runs]
        table43_input[spec["name"]] = core_runs
        for run in runs:
            if sum(run["analysis"]["observed"]["outcome_pct"].values()) < 99.999:
                failures.append(f"Outcome percentages do not sum to 100 for {spec['name']}")
            if sum(run["analysis"]["observed"][key] for key in ("classical_only", "wcs_only", "neither", "invalid")) != 69:
                failures.append(f"Outcome counts do not sum to 69 for {spec['name']}")
    table43, table43_errors = validate_table43(table43_input)
    failures.extend(table43_errors)
    return failures, category_counts, table43


def aggregate_runs(runs, getter):
    return mean_sd([getter(run) for run in runs])


def model_summary_rows(model_runs):
    rows = []
    for spec in MODEL_SPECS:
        name = spec["name"]
        runs = model_runs[name]
        row = {
            "model": name,
            "inference_setting": spec["setting"],
            "runs": len(runs),
            "questions_per_run": 69,
            "baseline_shuffles_per_run": N_SHUFFLES,
        }
        for metric in ("classical_pct", "wcs_pct", "delta_pp"):
            observed_mean, observed_sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"][metric])
            baseline_mean, baseline_sd = aggregate_runs(runs, lambda item: item["analysis"]["baseline"][f"{metric}_exact"])
            excess_mean, excess_sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"][metric] - item["analysis"]["baseline"][f"{metric}_exact"])
            null_sd_mean = statistics.mean(item["analysis"]["baseline"][f"{metric}_null_sd"] for item in runs)
            row[f"observed_{metric}_mean_pct"] = observed_mean
            row[f"observed_{metric}_sd_pct"] = observed_sd
            row[f"shuffled_expected_{metric}_mean_pct"] = baseline_mean
            row[f"shuffled_expected_{metric}_run_sd_pct"] = baseline_sd
            row[f"excess_{metric}_mean_pp"] = excess_mean
            row[f"excess_{metric}_sd_pp"] = excess_sd
            row[f"shuffle_null_{metric}_sd_mean_pp"] = null_sd_mean
        for answer, label in [*LABELS.items(), (INVALID, "Invalid")]:
            mean, sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"]["answer_pct"][answer])
            row[f"answer_{label.lower().replace(' ', '_')}_mean_pct"] = mean
            row[f"answer_{label.lower().replace(' ', '_')}_sd_pct"] = sd
        row["table43_validation"] = "PASS"
        rows.append(row)
    return rows


def category_rows(model_runs):
    rows = []
    for spec in MODEL_SPECS:
        for category, info in CATEGORY_INFO.items():
            runs = model_runs[spec["name"]]
            for metric in ("classical_pct", "wcs_pct", "wcs_yes_pct", "wcs_no_pct"):
                denominator_key = {
                    "wcs_yes_pct": "wcs_yes_n",
                    "wcs_no_pct": "wcs_no_n",
                }.get(metric)
                metric_denominator = (
                    runs[0]["analysis"]["categories"][category][denominator_key]
                    if denominator_key else runs[0]["analysis"]["categories"][category]["n"]
                )
                if metric_denominator == 0:
                    continue
                values = [run["analysis"]["categories"][category][metric] for run in runs]
                mean, sd = mean_sd(values)
                denominator = metric_denominator
                rows.append({
                    "model": spec["name"],
                    "inference_setting": spec["setting"],
                    "category": category,
                    "category_description": info["description"],
                    "metric": metric,
                    "observed_matches_mean": mean * denominator / 100,
                    "observed_matches_sd": sd * denominator / 100,
                    "observed_rate_mean_pct": mean,
                    "observed_rate_sd_pct": sd,
                    "denominator_questions": denominator,
                    "questions_total": 69,
                    "runs": len(runs),
                })
    return rows


def answer_distribution_rows(model_runs):
    rows = []
    for spec in MODEL_SPECS:
        runs = model_runs[spec["name"]]
        for index, label in [*LABELS.items(), (INVALID, "Invalid")]:
            count_mean, count_sd = mean_sd([run["analysis"]["observed"]["answer_counts"][index] for run in runs])
            pct_mean, pct_sd = mean_sd([run["analysis"]["observed"]["answer_pct"][index] for run in runs])
            rows.append({
                "model": spec["name"],
                "inference_setting": spec["setting"],
                "answer": label,
                "count_mean_per_69": count_mean,
                "count_sd_across_runs": count_sd,
                "rate_mean_pct": pct_mean,
                "rate_sd_pct": pct_sd,
                "denominator_per_run": 69,
                "runs": len(runs),
            })
    return rows


def format_value(mean, sd, runs, suffix="\\%"):
    if runs == 1:
        return f"{mean:.2f}{suffix}"
    return f"{mean:.2f} $\\pm$ {sd:.2f}{suffix}"


def write_appendix_latex(path, model_runs):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Observed alignment compared with a response-frequency baseline. For each model run, predictions were shuffled 5,000 times across the same 69 questions, preserving the model's Yes/No/CD/Invalid frequencies. Positive excess indicates alignment above that baseline.}",
        r"\label{tab:response-bias-alignment}",
        r"\textbf{Classical alignment}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Observed (\%) & Shuffled base (\%) & Excess (pp) \\",
        r"\midrule",
    ]
    for spec in MODEL_SPECS:
        runs = model_runs[spec["name"]]
        row = [spec["short"]]
        observed_mean, observed_sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"]["classical_pct"])
        expected_mean, expected_sd = aggregate_runs(runs, lambda item: item["analysis"]["baseline"]["classical_pct_exact"])
        excess_mean, excess_sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"]["classical_pct"] - item["analysis"]["baseline"]["classical_pct_exact"])
        row.extend([
            format_value(observed_mean, observed_sd, len(runs)),
            format_value(expected_mean, expected_sd, len(runs)),
            format_value(excess_mean, excess_sd, len(runs), " pp"),
        ])
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\textbf{WCS alignment}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Observed (\%) & Shuffled base (\%) & Excess (pp) \\",
        r"\midrule",
    ])
    for spec in MODEL_SPECS:
        runs = model_runs[spec["name"]]
        row = [spec["short"]]
        observed_mean, observed_sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"]["wcs_pct"])
        expected_mean, expected_sd = aggregate_runs(runs, lambda item: item["analysis"]["baseline"]["wcs_pct_exact"])
        excess_mean, excess_sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"]["wcs_pct"] - item["analysis"]["baseline"]["wcs_pct_exact"])
        row.extend([
            format_value(observed_mean, observed_sd, len(runs)),
            format_value(expected_mean, expected_sd, len(runs)),
            format_value(excess_mean, excess_sd, len(runs), " pp"),
        ])
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\textbf{WCS-minus-Classical difference ($\Delta$)}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Observed $\Delta$ & Shuffled expected $\Delta$ & Excess $\Delta$ \\",
        r"\midrule",
    ])
    for spec in MODEL_SPECS:
        runs = model_runs[spec["name"]]
        row = [spec["short"]]
        for metric in ("observed", "baseline", "excess"):
            if metric == "observed":
                mean, sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"]["delta_pp"])
            elif metric == "baseline":
                mean, sd = aggregate_runs(runs, lambda item: item["analysis"]["baseline"]["delta_pp_exact"])
            else:
                mean, sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"]["delta_pp"] - item["analysis"]["baseline"]["delta_pp_exact"])
            row.append(format_value(mean, sd, len(runs), " pp"))
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}",
        r"\scriptsize Base is the exact expected match rate under a uniform random permutation of the fixed answer multiset; 5,000 actual shuffles estimate the corresponding null SD and check the expectation. Excess = observed minus shuffled expectation. Percentages use all 69 questions. For vLLM, $\pm$ is the SD across ten generations; for FLAN-T5, outputs are deterministic. This baseline controls only answer frequencies, not question difficulty or dependencies among conclusions.",
        r"\end{minipage}",
        r"\end{table}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_excess_delta_figure(path, model_runs):
    labels = []
    means = []
    stds = []
    for spec in MODEL_SPECS:
        runs = model_runs[spec["name"]]
        mean, sd = aggregate_runs(
            runs,
            lambda item: item["analysis"]["observed"]["excess_delta_pp"],
        )
        labels.append(spec["short"])
        means.append(mean)
        stds.append(sd)

    colors = [
        "#4E805D" if value > 0 else "#5A7898" if value < 0 else "#78818A"
        for value in means
    ]
    positions = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.barh(
        positions,
        means,
        xerr=stds,
        height=0.62,
        color=colors,
        edgecolor="#36414A",
        linewidth=0.5,
        ecolor="#36414A",
        capsize=2,
        error_kw={"elinewidth": 0.9},
    )
    axis.axvline(0, color="#303840", linewidth=1.0, zorder=0)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.xaxis.grid(True, color="#D9DEE3", linewidth=0.7, linestyle="--")
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    lower = min(value - error for value, error in zip(means, stds))
    upper = max(value + error for value, error in zip(means, stds))
    padding = max(3.0, (upper - lower) * 0.12)
    axis.set_xlim(lower - padding, upper + padding + 2.5)

    for position, (value, error) in enumerate(zip(means, stds)):
        if value >= 0:
            axis.text(value + error + 0.5, position, f"{value:+.2f}", va="center", ha="left", fontsize=8)
        else:
            axis.text(value - error - 0.5, position, f"{value:+.2f}", va="center", ha="right", fontsize=8)

    axis.legend(
        handles=[
            Patch(facecolor="#4E805D", edgecolor="#36414A", label="WCS above baseline"),
            Patch(facecolor="#5A7898", edgecolor="#36414A", label="Classical above baseline"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def write_latex(path):
    lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\includegraphics[width=0.95\linewidth]{tables/response_bias_excess_delta.pdf}",
        r"\caption{The plotted difference is the observed WCS-minus-Classical match-rate difference minus its expected value under a shuffle preserving each model's Yes/No/CD/Invalid frequencies. Positive values indicate more WCS alignment than answer frequency alone predicts; negative values indicate more Classical alignment. Error bars show standard deviation across ten vLLM generations; FLAN-T5 outputs are deterministic.}",
        r"\label{fig:response-bias-delta}",
        r"\end{figure}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, model_runs, category_counts):
    lines = [
        "RESPONSE-FREQUENCY BASELINE ANALYSIS",
        "====================================",
        "Target: the same 69 neutral-role questions where Classical and WCS disagree.",
        "Categories: " + ", ".join(f"{category}={category_counts[category]}" for category in CATEGORY_INFO),
        f"Baseline: {N_SHUFFLES} random permutations per model run, preserving each run's Yes/No/CD/Invalid counts.",
        "The reported shuffled expectation is exact for a uniform permutation; simulations estimate its null spread and verify it.",
        "",
        "MODEL RESULTS (mean across runs; vLLM SDs are across ten generations)",
        "--------------------------------------------------------------------",
    ]
    for spec in MODEL_SPECS:
        runs = model_runs[spec["name"]]
        observed, baseline, excess = [], [], []
        for run in runs:
            result = run["analysis"]
            observed.append(result["observed"]["delta_pp"])
            baseline.append(result["baseline"]["delta_pp_exact"])
            excess.append(result["observed"]["delta_pp"] - result["baseline"]["delta_pp_exact"])
        obs_mean, obs_sd = mean_sd(observed)
        base_mean, base_sd = mean_sd(baseline)
        excess_mean, excess_sd = mean_sd(excess)
        direction = "WCS" if excess_mean > 0 else "Classical" if excess_mean < 0 else "neither"
        lines.append(
            f"{spec['name']}: observed Δ={obs_mean:+.2f} pp; answer-frequency baseline Δ={base_mean:+.2f} pp; "
            f"excess Δ={excess_mean:+.2f} pp; excess direction={direction}."
        )
    lines.extend([
        "",
        "LIMITATIONS",
        "-----------",
        "The baseline preserves each model run's marginal answer frequencies but randomizes which answer is assigned to which question.",
        "It therefore controls answer-frequency bias only; it does not control question difficulty or dependencies among conclusions.",
        "The 69 questions are the analysis sample. Ten vLLM generations measure output variability, not ten independent question samples.",
        "Excess alignment is descriptive and is not a significance test or evidence of internal implementation of either semantics.",
        "FLAN-T5 contributes one deterministic forced-choice output; vLLM contributes sampled Chain-of-Thought generations.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def run():
    targets, category_counts = load_targets()
    model_runs = {}
    validation_errors = []
    rng = random.Random(RANDOM_SEED)

    for spec in MODEL_SPECS:
        paths = prediction_paths(spec)
        if spec["setting"] == "sampled vLLM CoT" and len(paths) != 10:
            validation_errors.append(f"{spec['name']}: expected 10 vLLM generations, found {len(paths)}")
        runs = []
        for path in paths:
            predictions, report = load_prediction_run(path, targets)
            if report["matched"] != 69 or report["missing"] or report["duplicates"] or report["mismatches"]:
                validation_errors.append(
                    f"{spec['name']} {path.name}: matched={report['matched']}, missing={report['missing']}, "
                    f"duplicate_keys={report['duplicates']}, content/gold mismatches={len(report['mismatches'])}"
                )
                continue
            core = analyze_run(targets, predictions)
            analysis = summarize_run(targets, predictions, rng)
            runs.append({"path": path, "predictions": predictions, "core": core, "analysis": analysis})
        if not runs:
            validation_errors.append(f"{spec['name']}: no valid prediction runs")
        model_runs[spec["name"]] = runs

    table43_input = {"targets": targets, "denominators": denominator_counts(targets)}
    for spec in MODEL_SPECS:
        table43_input[spec["name"]] = [run["core"] for run in model_runs[spec["name"]]]
    table43_checks, table43_errors = validate_table43(table43_input)
    validation_errors.extend(table43_errors)
    if validation_errors:
        print("VALIDATION FAILED; no response-bias outputs written:")
        for error in validation_errors:
            print(f"- {error}")
        raise SystemExit(1)

    responses = [run for runs in model_runs.values() for run in runs]
    for run in responses:
        observed, baseline = run["analysis"]["observed"], run["analysis"]["baseline"]
        run["analysis"]["observed"]["excess_classical_pp"] = observed["classical_pct"] - baseline["classical_pct_exact"]
        run["analysis"]["observed"]["excess_wcs_pp"] = observed["wcs_pct"] - baseline["wcs_pct_exact"]
        run["analysis"]["observed"]["excess_delta_pp"] = observed["delta_pp"] - baseline["delta_pp_exact"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT_DIR / "response_bias_alignment.csv", model_summary_rows(model_runs))
    write_csv(REPORT_DIR / "response_bias_by_category.csv", category_rows(model_runs))
    write_csv(REPORT_DIR / "response_distribution.csv", answer_distribution_rows(model_runs))
    by_run_rows = []
    for spec in MODEL_SPECS:
        for index, run in enumerate(model_runs[spec["name"]], start=1):
            observed, baseline = run["analysis"]["observed"], run["analysis"]["baseline"]
            by_run_rows.append({
                "model": spec["name"],
                "inference_setting": spec["setting"],
                "run_index": index,
                "prediction_file": run["path"].name,
                "questions_denominator": 69,
                "yes_count": observed["answer_counts"][0],
                "no_count": observed["answer_counts"][1],
                "cd_count": observed["answer_counts"][2],
                "invalid_count": observed["answer_counts"][INVALID],
                "observed_classical_pct": observed["classical_pct"],
                "observed_wcs_pct": observed["wcs_pct"],
                "observed_delta_pp": observed["delta_pp"],
                "expected_classical_pct_exact": baseline["classical_pct_exact"],
                "expected_wcs_pct_exact": baseline["wcs_pct_exact"],
                "expected_delta_pp_exact": baseline["delta_pp_exact"],
                "excess_classical_pp": observed["excess_classical_pp"],
                "excess_wcs_pp": observed["excess_wcs_pp"],
                "excess_delta_pp": observed["excess_delta_pp"],
                "shuffle_classical_null_sd_pp": baseline["classical_pct_null_sd"],
                "shuffle_wcs_null_sd_pp": baseline["wcs_pct_null_sd"],
                "shuffle_delta_null_sd_pp": baseline["delta_pp_null_sd"],
                "n_shuffles": N_SHUFFLES,
            })
    write_csv(REPORT_DIR / "response_bias_by_run.csv", by_run_rows)
    figure_path = TABLES_DIR / "response_bias_excess_delta.pdf"
    write_excess_delta_figure(figure_path, model_runs)
    write_latex(TABLES_DIR / "response_bias_analysis.tex")
    write_appendix_latex(TABLES_DIR / "response_bias_analysis_appendix.tex", model_runs)
    summary = write_summary(REPORT_DIR / "response_bias_summary.txt", model_runs, category_counts)

    print(f"Validated {len(targets)} common questions; category counts: {dict(category_counts)}")
    print("All models passed matching, gold/content, duplicate/missing, and Table 4.3 checks.")
    print(f"Performed {N_SHUFFLES:,} answer shuffles per prediction run (seed {RANDOM_SEED}).")
    print(f"Wrote {TABLES_DIR / 'response_bias_analysis.tex'} with figure {figure_path}")
    print(f"Wrote {TABLES_DIR / 'response_bias_analysis_appendix.tex'} with the full table")
    print("Wrote four CSV reports and a summary under outputs/report/.")
    print("\n".join(summary[6:]))
    print("\nModel-level observed, shuffled expected, and excess Delta:")
    for spec in MODEL_SPECS:
        runs = model_runs[spec["name"]]
        observed_mean, observed_sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"]["delta_pp"])
        base_mean, base_sd = aggregate_runs(runs, lambda item: item["analysis"]["baseline"]["delta_pp_exact"])
        excess_mean, excess_sd = aggregate_runs(runs, lambda item: item["analysis"]["observed"]["excess_delta_pp"])
        print(f"{spec['name']}: observed {observed_mean:+.2f}, shuffled {base_mean:+.2f}, excess {excess_mean:+.2f} pp")


def main():
    run()


if __name__ == "__main__":
    main()
