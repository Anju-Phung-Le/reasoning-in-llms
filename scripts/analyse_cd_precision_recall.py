"""
NVC-focused analysis on seeded CoT runs (v2).

For each model, aggregated as mean +/- std across seeds:

  1. Gold-label distribution per framework (A / B / NVC).
     - How often is the gold label A, B, or NVC under Classical Logic?
     - How often under WCS?
     (This is a property of the dataset, not the model — reported once.)

  2. Model answer distribution overall (A / B / NVC).

  3. Case breakdown — group each (item, conclusion) by whether the two
     frameworks assign NVC:
       both     : classical = NVC AND wcs = NVC
       only_cls : classical = NVC AND wcs != NVC
       only_wcs : classical != NVC AND wcs = NVC
       neither  : classical != NVC AND wcs != NVC
     For each bucket, report the fraction of model answers that are NVC.

  4. Precision and recall on NVC.
       "positive" = at least one framework assigns NVC to this item
       precision = P(at least one gold = NVC | model = NVC)
       recall    = P(model = NVC | at least one gold = NVC)

Encoding: pred_index 0 = A, 1 = B, 2 = NVC.

Writes:
  outputs/report/cd_analysis_summary.csv
"""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

GOLD_FILE = Path("data/processed/dataset_all_forms.jsonl")
PRED_DIR = Path("data/predictions")
OUTPUT_DIR = Path("outputs/report")

def _vllm_files(subdir, prefix, seeds):
    return [f"vllm/{subdir}/{prefix}{s}.jsonl" for s in seeds]


# All models: sub256, CoT, nucleus (T=0.6, p=0.95), 10 seeds via vLLM (supervisor runs).
MODELS = {
    "Mistral-7B": {
        "files": _vllm_files(
            "mistral7b_vllm",
            "mistral_7b_cot_sub256_vllm_predictions_sample",
            range(10),
        ),
    },
    "Llama-3.1-8B": {
        "files": _vllm_files(
            "llama8b_vllm",
            "llama31_8b_sub256_predictions_sample",
            range(10),
        ),
    },
    "OLMo-2-13B": {
        "files": _vllm_files(
            "olmo2_13b_vllm",
            "olmo2_13b_sub256_predictions_sample",
            range(10),
        ),
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "files": _vllm_files(
            "deepseek_llama_8b_vllm",
            "vllm_Deepseek_256_predictions_sample",
            range(10),
        ),
    },
    "DeepSeek-R1-Distill-Qwen-14B": {
        "files": _vllm_files(
            "deepseek_qwen_14b_vllm",
            "deepseek14b_sub256_predictions_sample",
            range(10),
        ),
    },
    "Mistral-Small-24B": {
        "files": _vllm_files(
            "mistral24B_vllm",
            "mistral24b_vllm_1_predictions_sample",
            range(5),
        ) + _vllm_files(
            "mistral24B_vllm",
            "mistral24b_vllm_2_predictions_sample",
            range(5),
        ),
    },
}

ROLE = "none"
NVC = 2


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold[r["id"]] = r
    return gold


def load_preds(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def bucket(classical, wcs):
    c_nvc = (classical == NVC)
    w_nvc = (wcs == NVC)
    if c_nvc and w_nvc:
        return "both"
    if c_nvc and not w_nvc:
        return "only_cls"
    if not c_nvc and w_nvc:
        return "only_wcs"
    return "neither"


def gold_distribution(gold, pred_ids):
    """
    Return (classical_dist, wcs_dist, total) — proportions of A/B/NVC in
    the gold labels of all (item, conclusion) pairs that both this
    prediction file covers AND are role=none.
    """
    cls = Counter()
    wcs = Counter()
    total = 0
    for iid in pred_ids:
        item = gold.get(iid)
        if item is None or item["role"] != ROLE:
            continue
        for _, gc in item["conclusions"].items():
            cls[gc["classical"]] += 1
            wcs[gc["wcs"]] += 1
            total += 1
    return cls, wcs, total


def score_seed(pred_path, gold):
    preds = load_preds(pred_path)

    # Model answer distribution (role=none)
    model_dist = Counter()
    total = 0

    # Bucketed NVC rate
    bucket_total = Counter()
    bucket_nvc = Counter()

    # Precision/recall on NVC
    tp = 0  # model=NVC, at least one gold=NVC
    fp = 0  # model=NVC, both golds != NVC
    fn = 0  # model!=NVC, at least one gold=NVC
    tn = 0  # model!=NVC, both golds != NVC

    for iid, p in preds.items():
        item = gold.get(iid)
        if item is None or item["role"] != ROLE:
            continue
        for code, pc in p["conclusions"].items():
            gc = item["conclusions"].get(code)
            if gc is None:
                continue
            pred = pc.get("pred_index", -1)
            model_dist[pred] += 1
            total += 1

            b = bucket(gc["classical"], gc["wcs"])
            bucket_total[b] += 1
            if pred == NVC:
                bucket_nvc[b] += 1

            any_nvc_gold = (gc["classical"] == NVC or gc["wcs"] == NVC)
            model_nvc = (pred == NVC)
            if model_nvc and any_nvc_gold:
                tp += 1
            elif model_nvc and not any_nvc_gold:
                fp += 1
            elif not model_nvc and any_nvc_gold:
                fn += 1
            else:
                tn += 1

    def pct(n, d):
        return 100.0 * n / d if d else 0.0

    metrics = {
        "n_overall": total,
        "model_A_pct": pct(model_dist.get(0, 0), total),
        "model_B_pct": pct(model_dist.get(1, 0), total),
        "model_NVC_pct": pct(model_dist.get(NVC, 0), total),
    }
    for b in ("both", "only_cls", "only_wcs", "neither"):
        metrics[f"nvc_rate_{b}_pct"] = pct(bucket_nvc[b], bucket_total[b])
        metrics[f"n_{b}"] = bucket_total[b]

    n_classical_cd_gold = bucket_total["both"] + bucket_total["only_cls"]
    n_wcs_cd_gold = bucket_total["both"] + bucket_total["only_wcs"]
    metrics["n_classical_cd_gold"] = n_classical_cd_gold
    metrics["n_wcs_cd_gold"] = n_wcs_cd_gold
    metrics["nvc_rate_classical_cd_gold_pct"] = pct(
        bucket_nvc["both"] + bucket_nvc["only_cls"], n_classical_cd_gold
    )
    metrics["nvc_rate_wcs_cd_gold_pct"] = pct(
        bucket_nvc["both"] + bucket_nvc["only_wcs"], n_wcs_cd_gold
    )

    metrics["nvc_precision_pct"] = pct(tp, tp + fp)
    metrics["nvc_recall_pct"] = pct(tp, tp + fn)
    f1_denom = (metrics["nvc_precision_pct"] + metrics["nvc_recall_pct"])
    metrics["nvc_f1_pct"] = (
        2 * metrics["nvc_precision_pct"] * metrics["nvc_recall_pct"] / f1_denom
        if f1_denom else 0.0
    )

    return metrics


def mean_std(vals):
    if not vals:
        return (0.0, 0.0)
    m = statistics.fmean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (m, s)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = load_gold(GOLD_FILE)

    # Gold distribution — compute once per model (differs because sub64 vs sub256)
    print("=== Gold-label distribution per framework (role=none) ===\n")
    gold_rows = []
    for model, cfg in MODELS.items():
        first_seed_path = PRED_DIR / cfg["files"][0]
        if not first_seed_path.exists():
            continue
        preds = load_preds(first_seed_path)
        cls_dist, wcs_dist, total = gold_distribution(gold, preds.keys())
        print(f"{model}  (n={total})")
        print(f"  Classical: A={100 * cls_dist[0] / total:5.1f}%  "
              f"B={100 * cls_dist[1] / total:5.1f}%  "
              f"NVC={100 * cls_dist[NVC] / total:5.1f}%")
        print(f"  WCS      : A={100 * wcs_dist[0] / total:5.1f}%  "
              f"B={100 * wcs_dist[1] / total:5.1f}%  "
              f"NVC={100 * wcs_dist[NVC] / total:5.1f}%\n")
        gold_rows.append({
            "model": model,
            "n_overall": total,
            "gold_classical_A_pct": round(100 * cls_dist[0] / total, 2),
            "gold_classical_B_pct": round(100 * cls_dist[1] / total, 2),
            "gold_classical_NVC_pct": round(100 * cls_dist[NVC] / total, 2),
            "gold_wcs_A_pct": round(100 * wcs_dist[0] / total, 2),
            "gold_wcs_B_pct": round(100 * wcs_dist[1] / total, 2),
            "gold_wcs_NVC_pct": round(100 * wcs_dist[NVC] / total, 2),
        })

    # Score all seeds per model
    per_model = {}
    for model, cfg in MODELS.items():
        per_model[model] = []
        for rel_path in cfg["files"]:
            path = PRED_DIR / rel_path
            if not path.exists():
                print(f"[skip] {path} not found")
                continue
            per_model[model].append(score_seed(path, gold))

    # Print summary and write CSV
    print("=== Model NVC analysis (mean +/- std across seeds) ===\n")
    metric_keys = [
        "model_A_pct", "model_B_pct", "model_NVC_pct",
        "nvc_rate_both_pct", "nvc_rate_only_cls_pct",
        "nvc_rate_only_wcs_pct", "nvc_rate_neither_pct",
        "nvc_rate_classical_cd_gold_pct", "nvc_rate_wcs_cd_gold_pct",
        "nvc_precision_pct", "nvc_recall_pct", "nvc_f1_pct",
    ]

    # Print summary and write CSV.
    print("=== Model NVC analysis (mean +/- std across seeds) ===\n")
    metric_keys = [
        "model_A_pct", "model_B_pct", "model_NVC_pct",
        "nvc_rate_both_pct", "nvc_rate_only_cls_pct",
        "nvc_rate_only_wcs_pct", "nvc_rate_neither_pct",
        "nvc_rate_classical_cd_gold_pct", "nvc_rate_wcs_cd_gold_pct",
        "nvc_precision_pct", "nvc_recall_pct", "nvc_f1_pct",
    ]
    csv_rows = []
    for model, rows in per_model.items():
        if not rows:
            continue
        first = rows[0]
        print(f"{model} (n_seeds={len(rows)}, n_overall={first['n_overall']})")
        out_row = {"model": model, "n_seeds": len(rows), "n_overall": first["n_overall"]}
        for key in ("both", "only_cls", "only_wcs", "neither", "classical_cd_gold", "wcs_cd_gold"):
            out_row[f"n_{key}"] = first[f"n_{key}"]
        for key in metric_keys:
            mean, sd = mean_std([row[key] for row in rows])
            out_row[f"{key}_mean"] = round(mean, 2)
            out_row[f"{key}_std"] = round(sd, 2)
        csv_rows.append(out_row)

    if gold_rows:
        with (OUTPUT_DIR / "cd_gold_distribution.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(gold_rows[0]))
            writer.writeheader()
            writer.writerows(gold_rows)
    if csv_rows:
        with (OUTPUT_DIR / "cd_analysis_summary.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    def fmt(mean, sd):
        return f"{mean:.2f} $\\pm$ {sd:.2f}"

    main_tex = OUTPUT_DIR / "cd_analysis_main.tex"
    with main_tex.open("w", encoding="utf-8") as output:
        output.write(
            "\\begin{table}[htbp]\n"
            "\\caption{Model use of the \\emph{Cannot determine} response (option C). Precision, recall and F1 use the combined either-framework gold definition. Values are mean $\\pm$ standard deviation across sampling seeds.}\n"
            "\\label{tab:cannot-determine-analysis}\n"
            "\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Model & Overall CD (\\%) & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\\n\\midrule\n"
        )
        for row in csv_rows:
            output.write(
                f"{row['model']} & {fmt(row['model_NVC_pct_mean'], row['model_NVC_pct_std'])} & "
                f"{fmt(row['nvc_precision_pct_mean'], row['nvc_precision_pct_std'])} & "
                f"{fmt(row['nvc_recall_pct_mean'], row['nvc_recall_pct_std'])} & "
                f"{fmt(row['nvc_f1_pct_mean'], row['nvc_f1_pct_std'])} \\\\\n"
            )
        output.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    bucket_tex = OUTPUT_DIR / "cd_analysis_frameworks.tex"
    first = csv_rows[0]
    n_both = first["n_both"]
    n_only_cls = first["n_only_cls"]
    n_only_wcs = first["n_only_wcs"]
    n_neither = first["n_neither"]
    n_classical_cd_gold = first["n_classical_cd_gold"]
    n_wcs_cd_gold = first["n_wcs_cd_gold"]
    with bucket_tex.open("w", encoding="utf-8") as output:
        output.write(
            "\\begin{table}[H]\n\\centering\n\\scriptsize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            "\\textbf{Panel A: CD response rate by gold-label category}\\\\\n"
            "\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Model & Both & Classical only & WCS only & Neither \\\\\n\\midrule\n"
        )
        for row in csv_rows:
            cells = [row["model"]] + [
                fmt(row[f"nvc_rate_{bucket}_pct_mean"], row[f"nvc_rate_{bucket}_pct_std"])
                for bucket in ("both", "only_cls", "only_wcs", "neither")
            ]
            output.write(" & ".join(cells) + r" \\" + "\n")
        output.write(
            "\\bottomrule\n\\end{tabular}\n\\vspace{4pt}\n"
            "\\textbf{Panel B: Overall framework-specific CD response rate}\\\\\n"
            "\\begin{tabular}{lrr}\n\\toprule\n"
            "Model & Classical CD rate & WCS CD rate \\\\\n\\midrule\n"
        )
        for row in csv_rows:
            cells = [
                row["model"],
                fmt(row["nvc_rate_classical_cd_gold_pct_mean"], row["nvc_rate_classical_cd_gold_pct_std"]),
                fmt(row["nvc_rate_wcs_cd_gold_pct_mean"], row["nvc_rate_wcs_cd_gold_pct_std"]),
            ]
            output.write(" & ".join(cells) + r" \\" + "\n")
        output.write(
            "\\bottomrule\n\\end{tabular}\n"
            "\\caption{Model \\emph{Cannot determine} response rate (\\%) by gold-label category. Both = both frameworks label CD; Classical only = only Classical labels CD; WCS only = only WCS labels CD; Neither = neither labels CD. Bucket sizes are $n=409$, $45$, $24$, and $34$, respectively. Panel B combines Both + Classical only ($n=454$) for Classical and Both + WCS only ($n=433$) for WCS. Values are mean $\\pm$ standard deviation across ten generations.}\n"
            "\\label{tab:cannot-determine-per-frameworks}\n\\end{table}\n"
        )
    print(f"Wrote {main_tex}")
    print(f"Wrote {bucket_tex}")

    # Gold distribution table
    if gold_rows:
        gold_tex = OUTPUT_DIR / "cd_gold_distribution.tex"
        with open(gold_tex, "w", encoding="utf-8") as f:
            f.write(
                "\\begin{table}[htbp]\n"
                "\\caption{Distribution of gold labels A (Yes), B (No), and "
                "NVC (Cannot determine) under each framework, on the subset "
                "used per model (role=neutral).}\n"
                "\\label{tab:nvc-gold-distribution}\n"
                "\\begin{tabular}{llrrrr}\n"
                "\\toprule\n"
                "Model & Framework & $N$ & A (\\%) & B (\\%) & NVC (\\%) \\\\\n"
                "\\midrule\n"
            )
            for r in gold_rows:
                f.write(
                    f"{r['model']} & Classical & {r['n_overall']} & "
                    f"{r['gold_classical_A_pct']:.2f} & "
                    f"{r['gold_classical_B_pct']:.2f} & "
                    f"{r['gold_classical_NVC_pct']:.2f} \\\\\n"
                )
                f.write(
                    f" & WCS & {r['n_overall']} & "
                    f"{r['gold_wcs_A_pct']:.2f} & "
                    f"{r['gold_wcs_B_pct']:.2f} & "
                    f"{r['gold_wcs_NVC_pct']:.2f} \\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        print(f"Wrote {gold_tex}")


if __name__ == "__main__":
    main()
