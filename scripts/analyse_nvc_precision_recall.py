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
  outputs/report/nvc_analysis_summary.csv
"""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

GOLD_FILE = Path("data/processed/dataset_all_forms.jsonl")
PRED_DIR = Path("data/predictions")
OUTPUT_DIR = Path("outputs/report")

MODELS = {
    "DeepSeek-8B": {
        "pattern": "deepseek_8b_cot1024_sub64_v2_seed{seed}_predictions.jsonl",
        "seeds": [0, 1, 2, 3, 4],
    },
    "Mistral-7B": {
        "pattern": "mistral_7b_cot_sub256_v2_seed{seed}_predictions.jsonl",
        "seeds": [0, 1, 3, 4],
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
        first_seed_path = PRED_DIR / cfg["pattern"].format(seed=cfg["seeds"][0])
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
        for seed in cfg["seeds"]:
            path = PRED_DIR / cfg["pattern"].format(seed=seed)
            if not path.exists():
                continue
            per_model[model].append(score_seed(path, gold))

    # Print summary and write CSV
    print("=== Model NVC analysis (mean +/- std across seeds) ===\n")
    metric_keys = [
        "model_A_pct", "model_B_pct", "model_NVC_pct",
        "nvc_rate_both_pct", "nvc_rate_only_cls_pct",
        "nvc_rate_only_wcs_pct", "nvc_rate_neither_pct",
        "nvc_precision_pct", "nvc_recall_pct", "nvc_f1_pct",
    ]

    csv_rows = []
    for model, rows in per_model.items():
        if not rows:
            continue
        r0 = rows[0]
        print(f"{model}  (n_seeds={len(rows)}, n_overall={r0['n_overall']})")
        print(f"  bucket sizes: both={r0['n_both']}  only_cls={r0['n_only_cls']}  "
              f"only_wcs={r0['n_only_wcs']}  neither={r0['n_neither']}")
        out_row = {"model": model, "n_seeds": len(rows), "n_overall": r0["n_overall"]}
        for b in ("both", "only_cls", "only_wcs", "neither"):
            out_row[f"n_{b}"] = r0[f"n_{b}"]
        for k in metric_keys:
            m, s = mean_std([r[k] for r in rows])
            print(f"  {k:<26}: {m:6.2f} +/- {s:.2f}")
            out_row[f"{k}_mean"] = round(m, 2)
            out_row[f"{k}_std"] = round(s, 2)
        print()
        csv_rows.append(out_row)

    # Write CSVs
    if gold_rows:
        gold_path = OUTPUT_DIR / "nvc_gold_distribution.csv"
        with open(gold_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(gold_rows[0].keys()))
            w.writeheader()
            w.writerows(gold_rows)
        print(f"Wrote {gold_path}")

    if csv_rows:
        analysis_path = OUTPUT_DIR / "nvc_analysis_summary.csv"
        with open(analysis_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Wrote {analysis_path}")

    # ---------------------------------------------------------------
    # LaTeX tables
    # ---------------------------------------------------------------
    def fmt(m, s):
        return f"{m:.2f} $\\pm$ {s:.2f}"

    # Main table: precision / recall / F1 + overall NVC rate
    main_tex = OUTPUT_DIR / "nvc_analysis_main.tex"
    with open(main_tex, "w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[htbp]\n"
            "\\caption{NVC prediction quality on seeded CoT runs. "
            "``NVC precision'' is the fraction of model-NVC answers where at "
            "least one framework (Classical Logic or WCS) assigns NVC as gold. "
            "``NVC recall'' is the fraction of items with at least one NVC gold "
            "where the model answers NVC. Values are mean $\\pm$ std across "
            "sampling seeds (nucleus, $T=0.6$, $p=0.95$).}\n"
            "\\label{tab:nvc-analysis}\n"
            "\\begin{tabular}{lrrrrr}\n"
            "\\toprule\n"
            "Model & $N$ & Overall NVC (\\%) & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\\n"
            "\\midrule\n"
        )
        for r in csv_rows:
            f.write(
                f"{r['model']} & {r['n_overall']} & "
                f"{fmt(r['model_NVC_pct_mean'], r['model_NVC_pct_std'])} & "
                f"{fmt(r['nvc_precision_pct_mean'], r['nvc_precision_pct_std'])} & "
                f"{fmt(r['nvc_recall_pct_mean'], r['nvc_recall_pct_std'])} & "
                f"{fmt(r['nvc_f1_pct_mean'], r['nvc_f1_pct_std'])} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Wrote {main_tex}")

    # Bucket table: NVC rate by (classical NVC, WCS NVC) status
    bucket_tex = OUTPUT_DIR / "nvc_analysis_buckets.tex"
    with open(bucket_tex, "w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[htbp]\n"
            "\\caption{Model NVC rate on (item, conclusion) pairs, grouped by "
            "whether each framework assigns NVC as the gold label. "
            "``Both'' = both frameworks say NVC; ``Only Cls.'' = only Classical "
            "Logic; ``Only WCS'' = only WCS; ``Neither'' = neither framework "
            "says NVC. Values are mean $\\pm$ std across sampling seeds.}\n"
            "\\label{tab:nvc-buckets}\n"
            "\\begin{tabular}{lrrrrr}\n"
            "\\toprule\n"
            "Model & Bucket & $n$ & NVC rate (\\%) & & \\\\\n"
            "\\midrule\n"
        )
        for r in csv_rows:
            for b, label in [
                ("both", "Both"),
                ("only_cls", "Only Cls."),
                ("only_wcs", "Only WCS"),
                ("neither", "Neither"),
            ]:
                f.write(
                    f"{r['model'] if b == 'both' else ''} & {label} & "
                    f"{r[f'n_{b}']} & "
                    f"{fmt(r[f'nvc_rate_{b}_pct_mean'], r[f'nvc_rate_{b}_pct_std'])} & & \\\\\n"
                )
            f.write("\\midrule\n")
        # strip trailing \midrule
        pass
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Wrote {bucket_tex}")

    # Gold distribution table
    if gold_rows:
        gold_tex = OUTPUT_DIR / "nvc_gold_distribution.tex"
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
