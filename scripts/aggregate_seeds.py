"""
Aggregate seeded CoT runs into mean +/- std tables.

For each model (DeepSeek, Mistral) we compute, per seed:

  Overall (role=none):
    - pred distribution: A%, B%, C(NVC)%

  Disagreement subset (classical != WCS gold, role=none):
    - classical_match_pct
    - wcs_match_pct
    - neither_pct
    - unparsed_pct
    - nvc_pct  (fraction of disagreement items where model predicted C)

Then aggregates mean and std across seeds.

Writes:
  outputs/report/seeds_per_seed.csv          # one row per (model, seed)
  outputs/report/seeds_mean_std.csv          # one row per model with mean +/- std
  outputs/report/seeds_nvc_disagreement.csv  # NVC-focused, mean +/- std per model
  outputs/report/seeds_disagreement_none_cot.csv
                                             # replicates disagreement_none_cot format,
                                             # mean +/- std across seeds

Encoding: pred_index 0=A(Yes) 1=B(No) 2=C(NVC).
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
        "pattern": "deepseek_8b_cot1024_sub64_seed{seed}_predictions.jsonl",
        "seeds": [0, 1, 2, 3, 4],
    },
    "Mistral-7B": {
        "pattern": "mistral_7b_cot_sub256_seed{seed}_predictions.jsonl",
        "seeds": [0, 1, 3, 4],
    },
}

ROLE = "none"


def load_jsonl_by_id(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def score_seed(pred_path, gold):
    """Return metrics dict for a single seed file."""
    preds = load_jsonl_by_id(pred_path)

    overall = Counter()
    overall_total = 0

    dis = Counter()
    dis_total = 0
    dis_nvc = 0

    for iid, p in preds.items():
        item = gold.get(iid)
        if item is None or item["role"] != ROLE:
            continue
        for code, pc in p["conclusions"].items():
            pred = pc.get("pred_index", -1)
            overall[pred] += 1
            overall_total += 1

            gc = item["conclusions"].get(code)
            if gc is None:
                continue
            classical = gc["classical"]
            wcs = gc["wcs"]
            if classical == wcs:
                continue

            dis_total += 1
            if pred == 2:
                dis_nvc += 1
            if pred == classical:
                dis["classical"] += 1
            elif pred == wcs:
                dis["wcs"] += 1
            elif pred in (0, 1, 2):
                dis["neither"] += 1
            else:
                dis["unparsed"] += 1

    def pct(n, d):
        return 100.0 * n / d if d else 0.0

    return {
        "n_overall": overall_total,
        "overall_A_pct": pct(overall.get(0, 0), overall_total),
        "overall_B_pct": pct(overall.get(1, 0), overall_total),
        "overall_C_pct": pct(overall.get(2, 0), overall_total),
        "n_disagreement": dis_total,
        "classical_match_pct": pct(dis["classical"], dis_total),
        "wcs_match_pct": pct(dis["wcs"], dis_total),
        "neither_pct": pct(dis["neither"], dis_total),
        "unparsed_pct": pct(dis["unparsed"], dis_total),
        "disagreement_nvc_pct": pct(dis_nvc, dis_total),
    }


def mean_std(values):
    if not values:
        return (0.0, 0.0)
    m = statistics.fmean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return (m, s)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = load_jsonl_by_id(GOLD_FILE)

    per_seed_rows = []
    per_model = {}

    for model, cfg in MODELS.items():
        per_model[model] = []
        for seed in cfg["seeds"]:
            path = PRED_DIR / cfg["pattern"].format(seed=seed)
            if not path.exists():
                print(f"[skip] {path} not found")
                continue
            metrics = score_seed(path, gold)
            metrics["model"] = model
            metrics["seed"] = seed
            per_seed_rows.append(metrics)
            per_model[model].append(metrics)

    # ---------------- per-seed CSV ----------------
    per_seed_path = OUTPUT_DIR / "seeds_per_seed.csv"
    fields = [
        "model", "seed", "n_overall",
        "overall_A_pct", "overall_B_pct", "overall_C_pct",
        "n_disagreement",
        "classical_match_pct", "wcs_match_pct",
        "neither_pct", "unparsed_pct",
        "disagreement_nvc_pct",
    ]
    with open(per_seed_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_seed_rows:
            w.writerow({k: (round(r[k], 2) if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"Wrote {per_seed_path}")

    # ---------------- mean +/- std CSV ----------------
    metric_keys = [
        "overall_A_pct", "overall_B_pct", "overall_C_pct",
        "classical_match_pct", "wcs_match_pct",
        "neither_pct", "unparsed_pct",
        "disagreement_nvc_pct",
    ]

    summary_path = OUTPUT_DIR / "seeds_mean_std.csv"
    header = ["model", "n_seeds"]
    for k in metric_keys:
        header += [f"{k}_mean", f"{k}_std"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for model, rows in per_model.items():
            if not rows:
                continue
            out = [model, len(rows)]
            for k in metric_keys:
                m, s = mean_std([r[k] for r in rows])
                out += [round(m, 2), round(s, 2)]
            w.writerow(out)
    print(f"Wrote {summary_path}")

    # ---------------- NVC-focused (mean +/- std) ----------------
    nvc_path = OUTPUT_DIR / "seeds_nvc_disagreement.csv"
    with open(nvc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "n_seeds",
            "overall_nvc_pct_mean", "overall_nvc_pct_std",
            "disagreement_nvc_pct_mean", "disagreement_nvc_pct_std",
            "classical_match_pct_mean", "classical_match_pct_std",
            "wcs_match_pct_mean", "wcs_match_pct_std",
        ])
        for model, rows in per_model.items():
            if not rows:
                continue
            o_m, o_s = mean_std([r["overall_C_pct"] for r in rows])
            d_m, d_s = mean_std([r["disagreement_nvc_pct"] for r in rows])
            c_m, c_s = mean_std([r["classical_match_pct"] for r in rows])
            w_m, w_s = mean_std([r["wcs_match_pct"] for r in rows])
            w.writerow([
                model, len(rows),
                round(o_m, 2), round(o_s, 2),
                round(d_m, 2), round(d_s, 2),
                round(c_m, 2), round(c_s, 2),
                round(w_m, 2), round(w_s, 2),
            ])
    print(f"Wrote {nvc_path}")

    # ---------------- disagreement_none_cot format (mean +/- std) ----------------
    dis_path = OUTPUT_DIR / "seeds_disagreement_none_cot.csv"
    with open(dis_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "n_seeds", "n_disagreement",
            "classical_match_pct_mean", "classical_match_pct_std",
            "wcs_match_pct_mean", "wcs_match_pct_std",
            "neither_pct_mean", "neither_pct_std",
            "unparsed_pct_mean", "unparsed_pct_std",
            "wcs_minus_classical_pp_mean", "wcs_minus_classical_pp_std",
        ])
        for model, rows in per_model.items():
            if not rows:
                continue
            n_dis = rows[0]["n_disagreement"]
            c_m, c_s = mean_std([r["classical_match_pct"] for r in rows])
            w_m, w_s = mean_std([r["wcs_match_pct"] for r in rows])
            n_m, n_s = mean_std([r["neither_pct"] for r in rows])
            u_m, u_s = mean_std([r["unparsed_pct"] for r in rows])
            delta = [r["wcs_match_pct"] - r["classical_match_pct"] for r in rows]
            d_m, d_s = mean_std(delta)
            w.writerow([
                model, len(rows), n_dis,
                round(c_m, 2), round(c_s, 2),
                round(w_m, 2), round(w_s, 2),
                round(n_m, 2), round(n_s, 2),
                round(u_m, 2), round(u_s, 2),
                round(d_m, 2), round(d_s, 2),
            ])
    print(f"Wrote {dis_path}")

    # ---------------- terminal summary ----------------
    print("\n=== Per-model summary (mean +/- std across seeds) ===")
    for model, rows in per_model.items():
        if not rows:
            continue
        print(f"\n{model}  (n_seeds={len(rows)}, n_disagreement={rows[0]['n_disagreement']})")
        for k in metric_keys:
            m, s = mean_std([r[k] for r in rows])
            print(f"  {k:<26}: {m:6.2f} +/- {s:.2f}")


if __name__ == "__main__":
    main()
