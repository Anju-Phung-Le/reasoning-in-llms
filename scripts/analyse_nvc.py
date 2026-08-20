"""
Analyse how DeepSeek and Mistral (CoT mode) handle the classical-vs-WCS
disagreement items, with a focus on the NVC ("Cannot determine") option.

Prints:
  1. Overall prediction distribution on sub64 (role=none) for each model.
  2. Item-by-item breakdown of the 12 disagreement items:
     gold classical, gold WCS, model prediction.
  3. How often each model picks NVC on disagreement items vs. overall.

Also writes:
  - outputs/report/analyse_nvc_disagreement_rows.csv
      one row per (item, conclusion) disagreement instance, with gold labels
      and each model's prediction.
  - outputs/report/analyse_nvc_summary.csv
      one row per model, with overall NVC%, disagreement-subset NVC%,
      classical-match%, and WCS-match%.

Encoding: pred_index 0 = A (Yes), 1 = B (No), 2 = C (Cannot determine / NVC).
"""

import csv
import json
from collections import Counter
from pathlib import Path

LETTERS = ["A(Yes)", "B(No)", "C(NVC)"]

GOLD_FILE = Path("data/processed/dataset_all_forms.jsonl")

MODEL_FILES = {
    "DeepSeek": Path("data/predictions/deepseek_8b_cot1024_sub64_predictions.jsonl"),
    "Mistral":  Path("data/predictions/mistral_7b_cot_sub256_predictions.jsonl"),
}

OUTPUT_DIR = Path("outputs/report")
ROWS_CSV = OUTPUT_DIR / "analyse_nvc_disagreement_rows.csv"
SUMMARY_CSV = OUTPUT_DIR / "analyse_nvc_summary.csv"


def load_jsonl_by_id(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def label(idx):
    return LETTERS[idx] if 0 <= idx < 3 else f"?({idx})"


def overall_distribution(preds, gold):
    """Prediction distribution across all conclusions, role=none only."""
    counter = Counter()
    total = 0
    for iid, p in preds.items():
        item = gold.get(iid)
        if item is None or item["role"] != "none":
            continue
        for _, pc in p["conclusions"].items():
            counter[pc.get("pred_index", -1)] += 1
            total += 1
    return counter, total


def disagreement_rows(preds_by_model, gold, allowed_ids):
    """Return one row per (item, conclusion) where classical != WCS gold."""
    rows = []
    for iid in sorted(allowed_ids):
        item = gold.get(iid)
        if item is None or item["role"] != "none":
            continue
        for code, gc in item["conclusions"].items():
            if gc["classical"] == gc["wcs"]:
                continue
            row = {
                "id": iid,
                "code": code,
                "classical": gc["classical"],
                "wcs": gc["wcs"],
            }
            for m, preds in preds_by_model.items():
                row[m] = preds[iid]["conclusions"].get(code, {}).get("pred_index", -1)
            rows.append(row)
    return rows


def main():
    gold = load_jsonl_by_id(GOLD_FILE)
    preds_by_model = {m: load_jsonl_by_id(p) for m, p in MODEL_FILES.items()}

    common_ids = set.intersection(*[set(p) for p in preds_by_model.values()])
    print(f"Common items across all models: {len(common_ids)}\n")

    # 1. Overall distribution
    for m, preds in preds_by_model.items():
        counter, total = overall_distribution(preds, gold)
        print(f"=== {m}: overall pred distribution (sub64, role=none) ===")
        for k in sorted(counter):
            pct = 100 * counter[k] / total if total else 0
            print(f"  {label(k):>10}: {counter[k]:>3} ({pct:.1f}%)")
        print()

    # 2. Disagreement items
    rows = disagreement_rows(preds_by_model, gold, common_ids)
    print(f"=== {len(rows)} disagreement items ===")
    header = f"{'item':>18} {'code':>4} {'clas':>7} {'wcs':>7}"
    for m in preds_by_model:
        header += f" {m[:8]:>8}"
    print(header)
    for r in rows:
        line = f"{r['id'][:18]:>18} {r['code']:>4} {label(r['classical']):>7} {label(r['wcs']):>7}"
        for m in preds_by_model:
            line += f" {label(r[m]):>8}"
        print(line)

    # 3. NVC-focused summary
    print("\n=== NVC-focused summary on disagreement items ===")
    n = len(rows)
    print(f"Total disagreement rows: {n}")
    print(f"  classical gold = NVC: {sum(1 for r in rows if r['classical'] == 2)}")
    print(f"  WCS       gold = NVC: {sum(1 for r in rows if r['wcs'] == 2)}")
    summary_rows = []
    for m in preds_by_model:
        n_nvc = sum(1 for r in rows if r[m] == 2)
        n_clas = sum(1 for r in rows if r[m] == r["classical"])
        n_wcs = sum(1 for r in rows if r[m] == r["wcs"])
        counter, total = overall_distribution(preds_by_model[m], gold)
        overall_nvc_pct = 100 * counter.get(2, 0) / total if total else 0.0
        print(
            f"  {m}: predicts NVC {n_nvc}/{n}, "
            f"matches classical {n_clas}/{n}, matches WCS {n_wcs}/{n}"
        )
        summary_rows.append({
            "model": m,
            "n_disagreement": n,
            "overall_nvc_pct": round(overall_nvc_pct, 2),
            "disagreement_nvc_pct": round(100 * n_nvc / n, 2) if n else 0.0,
            "classical_match_pct": round(100 * n_clas / n, 2) if n else 0.0,
            "wcs_match_pct": round(100 * n_wcs / n, 2) if n else 0.0,
        })

    # 4. Write CSVs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ROWS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "conclusion_code", "classical_gold", "wcs_gold"]
        for m in preds_by_model:
            fieldnames.append(f"{m}_pred")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {
                "id": r["id"],
                "conclusion_code": r["code"],
                "classical_gold": label(r["classical"]),
                "wcs_gold": label(r["wcs"]),
            }
            for m in preds_by_model:
                out[f"{m}_pred"] = label(r[m])
            w.writerow(out)
    print(f"\nWrote per-row table: {ROWS_CSV}")

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote summary:       {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
