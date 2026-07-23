"""Comprehensive local report across every model/regime you have.

Usage:
    python3 scripts/report.py            # print to terminal
    python3 scripts/report.py --csv      # ALSO write CSVs to outputs/report/

Prints/writes:
    1. answer_distribution.csv  — A/B/C/unparsed % per prediction file
    2. per_persona.csv          — n, c_usage, acc, deltas per (model, regime, role)
    3. baselines.csv            — always-C rate per dataset
    4. persona_effect.csv       — child vs none accuracy delta per (model, regime)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

OUT_DIR = "outputs/report"


# ─── PART 1: answer distributions from *_conversations.jsonl ────────────────
def answer_dist(write_csv: bool = False) -> None:
    print("=" * 78)
    print("PART 1 — Answer distribution (from *_conversations.jsonl)")
    print("=" * 78)
    files = sorted(glob.glob("data/predictions/*_conversations.jsonl"))
    if not files:
        print("  (no conversation files found)")
        return

    rows_out: list[dict] = []
    name_w = max(len(os.path.basename(p).replace("_conversations.jsonl", "")) for p in files)
    header = f"  {'file':<{name_w}}  {'n':>5}  {'A%':>5}  {'B%':>5}  {'C%':>5}  {'∅%':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for path in files:
        letters: Counter = Counter()
        total = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                total += 1
                letters[r.get("pred_letter") or "∅"] += 1
        pct = lambda x: 100 * letters.get(x, 0) / total if total else 0.0
        name = os.path.basename(path).replace("_conversations.jsonl", "")
        print(
            f"  {name:<{name_w}}  {total:>5}  "
            f"{pct('A'):>5.1f}  {pct('B'):>5.1f}  {pct('C'):>5.1f}  {pct('∅'):>5.1f}"
        )
        rows_out.append({
            "file": name, "n": total,
            "A_pct": round(pct("A"), 2), "B_pct": round(pct("B"), 2),
            "C_pct": round(pct("C"), 2), "unparsed_pct": round(pct("∅"), 2),
        })
    if write_csv and rows_out:
        _write_csv("answer_distribution.csv", rows_out)
    print()


# ─── PART 2: eval CSVs (forced-choice vs CoT per model) ─────────────────────
def _parse_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _classify(fname: str) -> tuple[str, str]:
    """Return (model, regime) tuple from CSV filename."""
    m = re.match(r"(deepseek_8b|mistral_7b|flan_t5_[a-z0-9]+)_", fname)
    model = m.group(1) if m else fname
    if "_cot1024_sub64_" in fname:
        regime = "CoT-1024 sub64"
    elif "_cot_sub256_" in fname:
        regime = "CoT-384 sub256"
    else:
        regime = "forced-choice"
    return model, regime


def eval_summary(write_csv: bool = False) -> None:
    print("=" * 78)
    print("PART 2 — Per-persona accuracy (from outputs/*.csv)")
    print("=" * 78)
    csv_files = sorted(glob.glob("outputs/*_outputs_log.csv"))
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for p in csv_files:
        model, regime = _classify(os.path.basename(p))
        grouped[model][regime] = _parse_csv(p)

    rows_out: list[dict] = []
    for model, regimes in grouped.items():
        print(f"\n  ── {model} ──")
        header = f"    {'regime':<14}  {'role':<6}  {'n':>5}  {'c_use%':>7}  {'acc_c%':>7}  {'acc_w%':>7}  {'Δwcs':>6}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for regime in ("forced-choice", "CoT-384 sub256", "CoT-1024 sub64"):
            if regime not in regimes:
                continue
            for row in regimes[regime]:
                c_use = row.get("c_usage", "")
                c_use_str = f"{100*float(c_use):>6.1f}" if c_use else "   n/a"
                acc_c = 100 * float(row["acc_classical"])
                acc_w = 100 * float(row["acc_wcs"])
                delta = float(row.get("delta_acc_wcs_minus_classical", row.get("delta_wcs_minus_classical", 0)))
                print(
                    f"    {regime:<14}  {row['role']:<6}  {row['n']:>5}  "
                    f"{c_use_str:>7}  {acc_c:>6.1f}  {acc_w:>6.1f}  {100*delta:>+5.1f}"
                )
                rows_out.append({
                    "model": model, "regime": regime, "role": row["role"], "n": row["n"],
                    "c_usage_pct": round(100 * float(c_use), 2) if c_use else "",
                    "acc_classical_pct": round(acc_c, 2),
                    "acc_wcs_pct": round(acc_w, 2),
                    "delta_wcs_minus_classical_pct": round(100 * delta, 2),
                })
    if write_csv and rows_out:
        _write_csv("per_persona.csv", rows_out)
    print()


# ─── PART 3: baselines for context ──────────────────────────────────────────
def baselines(write_csv: bool = False) -> None:
    print("=" * 78)
    print("PART 3 — Baselines (for context)")
    print("=" * 78)
    print("  Random uniform (⅓ A, ⅓ B, ⅓ C):             ~33.3% accuracy")
    rows_out: list[dict] = [
        {"baseline": "random_uniform", "dataset": "-", "classical_pct": 33.3, "wcs_pct": 33.3, "n": ""},
    ]
    for gold_path in ("data/processed/dataset_sub256.jsonl", "data/processed/dataset_all_forms.jsonl"):
        if not os.path.exists(gold_path):
            continue
        n_c, n_c_w, n = 0, 0, 0
        with open(gold_path) as f:
            for line in f:
                item = json.loads(line)
                for conc in item["conclusions"].values():
                    n += 1
                    n_c += 1 if conc.get("classical") == 2 else 0
                    n_c_w += 1 if conc.get("wcs") == 2 else 0
        name = os.path.basename(gold_path)
        print(f"  Always-'C' on {name:<28}  "
              f"classical={100*n_c/n:5.1f}%  wcs={100*n_c_w/n:5.1f}%  (n={n})")
        rows_out.append({
            "baseline": "always_C", "dataset": name,
            "classical_pct": round(100 * n_c / n, 2),
            "wcs_pct": round(100 * n_c_w / n, 2),
            "n": n,
        })
    if write_csv and rows_out:
        _write_csv("baselines.csv", rows_out)
    print()


# ─── PART 4: persona effect (child vs no-persona) ───────────────────────────
def persona_effect(write_csv: bool = False) -> None:
    print("=" * 78)
    print("PART 4 — Persona effect (does 'child' outperform 'none'?)")
    print("=" * 78)
    csv_files = sorted(glob.glob("outputs/*_outputs_log.csv"))
    rows_out: list[dict] = []
    for p in csv_files:
        rows = {r["role"]: r for r in _parse_csv(p)}
        if "child" not in rows or "none" not in rows:
            continue
        model, regime = _classify(os.path.basename(p))
        child_acc = 100 * float(rows["child"]["acc_classical"])
        none_acc = 100 * float(rows["none"]["acc_classical"])
        delta = child_acc - none_acc
        child_c = rows["child"].get("c_usage")
        none_c = rows["none"].get("c_usage")
        c_line = ""
        if child_c and none_c:
            c_line = f"  |  c_usage:  child={100*float(child_c):.1f}%  none={100*float(none_c):.1f}%"
        print(f"  {model:<15} {regime:<14}  child={child_acc:5.1f}%  none={none_acc:5.1f}%  Δ={delta:+5.1f}pp{c_line}")
        rows_out.append({
            "model": model, "regime": regime,
            "child_acc_pct": round(child_acc, 2),
            "none_acc_pct": round(none_acc, 2),
            "delta_pp": round(delta, 2),
            "child_c_usage_pct": round(100 * float(child_c), 2) if child_c else "",
            "none_c_usage_pct": round(100 * float(none_c), 2) if none_c else "",
        })
    if write_csv and rows_out:
        _write_csv("persona_effect.csv", rows_out)
    print()


# ─── CSV writer helper ──────────────────────────────────────────────────────
def _write_csv(name: str, rows: list[dict]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  → wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="also write CSV files to outputs/report/")
    args = ap.parse_args()
    answer_dist(args.csv)
    eval_summary(args.csv)
    baselines(args.csv)
    persona_effect(args.csv)
