"""Quick answer-distribution overview across all prediction files.

Usage:
    python3 scripts/summary.py

Reads every *_conversations.jsonl in data/predictions/ and prints one table:
name, n, A%, B%, C%, unparsed%.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter


def summarize(path: str) -> dict:
    letters: Counter = Counter()
    total = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            total += 1
            letters[r.get("pred_letter") or "∅"] += 1
    pct = lambda x: 100 * letters.get(x, 0) / total if total else 0.0
    return {
        "name": os.path.basename(path).replace("_conversations.jsonl", ""),
        "n": total,
        "A%": pct("A"),
        "B%": pct("B"),
        "C%": pct("C"),
        "∅%": pct("∅"),
    }


def main() -> None:
    files = sorted(glob.glob("data/predictions/*_conversations.jsonl"))
    if not files:
        print("no conversation files found under data/predictions/")
        return

    rows = [summarize(p) for p in files]

    name_w = max(len(r["name"]) for r in rows)
    header = f"{'name':<{name_w}}  {'n':>5}  {'A%':>5}  {'B%':>5}  {'C%':>5}  {'∅%':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<{name_w}}  "
            f"{r['n']:>5}  "
            f"{r['A%']:>5.1f}  {r['B%']:>5.1f}  {r['C%']:>5.1f}  {r['∅%']:>5.1f}"
        )

    # gold distribution for context
    print()
    print("Gold C-rate on sub256 dataset (for reference):")
    for gold_path in ["data/processed/dataset_sub256.jsonl", "data/processed/dataset_all_forms.jsonl"]:
        if not os.path.exists(gold_path):
            continue
        gold_c_classical = 0
        gold_c_wcs = 0
        total_conc = 0
        with open(gold_path) as f:
            for line in f:
                item = json.loads(line)
                for conc in item["conclusions"].values():
                    total_conc += 1
                    if conc.get("classical") == 2:
                        gold_c_classical += 1
                    if conc.get("wcs") == 2:
                        gold_c_wcs += 1
        print(
            f"  {os.path.basename(gold_path)}: "
            f"classical C = {100*gold_c_classical/total_conc:.1f}%,  "
            f"wcs C = {100*gold_c_wcs/total_conc:.1f}%  (n={total_conc})"
        )


if __name__ == "__main__":
    main()
