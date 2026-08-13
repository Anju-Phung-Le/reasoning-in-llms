"""
Diagnostic-item lean analysis.

For every prediction file in data/predictions/, restrict to the items where
the classical gold and the WCS gold disagree, then report the percentage of
model answers that match each gold (and the percentage that match neither).

This directly answers: on the items where the two reference systems make
different predictions, which side does the model lean towards?

Usage
-----
    # Report for all prediction files, all personas together:
    python3 scripts/diagnostic_lean.py

    # Break down by persona as well:
    python3 scripts/diagnostic_lean.py --by-persona

    # Write CSV output alongside the console table:
    python3 scripts/diagnostic_lean.py --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "data" / "predictions"
GOLD_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "report"

# Gold labels are stored as ints: 0=A (Yes), 1=B (No), 2=C (Cannot determine).
# Predictions store the same convention in `pred_index`; -1 means unparsable.
LETTERS = ["A", "B", "C"]


def _gold_file_for(pred_name: str) -> Path:
    """Match a predictions filename to its gold dataset."""
    if "_all_forms" in pred_name:
        return GOLD_DIR / "dataset_all_forms.jsonl"
    if "_sub256" in pred_name:
        return GOLD_DIR / "dataset_sub256.jsonl"
    if "_sub64" in pred_name:
        return GOLD_DIR / "dataset_sub64.jsonl"
    raise ValueError(f"Cannot infer gold file for {pred_name}")


def _classify(pred_name: str) -> tuple[str, str]:
    """Return (model, regime) tuple from a predictions filename."""
    m = re.match(r"(deepseek_8b|mistral_7b|flan_t5_[a-z0-9]+)_", pred_name)
    model = m.group(1) if m else pred_name
    if "_cot1024_sub64_" in pred_name:
        regime = "CoT-1024 sub64"
    elif "_cot_sub256_" in pred_name:
        regime = "CoT-384 sub256"
    else:
        regime = "forced-choice"
    return model, regime


def _load_gold(path: Path) -> dict[tuple[str, str, str], tuple[int, int]]:
    """
    Return {(item_id, role, conclusion_code): (classical_gold, wcs_gold)}.

    Persona (`role`) is included in the key because the same syllogism appears
    once per persona in the dataset with a different id suffix.
    """
    out: dict[tuple[str, str, str], tuple[int, int]] = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            role = item.get("role", "")
            for code, conc in item["conclusions"].items():
                out[(item["id"], role, code)] = (conc["classical"], conc["wcs"])
    return out


def _load_preds(path: Path) -> dict[tuple[str, str, str], int]:
    """Return {(item_id, role, conclusion_code): pred_index}."""
    out: dict[tuple[str, str, str], int] = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            role = item.get("role", "")
            for code, conc in item["conclusions"].items():
                out[(item["id"], role, code)] = conc.get("pred_index", -1)
    return out


def lean(
    gold: dict[tuple[str, str, str], tuple[int, int]],
    preds: dict[tuple[str, str, str], int],
    role_filter: str | None = None,
) -> tuple[dict[str, float], int]:
    """
    Compute lean percentages on diagnostic items.

    Returns
    -------
    ({classical, wcs, neither, unparsed}, n_diagnostic_items)
    """
    counts: Counter[str] = Counter()
    total = 0
    for key, (gc, gw) in gold.items():
        if role_filter is not None and key[1] != role_filter:
            continue
        if gc == gw:  # agreement item -> not diagnostic
            continue
        p = preds.get(key, -1)
        total += 1
        if p == -1:
            counts["unparsed"] += 1
        elif p == gc:
            counts["classical"] += 1
        elif p == gw:
            counts["wcs"] += 1
        else:
            counts["neither"] += 1
    if total == 0:
        return {}, 0
    return {k: 100 * v / total for k, v in counts.items()}, total


def _rows_for_file(pred_path: Path, by_persona: bool) -> list[dict]:
    gold_path = _gold_file_for(pred_path.name)
    gold = _load_gold(gold_path)
    preds = _load_preds(pred_path)
    model, regime = _classify(pred_path.name)

    if by_persona:
        roles = sorted({key[1] for key in gold.keys()})
        rows = []
        for role in roles:
            pct, n = lean(gold, preds, role_filter=role)
            if n == 0:
                continue
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "role": role,
                    "n_diagnostic": n,
                    "pct_classical": round(pct.get("classical", 0.0), 1),
                    "pct_wcs": round(pct.get("wcs", 0.0), 1),
                    "pct_neither": round(pct.get("neither", 0.0), 1),
                    "pct_unparsed": round(pct.get("unparsed", 0.0), 1),
                }
            )
        return rows

    pct, n = lean(gold, preds)
    return [
        {
            "model": model,
            "regime": regime,
            "role": "all",
            "n_diagnostic": n,
            "pct_classical": round(pct.get("classical", 0.0), 1),
            "pct_wcs": round(pct.get("wcs", 0.0), 1),
            "pct_neither": round(pct.get("neither", 0.0), 1),
            "pct_unparsed": round(pct.get("unparsed", 0.0), 1),
        }
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-persona", action="store_true",
                    help="Break down by persona (adult/child/teen/none).")
    ap.add_argument("--csv", action="store_true",
                    help="Also write CSV to outputs/report/diagnostic_lean.csv")
    args = ap.parse_args()

    pred_files = sorted(
        p for p in PRED_DIR.glob("*_predictions.jsonl")
        if "_conversations" not in p.name
    )

    all_rows: list[dict] = []
    for pf in pred_files:
        all_rows.extend(_rows_for_file(pf, args.by_persona))

    # Sort: model, then regime, then role
    all_rows.sort(key=lambda r: (r["model"], r["regime"], r["role"]))

    # Console table
    print()
    print("Diagnostic-item lean (items where classical gold != WCS gold)")
    print("=" * 88)
    hdr = f"{'model':<18s} {'regime':<16s} {'role':<6s} " \
          f"{'n':>5s} {'clas%':>6s} {'wcs%':>6s} {'nei%':>6s} {'unp%':>6s}"
    print(hdr)
    print("-" * 88)
    last_key = None
    for r in all_rows:
        key = (r["model"], r["regime"])
        if last_key is not None and key != last_key:
            print()
        print(
            f"{r['model']:<18s} {r['regime']:<16s} {r['role']:<6s} "
            f"{r['n_diagnostic']:>5d} "
            f"{r['pct_classical']:>6.1f} {r['pct_wcs']:>6.1f} "
            f"{r['pct_neither']:>6.1f} {r['pct_unparsed']:>6.1f}"
        )
        last_key = key

    if args.csv:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = OUT_DIR / "diagnostic_lean.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n-> wrote {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
