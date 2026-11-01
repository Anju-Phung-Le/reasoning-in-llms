#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


LABEL_NAMES = {
    0: "Yes",
    1: "No",
    2: "Cannot determine",
}

LABEL_ORDER = (0, 1, 2)


def iter_conclusion_values(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for conclusion in rec.get("conclusions", {}).values():
                yield conclusion.get("classical"), conclusion.get("wcs")


def summarize_dataset(path: Path):
    counts = {
        "classical": defaultdict(int),
        "wcs": defaultdict(int),
    }
    total = {"classical": 0, "wcs": 0}

    for classical_val, wcs_val in iter_conclusion_values(path):
        for method, value in (("classical", classical_val), ("wcs", wcs_val)):
            if value is None:
                continue
            counts[method][LABEL_NAMES.get(value, str(value))] += 1
            total[method] += 1

    return counts, total


def summarize_agreement(path: Path):
    matrix = {c: {w: 0 for w in LABEL_ORDER} for c in LABEL_ORDER}
    total = 0

    for classical_val, wcs_val in iter_conclusion_values(path):
        if classical_val not in LABEL_NAMES or wcs_val not in LABEL_NAMES:
            continue
        matrix[classical_val][wcs_val] += 1
        total += 1

    return matrix, total


def latex_table(counts, total):
    rows = []
    for method in ("classical", "wcs"):
        method_label = "Classical" if method == "classical" else "WCS"
        yes = counts[method]["Yes"]
        no = counts[method]["No"]
        cannot = counts[method]["Cannot determine"]
        total_n = total[method]
        rows.append(
            f"{method_label} & {yes} ({yes / total_n * 100:.1f}\\%) & "
            f"{no} ({no / total_n * 100:.1f}\\%) & "
            f"{cannot} ({cannot / total_n * 100:.1f}\\%) \\\\"
        )

    return (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Method & Yes & No & Cannot determine \\\\ \n"
        "\\midrule\n"
        f"{chr(10).join(rows)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def latex_agreement_table(matrix, total):
    header = "Classical / WCS"
    lines = []

    for c in LABEL_ORDER:
        row_name = LABEL_NAMES[c]
        row_total = sum(matrix[c][w] for w in LABEL_ORDER)
        cells = []
        for w in LABEL_ORDER:
            n = matrix[c][w]
            pct = (100.0 * n / row_total) if row_total else 0.0
            cells.append(f"{n} ({pct:.1f}\\%)")
        lines.append(f"{row_name} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

    return (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        f"{header} & Yes & No & Cannot determine \\\\ \n"
        "\\midrule\n"
        f"{chr(10).join(lines)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"% Total evaluated conclusions: {total}\n"
    )


def latex_agreement_heatmap_table(matrix, total):
    header = "Classical / WCS"
    lines = []

    # Use global percentage for shading intensity so cells are comparable.
    for c in LABEL_ORDER:
        row_name = LABEL_NAMES[c]
        cells = []
        for w in LABEL_ORDER:
            n = matrix[c][w]
            pct_global = (100.0 * n / total) if total else 0.0
            shade = max(0, min(100, int(round(pct_global))))
            cells.append(f"\\cellcolor{{blue!{shade}}}{n} ({pct_global:.1f}\\%)")
        lines.append(f"{row_name} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

    return (
        "% Requires \\usepackage[table]{xcolor} and \\usepackage{booktabs}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        f"{header} & Yes & No & Cannot determine \\\\ \n"
        "\\midrule\n"
        f"{chr(10).join(lines)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"% Total evaluated conclusions: {total}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Summarize classical and WCS label distributions for a dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/dataset_all_forms.jsonl"),
        help="Path to the JSONL dataset with classical and wcs conclusion labels.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tables/label_distribution.tex"),
        help="Output path for the LaTeX table.",
    )
    parser.add_argument(
        "--agreement-out",
        type=Path,
        default=Path("tables/classical_wcs_agreement.tex"),
        help="Output path for the LaTeX agreement matrix table.",
    )
    parser.add_argument(
        "--agreement-heatmap-out",
        type=Path,
        default=Path("tables/classical_wcs_agreement_heatmap.tex"),
        help="Output path for the LaTeX agreement matrix heatmap table.",
    )
    args = parser.parse_args()

    counts, total = summarize_dataset(args.dataset)
    latex = latex_table(counts, total)
    matrix, matrix_total = summarize_agreement(args.dataset)
    agreement_latex = latex_agreement_table(matrix, matrix_total)
    agreement_heatmap_latex = latex_agreement_heatmap_table(matrix, matrix_total)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex, encoding="utf-8")
    args.agreement_out.parent.mkdir(parents=True, exist_ok=True)
    args.agreement_out.write_text(agreement_latex, encoding="utf-8")
    args.agreement_heatmap_out.parent.mkdir(parents=True, exist_ok=True)
    args.agreement_heatmap_out.write_text(agreement_heatmap_latex, encoding="utf-8")

    print(f"Dataset: {args.dataset}")
    print(f"Wrote LaTeX table to: {args.out}")
    print(f"Wrote LaTeX agreement table to: {args.agreement_out}")
    print(f"Wrote LaTeX agreement heatmap table to: {args.agreement_heatmap_out}")
    print()
    print("Table 1: Distribution")
    print(latex)
    print("Table 2: Agreement")
    print(agreement_latex)
    print("Table 2b: Agreement Heatmap")
    print(agreement_heatmap_latex)


if __name__ == "__main__":
    main()
