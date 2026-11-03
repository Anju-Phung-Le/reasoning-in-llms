import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data/processed/dataset_sub256.jsonl"
PRED_ROOT = ROOT / "data/predictions/vllm"
TABLES_DIR = ROOT / "tables"

ROLE_ORDER = ["none", "child", "teen", "adult"]
DOMAIN_ORDER = ["animals", "nature", "objects", "people"]
MODEL_LABELS = {
    "mistral7b_vllm": "Mistral-7B",
    "llama8b_vllm": "Llama-3.1-8B",
    "olmo2_13b_vllm": "OLMo-2-13B",
    "deepseek_llama_8b_vllm": "DeepSeek-R1-Distill-Llama-8B",
    "deepseek_qwen_14b_vllm": "DeepSeek-R1-Distill-Qwen-14B",
    "mistral24B_vllm": "Mistral-Small-24B",
}


def load_gold(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_predictions_by_id(path: Path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["id"]] = row
    return out


def summarize(values):
    return statistics.fmean(values) if values else 0.0


def metric_by_group(rows, pred_rows, metric_name):
    by_role = {r: [] for r in ROLE_ORDER}
    by_domain = {d: [] for d in DOMAIN_ORDER}

    for row in rows:
        role = row["role"]
        domain = row["domain"]
        pred = pred_rows.get(row["id"])
        if pred is None:
            continue

        total = 0
        matches = 0

        for code, conc in row["conclusions"].items():
            total += 1
            gold_classical = conc["classical"]
            gold_wcs = conc["wcs"]
            pred_idx = pred["conclusions"].get(code, {}).get("pred_index", -1)

            if metric_name == "classical":
                if pred_idx == gold_classical:
                    matches += 1
            elif metric_name == "wcs":
                if pred_idx == gold_wcs:
                    matches += 1
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

        by_role.setdefault(role, []).append(100.0 * matches / total)
        by_domain.setdefault(domain, []).append(100.0 * matches / total)

    return {"role": by_role, "domain": by_domain}


def compute_model_stats(gold_rows):
    model_stats = {}

    for model_dir in sorted(PRED_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue

        files = sorted(model_dir.glob("*sample*.jsonl"))
        files = [f for f in files if "_conversations" not in f.name]
        if not files:
            continue

        role_classical = {r: [] for r in ROLE_ORDER}
        role_wcs = {r: [] for r in ROLE_ORDER}
        domain_classical = {d: [] for d in DOMAIN_ORDER}
        domain_wcs = {d: [] for d in DOMAIN_ORDER}

        for pred_file in files:
            pred_rows = load_predictions_by_id(pred_file)
            full_classical = metric_by_group(gold_rows, pred_rows, "classical")
            full_wcs = metric_by_group(gold_rows, pred_rows, "wcs")

            for role in ROLE_ORDER:
                role_classical[role].extend(full_classical["role"].get(role, []))
                role_wcs[role].extend(full_wcs["role"].get(role, []))
            for domain in DOMAIN_ORDER:
                domain_classical[domain].extend(full_classical["domain"].get(domain, []))
                domain_wcs[domain].extend(full_wcs["domain"].get(domain, []))

        model_stats[model_dir.name] = {
            "role_classical": {r: summarize(role_classical[r]) for r in ROLE_ORDER},
            "role_wcs": {r: summarize(role_wcs[r]) for r in ROLE_ORDER},
            "domain_classical": {d: summarize(domain_classical[d]) for d in DOMAIN_ORDER},
            "domain_wcs": {d: summarize(domain_wcs[d]) for d in DOMAIN_ORDER},
        }

    return model_stats


def intensity_for_value(value, min_val=0.0, max_val=100.0, base=8, top=90):
    if max_val <= min_val:
        return base
    frac = (value - min_val) / (max_val - min_val)
    return int(round(base + frac * (top - base)))


def latex_row(model_name, values, columns, color_name):
    cells = [model_name]
    max_val = max(values.values()) if values else 0.0
    min_val = min(values.values()) if values else 0.0
    for key in columns:
        v = values.get(key, 0.0)
        intensity = intensity_for_value(v, min_val, max_val)
        cells.append(f"\\cellcolor{{{color_name}!{intensity}}}{{{v:.2f}}}")
    return " & ".join(cells) + r" \\" + "\n"


def write_role_heatmap(model_stats, path: Path):
    lines = [
        r"\begin{figure}[H]",
        r"\centering",
        r"\caption{Agreement rates by role condition for the vLLM sub256 sample under Classical Logic and WCS. Darker cells indicate higher agreement.}",
        r"\label{fig:role-heatmap-vllm}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\multicolumn{5}{c}{\textbf{Classical Logic}} \\",
        r"\toprule",
        r"Model & None & Child & Teen & Adult \\",
        r"\midrule",
    ]

    for key in sorted(model_stats):
        model_name = MODEL_LABELS.get(key, key)
        vals = {role: model_stats[key]["role_classical"].get(role, 0.0) for role in ROLE_ORDER}
        lines.append(latex_row(model_name, vals, ROLE_ORDER, "green"))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1.0em}",
        r"\begin{tabular}{lcccc}",
        r"\multicolumn{5}{c}{\textbf{WCS}} \\",
        r"\toprule",
        r"Model & None & Child & Teen & Adult \\",
        r"\midrule",
    ]

    for key in sorted(model_stats):
        model_name = MODEL_LABELS.get(key, key)
        vals = model_stats[key]["role_wcs"]
        lines.append(latex_row(model_name, vals, ROLE_ORDER, "blue"))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{figure}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_domain_heatmap(model_stats, path: Path):
    lines = [
        r"\begin{figure}[H]",
        r"\centering",
        r"\caption{Agreement rates by lexical domain for the vLLM sub256 sample under Classical Logic and WCS. Darker cells indicate higher agreement.}",
        r"\label{fig:domain-heatmap-vllm}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\multicolumn{5}{c}{\textbf{Classical Logic}} \\",
        r"\toprule",
        r"Model & Animals & Nature & Objects & People \\",
        r"\midrule",
    ]

    for key in sorted(model_stats):
        model_name = MODEL_LABELS.get(key, key)
        vals = model_stats[key]["domain_classical"]
        lines.append(latex_row(model_name, vals, DOMAIN_ORDER, "green"))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1.0em}",
        r"\begin{tabular}{lcccc}",
        r"\multicolumn{5}{c}{\textbf{WCS}} \\",
        r"\toprule",
        r"Model & Animals & Nature & Objects & People \\",
        r"\midrule",
    ]

    for key in sorted(model_stats):
        model_name = MODEL_LABELS.get(key, key)
        vals = model_stats[key]["domain_wcs"]
        lines.append(latex_row(model_name, vals, DOMAIN_ORDER, "blue"))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{figure}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    gold_rows = load_gold(GOLD_PATH)
    model_stats = compute_model_stats(gold_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    write_role_heatmap(model_stats, TABLES_DIR / "role_alignment_heatmap_vllm.tex")
    write_domain_heatmap(model_stats, TABLES_DIR / "domain_alignment_heatmap_vllm.tex")
    print(f"Wrote {TABLES_DIR / 'role_alignment_heatmap_vllm.tex'}")
    print(f"Wrote {TABLES_DIR / 'domain_alignment_heatmap_vllm.tex'}")


if __name__ == "__main__":
    main()
