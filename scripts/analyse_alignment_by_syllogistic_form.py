import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data/processed/dataset_sub256.jsonl"
PREDICTION_ROOT = ROOT / "data/predictions/vllm"
TABLES_DIR = ROOT / "tables"

MODEL_ORDER = [
    "llama8b_vllm",
    "deepseek_llama_8b_vllm",
    "deepseek_qwen_14b_vllm",
    "mistral7b_vllm",
    "mistral24B_vllm",
    "olmo2_13b_vllm",
]
MODEL_LABELS = {
    "llama8b_vllm": "Llama-8B",
    "deepseek_llama_8b_vllm": "DS-Llama-8B",
    "deepseek_qwen_14b_vllm": "DS-Qwen-14B",
    "mistral7b_vllm": "Mistral-7B",
    "mistral24B_vllm": "Mistral-24B",
    "olmo2_13b_vllm": "OLMo-13B",
}
OUTCOMES = ("classical_only", "wcs_only", "neither", "invalid")
CONCLUSION_TYPES = ("Universal (A/E)", "Particular (I/O)")


def load_disagreement_cases(path: Path):
    cases = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            for code, conclusion in row["conclusions"].items():
                classical = conclusion["classical"]
                wcs = conclusion["wcs"]
                if classical == wcs:
                    continue
                if code[0] in ("A", "E"):
                    conclusion_type = "Universal (A/E)"
                elif code[0] in ("I", "O"):
                    conclusion_type = "Particular (I/O)"
                else:
                    raise ValueError(f"Unknown conclusion code: {code}")
                cases[(row["id"], code)] = {
                    "form": row["form"],
                    "type": conclusion_type,
                    "classical": classical,
                    "wcs": wcs,
                }
    return cases


def classify_prediction(prediction, case):
    pred_index = prediction.get("pred_index")
    if pred_index not in (0, 1, 2):
        return "invalid"
    if pred_index == case["classical"]:
        return "classical_only"
    if pred_index == case["wcs"]:
        return "wcs_only"
    return "neither"


def analyze_model(model_dir: Path, cases):
    prediction_files = sorted(
        path
        for path in model_dir.glob("*sample*.jsonl")
        if "_conversations" not in path.name
    )
    if not prediction_files:
        return None

    by_form = defaultdict(Counter)
    by_type = defaultdict(Counter)
    expected_keys = set(cases)

    for prediction_file in prediction_files:
        seen_keys = set()
        with prediction_file.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                for code, prediction in row["conclusions"].items():
                    key = (row["id"], code)
                    case = cases.get(key)
                    if case is None:
                        continue
                    seen_keys.add(key)
                    outcome = classify_prediction(prediction, case)
                    by_form[case["form"]][outcome] += 1
                    by_type[case["type"]][outcome] += 1

        missing = expected_keys - seen_keys
        if missing:
            raise ValueError(
                f"{prediction_file.name} is missing {len(missing)} disagreement predictions"
            )

    return {
        "runs": len(prediction_files),
        "by_form": by_form,
        "by_type": by_type,
    }


def rate_delta(counts):
    total = sum(counts.values())
    if not total:
        return 0.0
    return 100.0 * (counts["wcs_only"] - counts["classical_only"]) / total


def rate(counts, outcome):
    total = sum(counts.values())
    return 100.0 * counts[outcome] / total if total else 0.0


def write_form_table(stats, cases, path: Path):
    forms = sorted({case["form"] for case in cases.values()})
    prompts_by_form = Counter(case["form"] for case in cases.values())
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Net WCS-versus-Classical alignment on disagreement questions pooled across all four role conditions (none, child, teen, adult), by syllogistic form. The 27 forms shown are those with disagreements among the 64 possible forms. Entries are percentage-point differences: WCS-only response rate minus Classical-only response rate.}",
        r"\label{tab:alignment-by-syllogistic-form-appendix}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Form & $n$ & Llama-8B & DS-Llama & DS-Qwen & Mistral-7B & Mistral-24B & OLMo-13B \\",
        r"\midrule",
    ]

    for form in forms:
        cells = [form, str(prompts_by_form[form])]
        for model in MODEL_ORDER:
            model_stats = stats.get(model)
            counts = model_stats["by_form"][form] if model_stats else Counter()
            cells.append(f"{rate_delta(counts):+.1f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{\linewidth}",
            r"\scriptsize $n$ is the number of unique disagreement prompts in that form pooled across all four roles. The 349 prompts comprise 69 none, 86 child, 102 teen, and 92 adult prompts; the neutral-role subset alone has $n=69$. Each prompt has 10 generated responses per model. The denominator for each difference includes WCS-only, Classical-only, neither, and invalid responses; positive values lean WCS, negative values lean Classical.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_form_heatmap(
    stats, cases, pdf_path: Path, tex_path: Path,
    model_order=None, model_labels=None, graphic_path=None, label_suffix="",
):
    model_order = MODEL_ORDER if model_order is None else model_order
    model_labels = MODEL_LABELS if model_labels is None else model_labels
    graphic_path = graphic_path or "tables/alignment_by_syllogistic_form_heatmap.pdf"
    forms = sorted({case["form"] for case in cases.values()})
    prompts_by_form = Counter(case["form"] for case in cases.values())
    values = [
        [
            rate_delta(stats[model]["by_form"][form]) if model in stats else 0.0
            for model in model_order
        ]
        for form in forms
    ]

    cmap = LinearSegmentedColormap.from_list(
        "classical_neutral_wcs",
        ["#3C784B", "#F2F1ED", "#3C5896"],
    )
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    figure_height = max(11.0, len(forms) * 0.40)
    figure_width = max(8.5, len(model_order) * 1.4)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    image = axis.imshow(values, aspect="auto", cmap=cmap, norm=norm)

    axis.set_xticks(range(len(model_order)))
    axis.set_xticklabels([model_labels[model] for model in model_order], rotation=0, ha="center", fontsize=9)
    axis.set_yticks(range(len(forms)))
    axis.set_yticklabels([f"{form} (n={prompts_by_form[form]})" for form in forms], fontsize=10)

    axis.set_xticks([index - 0.5 for index in range(len(model_order) + 1)], minor=True)
    axis.set_yticks([index - 0.5 for index in range(len(forms) + 1)], minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            red, green, blue, _ = cmap(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "white" if luminance < 0.48 else "#20252A"
            axis.text(column_index, row_index, f"{value:+.1f}", ha="center", va="center", color=text_color, fontsize=10)

    colorbar = figure.colorbar(image, ax=axis, ticks=[-100, -50, 0, 50, 100], pad=0.025)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.ax.set_yticklabels(["-100 Classical", "-50", "0 Neutral", "+50", "+100 WCS"])
    axis.legend(
        handles=[
            Patch(facecolor="#3C784B", label="Negative: Classical-leaning"),
            Patch(facecolor="#F2F1ED", edgecolor="#A0A0A0", label="Near zero: little relative preference"),
            Patch(facecolor="#3C5896", label="Positive: WCS-leaning"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(figure)

    tex_path.write_text(
        "\n".join([
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=\linewidth,height=0.94\textheight,keepaspectratio]{" + graphic_path + "}",
            r"\caption{WCS-only minus Classical-only response-rate difference by syllogistic form. Green indicates a more Classical-leaning difference; blue indicates a more WCS-leaning difference; pale colors indicate little relative preference. Each row gives the number of unique disagreement prompts pooled across all four role conditions. The shared color scale runs from $-100$ to $+100$ percentage points and is centered at zero.}",
            r"\label{tab:alignment-by-syllogistic-form" + label_suffix + "}",
            r"\end{figure}",
        ]) + "\n",
        encoding="utf-8",
    )


def write_conclusion_type_table(stats, cases, path: Path):
    prompts_by_type = Counter(case["type"] for case in cases.values())
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Model responses on Classical/WCS disagreement questions pooled across all four role conditions (none, child, teen, adult), split by conclusion type.}",
        r"\label{tab:alignment-by-conclusion-type}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Type & $n$ & \shortstack{Classical\\only} & \shortstack{WCS\\only} & Neither & Invalid & $\Delta_{W-C}$ \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        model_stats = stats.get(model)
        if model_stats is None:
            continue
        for conclusion_type in CONCLUSION_TYPES:
            counts = model_stats["by_type"][conclusion_type]
            values = [rate(counts, outcome) for outcome in OUTCOMES]
            cells = [
                MODEL_LABELS[model],
                conclusion_type,
                str(prompts_by_type[conclusion_type]),
                *(f"{value:.1f}\\%" for value in values),
                f"{rate_delta(counts):+.1f} pp",
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{\linewidth}",
            r"\scriptsize Universal conclusions begin with A or E; particular conclusions begin with I or O. $n$ counts unique disagreement prompts pooled across all four roles, each evaluated with 10 generations. Percentages use all generated responses, including invalid outputs; $\Delta_{W-C}$ is WCS-only minus Classical-only in percentage points.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cases = load_disagreement_cases(GOLD_PATH)
    stats = {}
    for model in MODEL_ORDER:
        model_dir = PREDICTION_ROOT / model
        model_stats = analyze_model(model_dir, cases)
        if model_stats is not None:
            stats[model] = model_stats

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    form_path = TABLES_DIR / "alignment_by_syllogistic_form_appendix.tex"
    figure_tex_path = TABLES_DIR / "alignment_by_syllogistic_form.tex"
    figure_pdf_path = TABLES_DIR / "alignment_by_syllogistic_form_heatmap.pdf"
    type_path = TABLES_DIR / "alignment_by_conclusion_type.tex"
    write_form_table(stats, cases, form_path)
    write_form_heatmap(stats, cases, figure_pdf_path, figure_tex_path)
    write_conclusion_type_table(stats, cases, type_path)

    print(f"Gold disagreement prompts: {len(cases)}")
    print(f"Forms represented: {len({case['form'] for case in cases.values()})}")
    for model, model_stats in stats.items():
        total = sum(model_stats["by_type"][kind][outcome] for kind in CONCLUSION_TYPES for outcome in OUTCOMES)
        invalid = sum(model_stats["by_type"][kind]["invalid"] for kind in CONCLUSION_TYPES)
        print(f"{MODEL_LABELS[model]}: {model_stats['runs']} runs, {total} responses, {invalid} invalid")
    print(f"Wrote {form_path}")
    print(f"Wrote {figure_tex_path}")
    print(f"Wrote {figure_pdf_path}")
    print(f"Wrote {type_path}")


if __name__ == "__main__":
    main()