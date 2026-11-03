import json
from collections import Counter, defaultdict
from pathlib import Path

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
    "llama8b_vllm": "Llama-3.1-8B",
    "deepseek_llama_8b_vllm": "DS-Llama-8B",
    "deepseek_qwen_14b_vllm": "DS-Qwen-14B",
    "mistral7b_vllm": "Mistral-7B",
    "mistral24B_vllm": "Mistral-24B",
    "olmo2_13b_vllm": "OLMo-2-13B",
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
        r"\label{tab:alignment-by-syllogistic-form}",
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
    form_path = TABLES_DIR / "alignment_by_syllogistic_form.tex"
    type_path = TABLES_DIR / "alignment_by_conclusion_type.tex"
    write_form_table(stats, cases, form_path)
    write_conclusion_type_table(stats, cases, type_path)

    print(f"Gold disagreement prompts: {len(cases)}")
    print(f"Forms represented: {len({case['form'] for case in cases.values()})}")
    for model, model_stats in stats.items():
        total = sum(model_stats["by_type"][kind][outcome] for kind in CONCLUSION_TYPES for outcome in OUTCOMES)
        invalid = sum(model_stats["by_type"][kind]["invalid"] for kind in CONCLUSION_TYPES)
        print(f"{MODEL_LABELS[model]}: {model_stats['runs']} runs, {total} responses, {invalid} invalid")
    print(f"Wrote {form_path}")
    print(f"Wrote {type_path}")


if __name__ == "__main__":
    main()