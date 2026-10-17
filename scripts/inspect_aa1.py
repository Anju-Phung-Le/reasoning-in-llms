from pathlib import Path
from src.classical.classical import (
    all_models,
    all_satisfying_models,
    all_A_are_B,
    no_A_are_B,
    some_A_are_B,
    some_A_are_not_B,
    check_9_conclusions,
)

conclusion_names = {
    "Aac": "All A are C",
    "Eac": "No A are C",
    "Iac": "Some A are C",
    "Oac": "Some A are not C",
    "Aca": "All C are A",
    "Eca": "No C are A",
    "Ica": "Some C are A",
    "Oca": "Some C are not A",
}

conclusions = {
    "Aac": (all_A_are_B, "a", "c"),
    "Eac": (no_A_are_B, "a", "c"),
    "Iac": (some_A_are_B, "a", "c"),
    "Oac": (some_A_are_not_B, "a", "c"),
    "Aca": (all_A_are_B, "c", "a"),
    "Eca": (no_A_are_B, "c", "a"),
    "Ica": (some_A_are_B, "c", "a"),
    "Oca": (some_A_are_not_B, "c", "a"),
}

latex = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Classical evaluation of the eight conclusions for AA1.}",
    r"\label{tab:aa1-classical}",
    r"\begin{tabular}{llcl}",
    r"\toprule",
    r"Code & Candidate conclusion & True models & Gold label \\",
    r"\midrule",
]

# Get all models satisfying the two AA1 premises
valid_models = all_satisfying_models("AA1")

# Compute the Classical gold labels
gold = check_9_conclusions(valid_models)

labels = {
    0: "Yes",
    1: "No",
    2: "Cannot Determine",
}

for code, (fn, subj, pred) in conclusions.items():
    true_count = sum(
        fn(model, subj, pred) for model in valid_models
    )

    row = (
        f"{code} & {conclusion_names[code]} & "
        f"{true_count}/{len(valid_models)} & "
        f"{labels[gold[code]]} "
        + r"\\"
    )
    latex.append(row)

latex.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])

# Save the generated LaTeX table
output = Path("tables/aa1_classical.tex")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(latex) + "\n", encoding="utf-8")

print(f"\nLaTeX table saved to: {output}")

print("\nAA1 Classical Gold Labels")
print(f"{'Conclusion':<12} {'True models':<15} {'Gold label'}")
print("-" * 50)

for code, (fn, subj, pred) in conclusions.items():
    true_count = sum(
        fn(model, subj, pred) for model in valid_models
    )
    print(
        f"{code:<12} "
        f"{true_count}/{len(valid_models):<13} "
        f"{labels[gold[code]]}"
    )