
from pathlib import Path
import json

from src.wcs.encoders import build_program_for_form
from src.wcs.leastmodel import least_model
from src.wcs.gold_wcs import check_9_conclusions
from src.wcs.tv import TV


# The eight candidate conclusions
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

labels = {
    TV.TRUE: "Yes",
    TV.FALSE: "No",
    TV.UNKNOWN: "Cannot Determine",
}

gold_codes = {
    TV.TRUE: 0,
    TV.FALSE: 1,
    TV.UNKNOWN: 2,
}

# Step 1: Build the program for AA1
P, domain = build_program_for_form("AA1")

# Step 2: Compute the least model
I = least_model(P, domain=domain)

# Step 3: Evaluate the eight conclusions
results = check_9_conclusions(I, a="a", c="c")


# Print the least-model interpretation
print("\nAA1 WCS Least Model")
print(f"{'Object':<10} {'A':<10} {'B':<10} {'C'}")
print("-" * 45)

for obj in domain:
    a = I.get("a", obj).name.title()
    b = I.get("b", obj).name.title()
    c = I.get("c", obj).name.title()
    print(f"{obj:<10} {a:<10} {b:<10} {c}")


# Print the eight conclusions
print("\nAA1 WCS Gold Labels")
print(f"{'Code':<8} {'Truth value':<15} {'Gold label'}")
print("-" * 50)

for code in conclusion_names:
    tv = results[code]
    print(f"{code:<8} {tv.name.title():<15} {labels[tv]}")


# Step 4: Generate the LaTeX table
latex = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{WCS evaluation of the eight conclusions for AA1.}",
    r"\label{tab:aa1-wcs}",
    r"\begin{tabular}{llcl}",
    r"\toprule",
    r"Code & Candidate conclusion & Truth value & Gold label \\",
    r"\midrule",
]

for code, name in conclusion_names.items():
    tv = results[code]

    row = (
        f"{code} & {name} & "
        f"{tv.name.title()} & {labels[tv]} "
        + r"\\"
    )
    latex.append(row)

latex.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])

output = Path("tables/aa1_wcs.tex")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(latex) + "\n", encoding="utf-8")

print(f"\nLaTeX table saved to: {output}")


# Step 5: Compare with the existing benchmark gold labels
dataset = Path("data/processed/dataset_all_forms.jsonl")

if dataset.exists():
    with dataset.open(encoding="utf-8") as f:
        record = next(
            (
                json.loads(line)
                for line in f
                if json.loads(line)["form"] == "AA1"
                and json.loads(line)["role"] == "none"
            ),
            None,
        )

    if record is not None:
        mismatches = []

        for code in conclusion_names:
            computed = gold_codes[results[code]]
            stored = record["conclusions"][code]["wcs"]

            if computed != stored:
                mismatches.append((code, computed, stored))

        if not mismatches:
            print("Verification: all 8 labels match the dataset.")
        else:
            print("WARNING: differences found:", mismatches)
