import json, random, uuid
from pathlib import Path
from .wcs.gold_wcs import check_9_conclusions
from .wcs.encoders import build_program_for_form
from .wcs.leastmodel import least_model
from .wcs.tv import TV
from .classical.classical import compute_classical_label


# Natural-language patterns for premise moods
_MOOD_PREMISE = {
    "A": "All {S} are {P}.",
    "E": "No {S} are {P}.",
    "I": "Some {S} are {P}.",
    "O": "Some {S} are not {P}.",
}

# Figure determines how A (end term), B (middle term), C (end term)
# map to subject/predicate in each premise.
_FIGURE_SP = {
    1: (("{A}", "{B}"), ("{B}", "{C}")),
    2: (("{B}", "{A}"), ("{C}", "{B}")),
    3: (("{A}", "{B}"), ("{C}", "{B}")),
    4: (("{B}", "{A}"), ("{B}", "{C}")),
}

# All 8 possible conclusion questions
_CONCLUSION_Q = {
    "Aac": "Are all {A} {C}?",
    "Eac": "Are no {A} {C}?",
    "Iac": "Are some {A} {C}?",
    "Oac": "Are some {A} not {C}?",
    "Aca": "Are all {C} {A}?",
    "Eca": "Are no {C} {A}?",
    "Ica": "Are some {C} {A}?",
    "Oca": "Are some {C} not {A}?",
}


def build_all_form_templates():
    """
    Generate templates for all 64 syllogistic forms × 8 conclusions.
    Returns a list of 512 template dicts, each with the same shape as syllogism_templates.json.
    """
    moods = ["A", "E", "I", "O"]
    templates = []

    for m1 in moods:
        for m2 in moods:
            for fig in range(1, 5):
                form = f"{m1}{m2}{fig}"
                (s1, p1), (s2, p2) = _FIGURE_SP[fig]

                prem1 = _MOOD_PREMISE[m1].format(S=s1, P=p1)
                prem2 = _MOOD_PREMISE[m2].format(S=s2, P=p2)

                for conc_code, question in _CONCLUSION_Q.items():
                    templates.append({
                        "form": form,
                        "premises": [prem1, prem2],
                        "question": question,
                        "options": ["Yes", "No", "Cannot determine"],
                        "conclusion": conc_code,
                    })

    return templates

def _inst(template, A, B, C):
    """
    Helper function takes syllogism template and fills in the placeholders {A}, {B}, {C}
    with real words from domains.json.

    Returns: the instantiated premises and question.
    """
    # Replace placeholders {A}, {B}, {C} in premises and question
    prems = [p.format(A=A, B=B, C=C) for p in template["premises"]]
    q = template["question"].format(A=A, B=B, C=C)
    return [(x) for x in prems], (q)


def question_to_conclusion(question: str) -> str:
    """
    Map the question text string to the conclusion code (e.g., 'Aac').
    """
    if "all {A} {C}" in question:
        return "Aac"
    elif "no {A} {C}" in question:
        return "Eac"
    elif "some {A} {C}" in question:
        return "Iac"
    elif "some {A} not {C}" in question:
        return "Oac"
    elif "some {C} {A}" in question:
        return "Ica"
    elif "some {C} not {A}" in question:
        return "Oca"
    elif "all {C} {A}" in question:
        return "Aca"
    elif "no {C} {A}" in question:
        return "Eca"
    else:
        raise ValueError(f"Unknown question format: {question}")


def compute_wcs_label(form: str, question: str) -> int:
    """
    Compute the WCS label (0/1/2) for the given form and question.
    """
    P, domain = build_program_for_form(form)
    I = least_model(P, domain=domain)
    vals = check_9_conclusions(I, a="a", c="c")

    conc = question_to_conclusion(question)
    tv = vals[conc]

    if tv == TV.TRUE:
        return 0  # Yes
    elif tv == TV.FALSE:
        return 1  # No
    else:
        return 2  # Cannot determine


def make_seed(n_items: int,
              templates_path: str,
              domains_path: str, 
              out_fp: str,
              rnd_seed: int = 42):
    """
    Generates a dataset of syllogism examples 
    based on syllogism_templates.json and domains.json.
    This is the --template-based random generation method
    This is mainly for the few first pilot experiments, where we want a smaller dataset with some variability.

    Each generated example is created by combining:
      - one syllogism form (template) e.g. AA4, OA4 from configs/syllogism_templates.json
      - one set of (A, B, C) terms sampled from configs/domains.json

    Args:
        n_items (int): Number of syllogism examples to generate.
        templates_path (str): Path to the JSON file containing syllogism templates.
        out_fp (str): Output file.
        rnd_seed (int): Random seed for reproducibility.
    """

    # Set random seed for reproducible generation
    random.seed(rnd_seed)

    # Load syllogism templates from configs/syllogism_templates.json
    with open(templates_path) as f:
        templates = json.load(f)

    # Load domains from configs/domains.json
    with open(domains_path) as f:          
        domains = json.load(f)

    # Collect all domain triplets (domain name + A, B, C terms)
    triples = []
    for dom, rows in domains.items():
        for (A, B, C) in rows:
            triples.append((dom, A, B, C))

    items = []
    i = 0

    # Generate items until reaching the desired count
    while len(items) < n_items:
        tmpl = templates[i % len(templates)]          # cycle through templates
        dom, A, B, C = random.choice(triples)         # randomly choose a domain set
        premises, question = _inst(tmpl, A, B, C)     # fill in the placeholders

        # Build one syllogism example
        conclusion_code = question_to_conclusion(tmpl["question"])
        item = {
            "id": f"ex{uuid.uuid4().hex[:8]}",        
            "premises": premises,                     
            "question": question,                     
            "options": tmpl["options"],               
            "label_classical": compute_classical_label(tmpl["form"], conclusion_code),
            "label_wcs": compute_wcs_label(tmpl["form"], tmpl["question"]),           
            "role": "none",                           
            "domain": dom,                            
            "form": tmpl["form"]                      
        }

        items.append(item)
        i += 1

    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)

    with open(out_fp, "w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} seeds to {out_fp}")


def make_seed_all_forms(domains_path: str, out_fp: str):
    """
    Generate an exhaustive dataset covering all 64 forms × 8 conclusions from build_all_form_templates() 
    and x with all domain triplets.
    No random sampling — every combination is included exactly once. Full dataset with expanded forms.
    This is for the final pipeline evaluation, where we want to test all syllogism forms without sampling variability.

    Each output record groups all 8 conclusions under one form + domain triplet, 
    with their respective classical and WCS labels.
    """
    moods = ["A", "E", "I", "O"]

    with open(domains_path) as f:
        domains = json.load(f)

    triples = []
    for dom, rows in domains.items():
        for (A, B, C) in rows:
            triples.append((dom, A, B, C))

    items = []
    for m1 in moods:
        for m2 in moods:
            for fig in range(1, 5):
                form = f"{m1}{m2}{fig}"
                (s1, p1), (s2, p2) = _FIGURE_SP[fig]
                prem1 = _MOOD_PREMISE[m1].format(S=s1, P=p1)
                prem2 = _MOOD_PREMISE[m2].format(S=s2, P=p2)

                for dom, A, B, C in triples:
                    premises = [
                        prem1.format(A=A, B=B, C=C),
                        prem2.format(A=A, B=B, C=C),
                    ]

                    conclusions = {}
                    for conc_code, q_template in _CONCLUSION_Q.items():
                        question = q_template.format(A=A, B=B, C=C)
                        conclusions[conc_code] = {
                            "question": question,
                            "classical": compute_classical_label(form, conc_code),
                            "wcs": compute_wcs_label(form, q_template),
                        }

                    item = {
                        "id": f"ex{uuid.uuid4().hex[:8]}",
                        "form": form,
                        "domain": dom,
                        "role": "none",
                        "premises": premises,
                        "conclusions": conclusions,
                    }
                    items.append(item)

    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)

    with open(out_fp, "w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} seeds to {out_fp} "
          f"(64 forms × {len(triples)} domain triplets, 8 conclusions each)")
