import json, random, uuid
from pathlib import Path

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


def make_seed(n_items: int,
              templates_path: str,
              domains_path: str, 
              out_fp: str,
              rnd_seed: int = 42):
    """
    Generates a dataset of syllogism examples 
    based on syllogism templates and different domains.

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
        item = {
            "id": f"ex{uuid.uuid4().hex[:8]}",        
            "premises": premises,                     
            "question": question,                     
            "options": tmpl["options"],               
            "label_classical": tmpl["label_classical"],
            "label_wcs": tmpl["label_wcs"],           
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
