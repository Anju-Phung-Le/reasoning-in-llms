from pathlib import Path
import json, uuid

ROLES = ["none","child","teen","adult"]

def expand_roles(in_fp, out_fp):
    """ Reads a JSONL dataset of syllogism items and expands each example
    to include multiple roles. So in this case for one seed i will get 4 times as many items, 
    for each role in ROLES.

    Each item keeps the same content but updates its 'role' field and appends the role
    name to the 'id' (e.g., 'prem1_child').

    The expanded items are then written line-by-line to a new JSONL file.

    Args:
        in_fp (str): Path to the input JSONL file containing base items - raw/seed.jsonl
        out_fp (str): Path where the expanded dataset will be saved - processed/dataset_v1.jsonl"""
    
    with open(in_fp) as f: 
        items = [json.loads(l) for l in f] 
        out = []
    for item in items:
        base_id = item.get("id") or str(uuid.uuid4())[:8]
        for r in ROLES:
            x = dict(item)
            x["role"] = r
            x["id"] = f"{base_id}_{r}"
            out.append(x)
    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
