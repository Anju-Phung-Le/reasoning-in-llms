import json, csv
from collections import defaultdict

def evaluate(gold_fp: str, pred_fp: str, out_csv: str):
    """
    Compare model predictions to classical and WCS labels and write role-level metrics.

    Args:
        gold_fp (str): Path to the processed dataset (e.g. data/processed/dataset_v1.jsonl)
        pred_fp (str): Path to model predictions (e.g. outputs/flan_t5_predictions.jsonl)
        out_csv (str): Path to save the evaluation results (e.g. outputs/outputs_log.csv)
    """
    # Load gold(true) labels into a dict using each item's id as key
    gold = {}
    with open(gold_fp) as f:
        for line in f:
            item = json.loads(line)
            gold[item["id"]] = item

    # Initialize counters for each role
    by_role = defaultdict(lambda: {"n":0, "classical_ok":0, "wcs_ok":0})
    
    # Compare model predictions to gold labels
    with open(pred_fp) as f:
        for line in f:
            p = json.loads(line)
            g = gold.get(p["id"])
            if not g:
                continue
            role = g["role"]
            pred_idx = p.get("pred_index", -1)
            # Count total items per role
            by_role[role]["n"] += 1

            # Count correct answers for classical and WCS reasoning
            by_role[role]["classical_ok"] += int(pred_idx == g["label_classical"])
            by_role[role]["wcs_ok"] += int(pred_idx == g["label_wcs"])

        # Compute accuracy per role

    rows = []
    for role, agg in sorted(by_role.items()):
        n = agg["n"] or 1
        acc_classical = agg["classical_ok"]/n
        acc_wcs = agg["wcs_ok"]/n
        rows.append([role, n, round(acc_classical,3), round(acc_wcs,3), round(acc_wcs-acc_classical,3)])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role","n","acc_classical","acc_wcs","delta_wcs_minus_classical"])
        w.writerows(rows)

    print(f"Saved metrics to {out_csv}")
