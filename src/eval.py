import json, csv
from collections import defaultdict

# Class index → label used in the option list (A/B/C).
_CLASS_LETTERS = ["A", "B", "C"]
_CLASS_NAMES = ["Yes", "No", "Cannot determine"]


def _empty_cm():
    """3×3 confusion matrix: cm[gold_idx][pred_idx]."""
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def _metrics_from_cm(cm):
    """
    Compute per-class precision/recall/F1 and macro-F1 from a 3×3 confusion
    matrix cm[gold][pred]. Undefined values (0/0) are returned as 0.0.
    """
    per_class = {}
    for c in range(3):
        tp = cm[c][c]
        fp = sum(cm[g][c] for g in range(3)) - tp
        fn = sum(cm[c][p] for p in range(3)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn,
                        "support": sum(cm[c])}
    macro_f1 = sum(per_class[c]["f1"] for c in range(3)) / 3
    return per_class, macro_f1


def _print_role_report(role, n, gold_name, cm, per_class, macro_f1, acc, c_usage):
    """Pretty-print a per-(role, gold) confusion-matrix + metrics block."""
    print(f"\n─── role={role}  gold={gold_name}  n={n}  acc={acc:.3f}  "
          f"macro_F1={macro_f1:.3f}  C_usage={c_usage:.3f} ───")
    print("  Confusion matrix (rows=gold, cols=pred):")
    print("        " + "  ".join(f"pred={l:>3s}" for l in _CLASS_LETTERS) + "   support")
    for c in range(3):
        row = f"  gold={_CLASS_LETTERS[c]}  " + "  ".join(f"{cm[c][p]:>7d}" for p in range(3))
        row += f"   {per_class[c]['support']:>7d}"
        print(row)
    print(f"  {'':10s}{'prec':>8s}  {'recall':>8s}  {'F1':>8s}")
    for c in range(3):
        pc = per_class[c]
        print(f"  {_CLASS_LETTERS[c]}({_CLASS_NAMES[c][:6]:6s})"
              f"  {pc['precision']:8.3f}  {pc['recall']:8.3f}  {pc['f1']:8.3f}")


def evaluate(gold_fp: str, pred_fp: str, out_csv: str):
    """
    Compare model predictions to classical and WCS labels and write role-level metrics.

    Both gold and pred files use the grouped format where each record has a
    "conclusions" dict mapping conclusion codes to their labels/predictions.

    For each role, the CSV records (for both classical and WCS gold):
        - accuracy
        - macro-F1
        - per-class F1 (A=Yes, B=No, C=Cannot determine)
        - recall_C and precision_C (the "willingness to say I don't know" signal)
        - C-usage rate (fraction of predictions that were C)
        - delta_wcs_minus_classical for both accuracy and macro-F1

    A full confusion matrix + per-class precision/recall/F1 is also printed to
    stdout for every (role, gold) combination.

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

    # Per-role tallies:
    #   n           : total (item, conclusion) pairs seen
    #   c_pred      : # predictions equal to 2 (Cannot determine)
    #   cm_classical: 3×3 confusion matrix vs classical gold
    #   cm_wcs      : 3×3 confusion matrix vs wcs gold
    #   classical_ok/wcs_ok: raw-match counts (accuracy = ok/n)
    by_role = defaultdict(lambda: {"n": 0, "c_pred": 0,
                                   "cm_classical": _empty_cm(),
                                   "cm_wcs": _empty_cm(),
                                   "classical_ok": 0, "wcs_ok": 0})

    with open(pred_fp) as f:
        for line in f:
            p = json.loads(line)
            g = gold.get(p["id"])
            if not g:
                continue
            role = g["role"]

            for conc_code, pred_conc in p["conclusions"].items():
                gold_conc = g["conclusions"].get(conc_code)
                if not gold_conc:
                    continue
                pred_idx = pred_conc.get("pred_index", -1)
                r = by_role[role]
                r["n"] += 1
                if pred_idx == 2:
                    r["c_pred"] += 1

                gcls = gold_conc["classical"]
                gwcs = gold_conc["wcs"]

                # Accuracy (equivalent to sum of confusion-matrix diagonal)
                r["classical_ok"] += int(pred_idx == gcls)
                r["wcs_ok"] += int(pred_idx == gwcs)

                # Only fold well-formed predictions into the confusion matrix.
                # Parse failures (pred_idx=-1) still count toward `n` but do
                # not add to any TP cell — they implicitly hurt recall for the
                # true class without affecting precision.
                if 0 <= pred_idx <= 2:
                    if 0 <= gcls <= 2:
                        r["cm_classical"][gcls][pred_idx] += 1
                    if 0 <= gwcs <= 2:
                        r["cm_wcs"][gwcs][pred_idx] += 1

    header = [
        "role", "n", "c_usage",
        "acc_classical", "macro_f1_classical",
        "f1_A_classical", "f1_B_classical", "f1_C_classical",
        "recall_C_classical", "precision_C_classical",
        "acc_wcs", "macro_f1_wcs",
        "f1_A_wcs", "f1_B_wcs", "f1_C_wcs",
        "recall_C_wcs", "precision_C_wcs",
        "delta_acc_wcs_minus_classical",
        "delta_macro_f1_wcs_minus_classical",
    ]
    rows = []

    for role, agg in sorted(by_role.items()):
        n = agg["n"] or 1
        c_usage = agg["c_pred"] / n

        acc_cls = agg["classical_ok"] / n
        acc_wcs = agg["wcs_ok"] / n

        per_cls_cls, macro_f1_cls = _metrics_from_cm(agg["cm_classical"])
        per_cls_wcs, macro_f1_wcs = _metrics_from_cm(agg["cm_wcs"])

        _print_role_report(role, agg["n"], "classical",
                           agg["cm_classical"], per_cls_cls, macro_f1_cls,
                           acc_cls, c_usage)
        _print_role_report(role, agg["n"], "wcs",
                           agg["cm_wcs"], per_cls_wcs, macro_f1_wcs,
                           acc_wcs, c_usage)

        rows.append([
            role, agg["n"], round(c_usage, 3),
            round(acc_cls, 3), round(macro_f1_cls, 3),
            round(per_cls_cls[0]["f1"], 3),
            round(per_cls_cls[1]["f1"], 3),
            round(per_cls_cls[2]["f1"], 3),
            round(per_cls_cls[2]["recall"], 3),
            round(per_cls_cls[2]["precision"], 3),
            round(acc_wcs, 3), round(macro_f1_wcs, 3),
            round(per_cls_wcs[0]["f1"], 3),
            round(per_cls_wcs[1]["f1"], 3),
            round(per_cls_wcs[2]["f1"], 3),
            round(per_cls_wcs[2]["recall"], 3),
            round(per_cls_wcs[2]["precision"], 3),
            round(acc_wcs - acc_cls, 3),
            round(macro_f1_wcs - macro_f1_cls, 3),
        ])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"\nSaved metrics to {out_csv}")