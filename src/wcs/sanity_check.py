from wcs.encoders import build_program_for_form
from wcs.leastmodel import least_model
from wcs.tv import TV
from wcs.gold_wcs import entailed_set_for_form
from wcs.syllogisms import (
    all_A_are_B, no_A_are_B, some_A_are_B, some_A_are_not_B
)

def tv_to_label(v: TV) -> int:
    # Yes=0, No=1, Cannot=2
    if v == TV.TRUE:
        return 0
    if v == TV.FALSE:
        return 1
    return 2

def wcs_answer(form: str, query: str) -> int:
    """
    query is one of: Aac, Eac, Iac, Oac, Aca, Eca, Ica, Oca
    """
    P, domain = build_program_for_form(form)
    I = least_model(P, domain=domain)

    mood = query[0]
    left = query[1]
    right = query[2]

    if mood == "A":
        v = all_A_are_B(I, left, right)
    elif mood == "E":
        v = no_A_are_B(I, left, right)
    elif mood == "I":
        v = some_A_are_B(I, left, right)
    elif mood == "O":
        v = some_A_are_not_B(I, left, right)
    else:
        raise ValueError(query)

    return tv_to_label(v)

def run():
    tests = [
        ("OA4", "Oca", 0),  # Yes
        ("AA4", "Aac", 0),  # Yes
        ("EA3", "Eac", 2),  # Cannot
        ("IE1", "Oac", 2),  # Cannot
    ]

    ok = True
    for form, query, expected in tests:
        got = wcs_answer(form, query)
        status = "OK" if got == expected else "FAIL"
        print(f"{status}: {form} {query} -> got {got}, expected {expected}")
        ok = ok and (got == expected)

    if not ok:
        raise SystemExit("Sanity checks failed.")
    print("YAY all sanity checks passed!")

def run_all_forms():
    moods = ["A", "E", "I", "O"]
    results = {}
    for m1 in moods:
        for m2 in moods:
            for f in range(1, 5):
                form = f"{m1}{m2}{f}"
                entailed = sorted(entailed_set_for_form(form))
                results[form] = entailed
                print(form, entailed)
    return results


def save_all_forms_json(out_path: str = "data/processed/wcs_gold_table.json"):
    import json
    from pathlib import Path

    results = run_all_forms()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved WCS gold table to {out_path}")
    return out_path


if __name__ == "__main__":
    run()
    save_all_forms_json()