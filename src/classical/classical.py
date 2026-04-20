from itertools import product, combinations
from wcs.encoders import figure_pairs

# All 8 possible object types: (a_val, b_val, c_val), each 0 or 1.
# e.g. (1, 0, 1) = object that is a-member, not b-member, c-member.
# We can represent states with 0 or 1 since we're doing two-valued logic.
ALL_TYPES = list(product([0, 1], repeat=3))
PRED_IDX = {"a": 0, "b": 1, "c": 2}


# ── Syllogism predicates ──────────────────────────────────────────────────────
# These mirror src/wcs/syllogisms.py but for classical (two-valued) models.
# A model is a list of non-empty types: each type is (a_val, b_val, c_val).

def all_A_are_B(model, A, B):
    """All A are B: no object is A without being B."""
    ai, bi = PRED_IDX[A], PRED_IDX[B]
    return not any(t[ai] == 1 and t[bi] == 0 for t in model)

def no_A_are_B(model, A, B):
    """No A are B: no object is both A and B."""
    ai, bi = PRED_IDX[A], PRED_IDX[B]
    return not any(t[ai] == 1 and t[bi] == 1 for t in model)

def some_A_are_B(model, A, B):
    """Some A are B: at least one object is both A and B."""
    ai, bi = PRED_IDX[A], PRED_IDX[B]
    return any(t[ai] == 1 and t[bi] == 1 for t in model)

def some_A_are_not_B(model, A, B):
    """Some A are not B: at least one object is A but not B."""
    ai, bi = PRED_IDX[A], PRED_IDX[B]
    return any(t[ai] == 1 and t[bi] == 0 for t in model)


def _premise_holds(model, mood, subj, pred):
    """Dispatch to the right syllogism function based on mood."""
    if mood == "A": return all_A_are_B(model, subj, pred)
    if mood == "E": return no_A_are_B(model, subj, pred)
    if mood == "I": return some_A_are_B(model, subj, pred)
    if mood == "O": return some_A_are_not_B(model, subj, pred)
    raise ValueError(f"Unknown mood: {mood}")


# ── Model enumeration ─────────────────────────────────────────────────────────

def all_models():
    """
    Generate all possible models as lists of non-empty types.

    A model is a non-empty subset of ALL_TYPES — it says which kinds of
    objects exist in the world. We generate every possible combination,
    from a world with only one type of object up to a world with all 8.

    There are 2^8 - 1 = 255 non-empty models in total.
    """
    for r in range(1, len(ALL_TYPES) + 1):
        for combo in combinations(ALL_TYPES, r):
            yield list(combo)


def all_satisfying_models(form):
    """
    Return all models in which both premises of the given syllogism form hold,
    under Aristotelian existential import: both end terms a and c must be
    non-empty in every valid model.

    This matches the FOL baseline used in the cognitive science literature
    (Khemlani & Johnson-Laird 2012) where terms are assumed to refer to
    non-empty sets — consistent with how human reasoning experiments are set up.

    This is the classical counterpart of least_model() in the WCS pipeline:
    instead of computing one minimal model, we collect every model that
    is consistent with the premises.

    Args:
        form: Syllogism form string, e.g. "EA3"
    Returns:
        List of models where both premises hold and a, c are non-empty.
    """
    mood1, mood2 = form[0], form[1]
    fig = int(form[2])
    (y1, z1), (y2, z2) = figure_pairs(fig)

    ai = PRED_IDX["a"]
    bi = PRED_IDX["b"]
    ci = PRED_IDX["c"]

    satisfying = []
    for model in all_models():
        premise1_holds = _premise_holds(model, mood1, y1, z1)
        premise2_holds = _premise_holds(model, mood2, y2, z2)
        a_nonempty = any(t[ai] == 1 for t in model)
        b_nonempty = any(t[bi] == 1 for t in model)
        c_nonempty = any(t[ci] == 1 for t in model)
        if premise1_holds and premise2_holds and a_nonempty and b_nonempty and c_nonempty:
            satisfying.append(model)
    return satisfying


# ── Conclusion checking ───────────────────────────────────────────────────────

def check_9_conclusions(satisfying_models, a="a", c="c"):
    """
    For each of the 8 standard conclusions, check whether it holds in
    all, none, or some of the satisfying models.

    This mirrors check_9_conclusions() in gold_wcs.py. The difference is
    that WCS evaluates conclusions against one least model (returning TV),
    while here we evaluate across the full set of satisfying models
    (returning 0/1/2 directly).

    Returns:
        Dict mapping conclusion code -> int:
            0 = Yes              (holds in ALL satisfying models)
            1 = No               (holds in NO  satisfying models)
            2 = Cannot determine (holds in some but not all)
    """
    conclusions = {
        "Aac": (all_A_are_B,      a, c),
        "Eac": (no_A_are_B,       a, c),
        "Iac": (some_A_are_B,     a, c),
        "Oac": (some_A_are_not_B, a, c),
        "Aca": (all_A_are_B,      c, a),
        "Eca": (no_A_are_B,       c, a),
        "Ica": (some_A_are_B,     c, a),
        "Oca": (some_A_are_not_B, c, a),
    }

    out = {}
    for code, (fn, subj, pred) in conclusions.items():
        true_in = sum(1 for m in satisfying_models if fn(m, subj, pred))
        if true_in == len(satisfying_models):
            out[code] = 0  # Yes
        elif true_in == 0:
            out[code] = 1  # No
        else:
            out[code] = 2  # Cannot determine
    return out


# ── Top-level entry point ─────────────────────────────────────────────────────

def compute_classical_label(form: str, conclusion_code: str) -> int:
    """
    Compute the classical logic label (0/1/2) for the given syllogism form
    and conclusion code.

    Mirrors compute_wcs_label() in generate.py:
        WCS:       build_program_for_form -> least_model -> check_9_conclusions -> label
        Classical: all_satisfying_models  ->                check_9_conclusions -> label

    Returns:
        0 -> Yes             (conclusion follows in ALL satisfying models)
        1 -> No              (conclusion fails  in ALL satisfying models)
        2 -> Cannot determine
    """
    satisfying_models = all_satisfying_models(form)
    vals = check_9_conclusions(satisfying_models)
    return vals[conclusion_code]


def entailed_set_for_form(form: str):
    """
    Return the set of conclusion codes that classically follow (label=0)
    from the given form. Returns {"NVC"} if nothing follows.

    Mirrors entailed_set_for_form() in gold_wcs.py.
    """
    satisfying_models = all_satisfying_models(form)
    vals = check_9_conclusions(satisfying_models)
    entailed = {code for code, label in vals.items() if label == 0}
    return entailed if entailed else {"NVC"}


def save_classical_gold_table(out_path: str = "data/processed/classical_gold_table.json"):
    """
    Build and save the classical gold table for all 64 syllogism forms.
    Maps each form to its list of classically entailed conclusion codes.

    Mirrors save_all_forms_json() in sanity_check.py.
    """
    import json
    from pathlib import Path

    moods = ["A", "E", "I", "O"]
    results = {}
    for m1 in moods:
        for m2 in moods:
            for fig in range(1, 5):
                form = f"{m1}{m2}{fig}"
                entailed = sorted(entailed_set_for_form(form))
                results[form] = entailed
                print(form, entailed)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved classical gold table to {out_path}")