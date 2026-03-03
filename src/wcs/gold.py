from .encoders import build_program_for_form
from .leastmodel import least_model
from .tv import TV
from .syllogisms import all_A_are_B, no_A_are_B, some_A_are_B, some_A_are_not_B

def question_to_query(question: str):
    q = question.strip().lower()
    # normalize spaces
    q = " ".join(q.split())

    # detect mood
    if q.startswith("are all"):
        mood = "A"
    elif q.startswith("are no"):
        mood = "E"
    elif q.startswith("are some") and " not " in q:
        mood = "O"
    elif q.startswith("are some"):
        mood = "I"
    else:
        raise ValueError(f"Unrecognized question pattern: {question}")

    # detect term order by placeholders
    # We rely on {A},{B},{C} being present in the original template.
    # If you already format them to actual words, keep a parallel field or infer from 'form'.
    if "{a}" in q and "{c}" in q:
        left, right = "a", "c"
    elif "{c}" in q and "{a}" in q:
        left, right = "c", "a"
    else:
        # fallback: many of your templates are always about A and C anyway
        # but for robustness you'd keep placeholders before formatting
        raise ValueError(f"Cannot infer term placeholders from: {question}")

    return mood, left, right


def eval_query(I, mood: str, left: str, right: str) -> TV:
    if mood == "A":
        return all_A_are_B(I, left, right)
    if mood == "E":
        return no_A_are_B(I, left, right)
    if mood == "I":
        return some_A_are_B(I, left, right)
    if mood == "O":
        return some_A_are_not_B(I, left, right)
    raise ValueError(mood)


def wcs_label_for_item(form: str, question: str) -> int:
    P, domain = build_program_for_form(form)
    I = least_model(P, domain=domain)

    mood, left, right = question_to_query(question)
    v = eval_query(I, mood, left, right)

    if v == TV.TRUE:
        return 0  # Yes
    if v == TV.FALSE:
        return 1  # No
    return 2      # Cannot determine