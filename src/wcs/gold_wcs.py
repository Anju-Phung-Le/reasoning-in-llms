from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set

from .encoders import build_program_for_form
from .leastmodel import least_model
from .program import Program
from .syllogisms import (
    all_A_are_B,
    no_A_are_B,
    some_A_are_B,
    some_A_are_not_B,
)
from .tv import TV


CONCLUSIONS_9 = ("Aac", "Eac", "Iac", "Oac", "Aca", "Eca", "Ica", "Oca", "NVC")

def check_9_conclusions(I, a: str = "a", c: str = "c") -> Dict[str, TV]:
    """
    Evaluate all 9 possible conclusions for a given interpretation I.

    This function checks the 8 standard categorical conclusions and NVC.
    Each conclusion is evaluated using the syllogism predicates from syllogisms.py,
    which return TRUE, FALSE, or UNKNOWN based on the model's domain and assignments.

    Parameters:
    - I: The interpretation (model) from least_model, containing domain and predicate assignments.
    - a, c: Term names (default "a" and "c", as per standard syllogism notation).

    Returns:
    - Dict[str, TV]: Mapping of conclusion label to its truth value.
      - Aac: All a are c
      - Eac: No a are c
      - Iac: Some a are c
      - Oac: Some a are not c
      - Aca: All c are a
      - Eca: No c are a
      - Ica: Some c are a
      - Oca: Some c are not a
      - NVC: No Valid Conclusion (TRUE only if all 8 above are not TRUE)
    """
    out: Dict[str, TV] = {}
    out["Aac"] = all_A_are_B(I, a, c)
    out["Eac"] = no_A_are_B(I, a, c)
    out["Iac"] = some_A_are_B(I, a, c)
    out["Oac"] = some_A_are_not_B(I, a, c)

    out["Aca"] = all_A_are_B(I, c, a)
    out["Eca"] = no_A_are_B(I, c, a)
    out["Ica"] = some_A_are_B(I, c, a)
    out["Oca"] = some_A_are_not_B(I, c, a)

    # NVC: ONLY IF none of the 8 are TRUE
    out["NVC"] = TV.TRUE if all(v != TV.TRUE for k, v in out.items()) else TV.FALSE
    return out


@dataclass
class WCSResult:
    """
    Dataclass to hold the result of running a WCS program.

    Attributes:
    - I: The computed least model (interpretation).
    - values: Dict of conclusion labels to their TV values (from check_9_conclusions).

    Methods:
    - entailed_labels(): Returns the set of labels that are TRUE.
    """
    I: object  # The least model interpretation
    values: Dict[str, TV]  # Conclusion values

    def entailed_labels(self) -> Set[str]:
        """Return the set of conclusion labels that are TRUE in this result."""
        return {k for k, v in self.values.items() if v == TV.TRUE}


def run_wcs_program(P: Program, domain=None) -> WCSResult:
    """
    Run a full WCS evaluation on a given program.

    This is the point for evaluating a WCS program:
    1. Compute the least model using leastmodel.py.
    2. Evaluate all 9 conclusions using check_9_conclusions.

    Parameters:
    - P: The WCS program (from program.py).
    - domain: Optional domain override (list of objects).

    Returns:
    - WCSResult: Contains the model I and the conclusion values.

    Example:
    >>> P, dom = build_program_for_form("AA1")
    >>> result = run_wcs_program(P, domain=dom)
    >>> print(result.entailed_labels())  # e.g., {"Aac", "Iac"}
    """
    I = least_model(P, domain=domain)  # Compute the least model
    vals = check_9_conclusions(I, a="a", c="c")  # Evaluate conclusions
    return WCSResult(I=I, values=vals)


def wcs_predict_form(form: str):
    """
    Predict the entailed conclusions for a syllogism form.

    This is a simple wrapper around entailed_set_for_form for API consistency.

    Parameters:
    - form: Syllogism form string (e.g., "AA1").

    Returns:
    - Set[str]: Entailed conclusion labels.
    """
    return entailed_set_for_form(form)


def entailed_set_for_form(form: str, domain=None):
    """
    Compute the set of entailed conclusion labels for a given syllogism form.

    This is the core gold label function:
    1. Build the WCS program for the form using encoders.py.
    2. Compute the least model using leastmodel.py.
    3. Evaluate each of the 8 categorical conclusions.
    4. If none are TRUE, add "NVC" to the set.

    Parameters:
    - form: Syllogism form (e.g., "AA1", "EO2").
    - domain: Optional domain override.

    Returns:
    - Set[str]: Set of entailed labels. Either specific conclusions (e.g., {"Aac"}) or {"NVC"}.

    Notes:
    - This function focuses on definite entailments (TRUE conclusions).
    - UNKNOWN conclusions are not included (WCS is cautious).
    - NVC indicates the syllogism does not entail any definite conclusion.
    """
    # Build the WCS program and domain for the form
    P, dom = build_program_for_form(form)

    # Compute the least model (this is the "reasoning" step)
    I = least_model(P, domain=dom if domain is None else domain)

    # Check each conclusion and collect those that are TRUE
    entailed = set()
    if all_A_are_B(I, "a", "c") == TV.TRUE: entailed.add("Aac")
    if no_A_are_B(I,  "a", "c") == TV.TRUE: entailed.add("Eac")
    if some_A_are_B(I,"a", "c") == TV.TRUE: entailed.add("Iac")
    if some_A_are_not_B(I,"a","c") == TV.TRUE: entailed.add("Oac")

    if all_A_are_B(I, "c", "a") == TV.TRUE: entailed.add("Aca")
    if no_A_are_B(I,  "c", "a") == TV.TRUE: entailed.add("Eca")
    if some_A_are_B(I,"c", "a") == TV.TRUE: entailed.add("Ica")
    if some_A_are_not_B(I,"c","a") == TV.TRUE: entailed.add("Oca")

    # If no conclusions are TRUE, the syllogism has No Valid Conclusion
    if len(entailed) == 0:
        entailed.add("NVC")

    return entailed