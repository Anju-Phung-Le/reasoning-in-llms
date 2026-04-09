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

# Mapping
def check_9_conclusions(I, a: str = "a", c: str = "c") -> Dict[str, TV]:
    """
    Evaluate all 9 possible conclusions for a given interpretation I (8 standard conclusions and NVC).
    Each conclusion is evaluated using the syllogism predicates from syllogisms.py,
    which return TRUE, FALSE, or UNKNOWN.

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

    #   A  ⊃  I   ("All" subsumes "Some")
    #   E  ⊃  O   ("No"  subsumes "Some … not")
    # This is subalternation in traditional logic, so if Aac is TRUE, then Iac must be FALSE 
    # (since WCS is cautious and does not infer UNKNOWN as TRUE).
    if out["Aac"] == TV.TRUE:
        out["Iac"] = TV.FALSE
    if out["Aca"] == TV.TRUE:
        out["Ica"] = TV.FALSE
    if out["Eac"] == TV.TRUE:
        out["Oac"] = TV.FALSE
    if out["Eca"] == TV.TRUE:
        out["Oca"] = TV.FALSE

    # NVC: ONLY IF none of the 8 are TRUE
    out["NVC"] = TV.TRUE if all(v != TV.TRUE for k, v in out.items()) else TV.FALSE
    return out

#Return set of mapped labels
@dataclass
class WCSResult:
    """
    Dataclass to hold the result of running a WCS program.

    Attributes:
    - I: The computed least model (interpretation).
    - values: Dict of conclusion labels to their TV (from check_9_conclusions).

    Methods:
    - entailed_labels(): Returns the set of labels that are TRUE.
    """
    I: object  # Least model interpretation
    values: Dict[str, TV]  # Conclusion values

    def entailed_labels(self) -> Set[str]:
        """Return the set of conclusion labels that are TRUE in this result."""
        return {k for k, v in self.values.items() if v == TV.TRUE}


def run_wcs_program(P: Program, domain=None) -> WCSResult:
    """
    Runs full WCS evaluation on a given program.
    1. Compute the least model using leastmodel.py.
    2. Evaluate all 9 conclusions using check_9_conclusions.

    Parameters:
    - P: The WCS program (from program.py).
    - domain: Optional domain (list of objects).

    Returns:
    - WCSResult: Contains the model I and the conclusion values.

    f.eks:
    >>> P, dom = build_program_for_form("AA1")
    >>> result = run_wcs_program(P, domain=dom)
    >>> print(result.entailed_labels())  # e.g., {"Aac", "Iac"}
    """
    I = least_model(P, domain=domain)  # Compute the least model
    vals = check_9_conclusions(I, a="a", c="c")  # Evaluate conclusions
    return WCSResult(I=I, values=vals)


def wcs_predict_form(form: str):
    """
    Simple wrapper around entailed_set_for_form.
    Returns:
    - Set[str]: Entailed conclusion labels.
    """
    return entailed_set_for_form(form)


def entailed_set_for_form(form: str, domain=None) -> Set[str]:
    """
    Wrapper around run_wcs_program that extracts the entailed labels.

    Parameters:
    - form: Syllogism form (e.g., "AA1", "EO2").
    - domain: Optional domain (list of objects).

    Returns:
    - Set[str]: Set of entailed labels. Either specific conclusions (e.g., {"Aac"}) or {"NVC"}.

    Notes:
    - This function focuses on definite entailments (TRUE conclusions).
    - UNKNOWN conclusions are not included (WCS is cautious).
    - NVC indicates the syllogism does not entail any definite conclusion.
    """
    P, dom = build_program_for_form(form)
    result = run_wcs_program(P, domain=dom if domain is None else domain)
    return result.entailed_labels()

# TODO: use these results to generate dataset for all 64 forms. 