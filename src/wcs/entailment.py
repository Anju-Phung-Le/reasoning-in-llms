from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .encoders import build_program_for_form
from .fixpoint import least_model
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
    Evaluate the 8 quantified conclusions in the paper + NVC.
    Returns TV values for each label.
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

    # NVC: none of the 8 are TRUE
    out["NVC"] = TV.TRUE if all(v != TV.TRUE for k, v in out.items()) else TV.FALSE
    return out


@dataclass
class WCSResult:
    I: object
    values: Dict[str, TV]

    def entailed_labels(self) -> Set[str]:
        return {k for k, v in self.values.items() if v == TV.TRUE}


def run_wcs_program(P: Program, domain=None) -> WCSResult:
    """
    Compute least model and evaluate 9 conclusions.
    """
    I = least_model(P, domain=domain)
    vals = check_9_conclusions(I, a="a", c="c")
    return WCSResult(I=I, values=vals)

def wcs_predict_form(form: str):
    P = build_program_for_form(form)
    from .fixpoint import least_model

    I = least_model(P, domain=["o1", "o2", "o3"])

    return I