from __future__ import annotations
from .tv import Interpretation, TV
from .program import Program


def least_model(P: Program, domain=None, max_iters: int = 1000) -> Interpretation:
    """
    Compute least L-model of the weak completion using the Φ operator (paper Section 2.3).
    - Start with all atoms UNKNOWN
    - Iterate:
        TRUE  if some defining body is TRUE
        FALSE if there is a definition and all bodies are FALSE
        UNKNOWN otherwise
    """
    wcP = P.weak_completion()

    # If caller doesn't pass domain, use objects appearing in heads.
    dom = list(domain) if domain is not None else sorted(wcP.domain())

    I = Interpretation(dom)

    # Iterate to fixpoint
    for _ in range(max_iters):
        changed = False

        for (pred, obj), bodies in wcP.defs.items():
            # (after weak completion, each head has exactly 1 body, but keep list form)
            vals = [b(I) for b in bodies]

            if any(v == TV.TRUE for v in vals):
                new_val = TV.TRUE
            elif all(v == TV.FALSE for v in vals):
                new_val = TV.FALSE
            else:
                new_val = TV.UNKNOWN

            old_val = I.get(pred, obj)
            if new_val != old_val:
                I.set(pred, obj, new_val)
                changed = True

        if not changed:
            return I

    raise RuntimeError("least_model did not converge (max_iters reached)")