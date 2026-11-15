"""Three-valued evaluation of the eight candidate conclusions against a least model.

The Yes conditions implement the entailment definitions of Costa, Dietz, Hoelldobler
and Ragni, "Syllogistic Reasoning under the Weak Completion Semantics" (Bridging 2016),
Section 5, in their simplified (|=') form:

  A: there is an object entailed to be in A, and every object entailed to be in A
     is entailed to be in B. Objects whose A-membership is unknown are not counted.
  E: as A, with "entailed to be outside B" in place of "entailed to be in B".
  I: one object entailed to be in both A and B, and a second object entailed to be
     in A whose B-membership is NOT entailed (the principle of unknown generalization,
     so that "some" is never concluded when "all" already holds).
  O: one object entailed to be in A and outside B, and a second object entailed to be
     in A for which "outside B" is NOT entailed.

The No and Cannot-Determine levels extend the paper's two-valued entailment to the
benchmark's three answer options: No when the least model definitely refutes the
statement (a counterexample for the universals; every known A-object refuting the
witness for the particulars), and also when no object is or may be in A (failed
existential import). Cannot Determine otherwise.
"""

from .tv import TV


def _split(I, A, B):
    """Return (B-values of objects entailed to be in A, whether any object's A-membership is unknown)."""
    a_true = []
    a_unknown = False
    for x in I.domain:
        a = I.get(A, x)
        if a == TV.TRUE:
            a_true.append(I.get(B, x))
        elif a == TV.UNKNOWN:
            a_unknown = True
    return a_true, a_unknown


def all_A_are_B(I, A, B):
    a_true, a_unknown = _split(I, A, B)
    if a_true and all(b == TV.TRUE for b in a_true):
        return TV.TRUE
    if any(b == TV.FALSE for b in a_true):
        return TV.FALSE
    if not a_true and not a_unknown:
        return TV.FALSE
    return TV.UNKNOWN


def no_A_are_B(I, A, B):
    a_true, a_unknown = _split(I, A, B)
    if a_true and all(b == TV.FALSE for b in a_true):
        return TV.TRUE
    if any(b == TV.TRUE for b in a_true):
        return TV.FALSE
    if not a_true and not a_unknown:
        return TV.FALSE
    return TV.UNKNOWN


def some_A_are_B(I, A, B):
    a_true, a_unknown = _split(I, A, B)
    if any(b == TV.TRUE for b in a_true) and any(b != TV.TRUE for b in a_true):
        return TV.TRUE
    if a_true and all(b == TV.FALSE for b in a_true):
        return TV.FALSE
    if not a_true and not a_unknown:
        return TV.FALSE
    return TV.UNKNOWN


def some_A_are_not_B(I, A, B):
    a_true, a_unknown = _split(I, A, B)
    if any(b == TV.FALSE for b in a_true) and any(b != TV.FALSE for b in a_true):
        return TV.TRUE
    if a_true and all(b == TV.TRUE for b in a_true):
        return TV.FALSE
    if not a_true and not a_unknown:
        return TV.FALSE
    return TV.UNKNOWN
