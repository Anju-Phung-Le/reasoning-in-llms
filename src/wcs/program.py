from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple
from .tv import TV, l_or

Atom = Tuple[str, str]  # (predicate, object)
BodyFn = Callable[["InterpretationLike"], TV]

class InterpretationLike:
    """
    Just a simple interface for body functions: supports get(pred,obj)->TV 
    Using Interpretation class from tv.py.
    """
    def get(self, pred: str, obj: str) -> TV: 
        raise NotImplementedError

@dataclass
class Program:
    """
    A tiny WCS program that stores all rules in the logic program.
    head atom -> list of body functions.
    Multiple bodies for the same head represent multiple rules with that head.
    """
    defs: Dict[Atom, List[BodyFn]]

    def __init__(self) -> None:
        self.defs = {} # dictionary that stores rules: head -> list of body functions

    # add a rule to the program, e.g., z(X) <- y(X) ∧ ¬ab_yz(X) would be add_rule("z", "X", body_fn)
    def add_rule(self, head_pred: str, head_obj: str, body: BodyFn) -> None: # e.g., ("c","o1") → [body1, body2]
        head = (head_pred, head_obj)
        self.defs.setdefault(head, []).append(body)

    # return all head atoms (predicate, object pairs)
    def heads(self) -> Set[Atom]:
        return set(self.defs.keys())

    # extract all objects appearing in heads, which is the domain
    def domain(self) -> Set[str]:
        return {obj for (_, obj) in self.defs.keys()}

    # merges all the rules with the same head into a single rule using OR
    def weak_completion(self) -> "Program":
        """
        Weak completion (syllogism-specific):
        merge multiple rules with the same head into a single definition using OR.
        In callable form: body_wc(I) = OR(body1(I), body2(I), ...).
        """
        wc = Program()
        for (pred, obj), bodies in self.defs.items():
            if len(bodies) == 1:
                wc.add_rule(pred, obj, bodies[0])
            else:
                def body_wc(I, bodies=bodies):
                    v = TV.FALSE
                    for b in bodies:
                        v = l_or(v, b(I))
                    return v
                wc.add_rule(pred, obj, body_wc)
        return wc