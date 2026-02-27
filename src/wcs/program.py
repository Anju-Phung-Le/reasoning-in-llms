from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple
from .tv import TV, l_or

Atom = Tuple[str, str]  # (predicate, object)
BodyFn = Callable[["InterpretationLike"], TV]

class InterpretationLike:
    """
    Minimal protocol for body functions: must support get(pred,obj)->TV.
    Using Interpretation class from tv.py.
    """
    def get(self, pred: str, obj: str) -> TV: 
        raise NotImplementedError

@dataclass
class Program:
    """
    A tiny WCS program store: head atom -> list of body functions.
    Multiple bodies for the same head represent multiple rules with that head.
    """
    defs: Dict[Atom, List[BodyFn]]

    def __init__(self) -> None:
        self.defs = {}

    def add_rule(self, head_pred: str, head_obj: str, body: BodyFn) -> None:
        head = (head_pred, head_obj)
        self.defs.setdefault(head, []).append(body)

    def heads(self) -> Set[Atom]:
        return set(self.defs.keys())

    def domain(self) -> Set[str]:
        return {obj for (_, obj) in self.defs.keys()}

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