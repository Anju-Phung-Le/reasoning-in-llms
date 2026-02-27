# src/wcs/encoders.py

from .program import Program
from typing import Iterable
from .tv import TV, l_and, l_or, l_not


# ============================================================
# Body helpers (callable logic bodies)
# ============================================================

def atom(pred: str, obj: str):
    return lambda I: I.get(pred, obj)


def TOP():
    return lambda I: TV.TRUE


def BOT():
    return lambda I: TV.FALSE


def AND(*bs):
    return lambda I: _and_eval(I, bs)


def OR(*bs):
    return lambda I: _or_eval(I, bs)


def NOT(b):
    return lambda I: l_not(b(I))


def _and_eval(I, bs):
    v = TV.TRUE
    for b in bs:
        v = l_and(v, b(I))
    return v


def _or_eval(I, bs):
    v = TV.FALSE
    for b in bs:
        v = l_or(v, b(I))
    return v


# ============================================================
# Premise encoders (PA / PO minimal versions)
# We start with only what we need for OA4.
# ============================================================

def encode_A(P: Program, y: str, z: str, objs: Iterable[str], ab: str):
    """
    Encode A(y,z): All y are z
    Grounded over the given objs (important!).
      z(o) <- y(o) ∧ ¬ab(o)
      ab(o) <- ⊥
    """
    for o in objs:
        P.add_rule(ab, o, BOT())
        P.add_rule(z, o, AND(atom(y, o), NOT(atom(ab, o))))


def encode_O(P: Program, y: str, z: str, o1: str, o2: str, ab: str):
    """
    Encode O(y,z): Some y are not z

    Minimal witness construction:
    - o1 witnesses y and not z
    - o2 allows generalization
    """

    # witness 1: y true, z false
    P.add_rule(y, o1, TOP())
    P.add_rule(z, o1, BOT())

    # witness 2: y true, z true (keeps model flexible)
    P.add_rule(y, o2, TOP())
    P.add_rule(z, o2, TOP())

    # abnormality defaults
    P.add_rule(ab, o1, BOT())
    P.add_rule(ab, o2, BOT())


# ============================================================
# Form builder (start with OA4 only)
# ============================================================

def build_program_for_form(form: str) -> Program:
    if form != "OA4":
        raise NotImplementedError("Only OA4 implemented so far.")

    P = Program()

    domain = ["o1", "o2", "o3"]

    # Oba: Some b are not a  (creates b(o1)=T and a(o1)=F witness)
    encode_O(P, y="b", z="a", o1="o1", o2="o2", ab="ab1_ba")

    # Abc: All b are c  (must apply to the SAME objects, including o1!)
    encode_A(P, y="b", z="c", objs=domain, ab="ab2_bc")

    return P