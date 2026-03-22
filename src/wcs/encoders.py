from .program import Program
from .tv import TV, l_and, l_not


# Rules of bodies, mainly section 2 TODO: check other sections
def atom(pred: str, obj: str):
    return lambda I: I.get(pred, obj)

def TOP():
    return lambda I: TV.TRUE

def BOT():
    return lambda I: TV.FALSE

def AND(*bs):
    return lambda I: _and_eval(I, bs)

def NOT(b):
    return lambda I: l_not(b(I))

# evaluates the whole AND expressions (bs: list of body statements)
def _and_eval(I, bs):
    v = TV.TRUE
    for b in bs:
        v = l_and(v, b(I))
    return v


def encode_A(P: Program, y: str, z: str, domain, o: str, ab_yz: str):
    """
      PAyz (paper 4.1):
      z(X)  <- y(X) ∧ ¬ab_yz(X)
      ab_yz(X) <- ⊥
      y(o) <- ⊤
    """
    # Gricean Implcature / existential import
    P.add_rule(y, o, TOP())
    # licenses for inferences + abnormality
    for x in domain:
        P.add_rule(ab_yz, x, BOT())
        P.add_rule(z, x, AND(atom(y, x), NOT(atom(ab_yz, x))))

def encode_E(P: Program, y: str, z: str, domain, o: str, ab_y_nz: str, ab_nz_z: str):
    """
    PEyz (paper 4.2) with negation-by-transformation:
      z0(X) <- y(X) ∧ ¬ab_y¬z(X)
      ab_y¬z(X) <- ⊥
      z(X) <- ¬z0(X) ∧ ¬ab_¬z z(X)
      y(o) <- ⊤
      ab_¬zz(o) <- ⊥   (restricted to o to avoid double negation effects)
    """
    z0 = f"{z}0"  # auxiliary formula for negation-by-transformation

    P.add_rule(y, o, TOP())
    P.add_rule(ab_y_nz, o, BOT())
    for x in domain:
        P.add_rule(z0, x, AND(atom(y, x), NOT(atom(ab_y_nz, x))))
        P.add_rule(z,  x, AND(NOT(atom(z0, x)), NOT(atom(ab_nz_z, x))))

    # restricted abnormality clause (only for the existential object o)
    P.add_rule(ab_nz_z, o, BOT())

def encode_I(P: Program, y: str, z: str, domain, o1: str, o2: str, ab_yz: str):
    """
    PIyz (paper 4.3):
      z(X) <- y(X) ∧ ¬ab_yz(X)
      ab_yz(o1) <- ⊥
      y(o1) <- ⊤
      y(o2) <- ⊤
    Note: NO rule for ab_yz(o2) -> keeps z(o2) UNKNOWN (unknown generalization).
    """
    P.add_rule(y, o1, TOP())
    P.add_rule(y, o2, TOP())
    P.add_rule(ab_yz, o1, BOT())
    for x in domain:
        P.add_rule(z, x, AND(atom(y, x), NOT(atom(ab_yz, x))))


def encode_O(P: Program, y: str, z: str, domain, o1: str, o2: str, ab_y_nz: str, ab_nz_z: str):
    """
    POyz (paper 4.4):
      z0(X) <- y(X) ∧ ¬ab_y¬z(X)
      ab_y¬z(o1) <- ⊥        (restricted)
      z(X) <- ¬z0(X) ∧ ¬ab_¬z z(X)
      y(o1) <- ⊤             (Gricean implicature/ existential import)
      y(o2) <- ⊤             (unknown generalization)
      ab_¬z z(o1) <- ⊥        (restricted to o1,o2)
      ab_¬z z(o2) <- ⊥
    """
    z0 = f"{z}0" # auxiliary formula for negation-by-transformation

    P.add_rule(y, o1, TOP())
    P.add_rule(y, o2, TOP())

    # restricted abnormality for y¬z (only o1)
    P.add_rule(ab_y_nz, o1, BOT())

    for x in domain:
        P.add_rule(z0, x, AND(atom(y, x), NOT(atom(ab_y_nz, x))))
        P.add_rule(z,  x, AND(NOT(atom(z0, x)), NOT(atom(ab_nz_z, x))))

    # restricted ab_¬z z for o1 and o2 (avoid double negation)
    P.add_rule(ab_nz_z, o1, BOT())
    P.add_rule(ab_nz_z, o2, BOT())


def figure_pairs(fig: int):
    """
    Returns (prem1_pair, prem2_pair) where each pair is (subject, predicate).
    Terms are always among {a,b,c}.
    """
    if fig == 1:  
        return (("a", "b"), ("b", "c"))
    if fig == 2:  
        return (("b", "a"), ("c", "b"))
    if fig == 3:  
        return (("a", "b"), ("c", "b"))
    if fig == 4:  
        return (("b", "a"), ("b", "c"))
    raise ValueError(f"Invalid figure: {fig}")


def objs_for_mood(mood: str, start: int):
    """
    A/E need 1 object, I/O need 2 objects.
    Returns (objs, next_start).
    """
    if mood in ("A", "E"):
        return [f"o{start}"], start + 1
    if mood in ("I", "O"):
        return [f"o{start}", f"o{start+1}"], start + 2
    raise ValueError(f"Invalid mood: {mood}")


def _ab_names_for_premise(prem_idx: int, y: str, z: str, mood: str):
    """
    Create abnormality predicate names per premise.
    prem_idx in {1,2} to avoid collisions between premises.
    """
    if mood in ("A", "I"):
        return (f"ab{prem_idx}_{y}{z}",)
    if mood in ("E", "O"):
        # two ab predicates (paper’s y¬z and ¬zz)
        return (f"ab{prem_idx}_{y}n{z}", f"ab{prem_idx}_n{z}{z}")
    raise ValueError(mood)


def add_premise(P: Program, mood: str, y: str, z: str, domain, objs, prem_idx: int):
    """
    Sort out the correct encoder with the right objects/abnormalities.
    """
    if mood == "A":
        (ab_yz,) = _ab_names_for_premise(prem_idx, y, z, mood)
        o = objs[0]
        encode_A(P, y=y, z=z, domain=domain, o=o, ab_yz=ab_yz)
        return

    if mood == "I":
        (ab_yz,) = _ab_names_for_premise(prem_idx, y, z, mood)
        o1, o2 = objs
        encode_I(P, y=y, z=z, domain=domain, o1=o1, o2=o2, ab_yz=ab_yz)
        return

    if mood == "E":
        ab_y_nz, ab_nz_z = _ab_names_for_premise(prem_idx, y, z, mood)
        o = objs[0]
        encode_E(P, y=y, z=z, domain=domain, o=o, ab_y_nz=ab_y_nz, ab_nz_z=ab_nz_z)
        return

    if mood == "O":
        ab_y_nz, ab_nz_z = _ab_names_for_premise(prem_idx, y, z, mood)
        o1, o2 = objs
        encode_O(P, y=y, z=z, domain=domain, o1=o1, o2=o2, ab_y_nz=ab_y_nz, ab_nz_z=ab_nz_z)
        return

    raise ValueError(f"Invalid mood: {mood}")


def build_program_for_form(form: str):
    """
    Build WCS program + domain for any of the 64 forms: {A,E,I,O}{A,E,I,O}{1..4}
    Returns (Program, domain_list).
    """
    if len(form) != 3:
        raise ValueError(f"Expected form like 'OA4', got: {form}")
    
    # parse form into mood1, mood2, figure
    mood1, mood2, fig_ch = form[0], form[1], form[2]
    if fig_ch not in "1234":
        raise ValueError(f"Invalid figure in form: {form}")
    fig = int(fig_ch)

    # use figure pairs
    (y1, z1), (y2, z2) = figure_pairs(fig)

    # rename objects
    idx = 1
    objs1, idx = objs_for_mood(mood1, idx)
    objs2, idx = objs_for_mood(mood2, idx)

    # create domain as union of objects needed for both premises
    domain = objs1 + objs2 

    P = Program()
    add_premise(P, mood=mood1, y=y1, z=z1, domain=domain, objs=objs1, prem_idx=1)
    add_premise(P, mood=mood2, y=y2, z=z2, domain=domain, objs=objs2, prem_idx=2)

    return P, domain

