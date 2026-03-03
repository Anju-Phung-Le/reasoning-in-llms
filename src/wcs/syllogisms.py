from .tv import TV, l_and, l_implies, l_not, l_or
def all_A_are_B(I, A, B):
    values = []
    o1 = 0
    for x in I.domain:
        a = I.get(A, x)
        b = I.get(B, x)
        values.append(l_implies(a, b))
        if o1 < 1: o1 = max(o1, a.value) 
    return TV(min(min(v.value for v in values), o1))

def no_A_are_B(I, A, B):
    values = []
    o1 = 0
    for x in I.domain:
        a = I.get(A, x)
        b = I.get(B, x)
        values.append(l_implies(a, l_not(b)))
        if o1 < 1: o1 = max(o1, a.value) 
    return TV(min(min(v.value for v in values), o1))

def some_A_are_B(I, A, B):
    saw_A_true = False
    saw_unknown_witness = False

    for x in I.domain:
        a = I.get(A, x)
        b = I.get(B, x)
        if a == TV.TRUE:
            saw_A_true = True
            if b == TV.TRUE:
                return TV.TRUE
            if b == TV.UNKNOWN:
                saw_unknown_witness = True

    if saw_unknown_witness:
        return TV.UNKNOWN
    if saw_A_true:
        return TV.FALSE

    # If A is never TRUE but could be UNKNOWN, existential is UNKNOWN
    for x in I.domain:
        if I.get(A, x) == TV.UNKNOWN:
            return TV.UNKNOWN
    return TV.FALSE


def some_A_are_not_B(I, A, B):
    saw_A_true = False
    saw_unknown_witness = False

    for x in I.domain:
        a = I.get(A, x)
        b = I.get(B, x)
        if a == TV.TRUE:
            saw_A_true = True
            if b == TV.FALSE:
                return TV.TRUE
            if b == TV.UNKNOWN:
                saw_unknown_witness = True

    if saw_unknown_witness:
        return TV.UNKNOWN
    if saw_A_true:
        return TV.FALSE

    for x in I.domain:
        if I.get(A, x) == TV.UNKNOWN:
            return TV.UNKNOWN
    return TV.FALSE