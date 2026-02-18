from wcs.tv import TV, l_and, l_implies, l_not

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
    o1 = 0
    o2 = 0
    for x in I.domain:
        a = I.get(A, x)
        b = I.get(B, x)
        if a.value == 1:
            if b.value == 1 and o1 < 1:
                o1 = 1
            else:
                o2 = 1
    return TV(l_and(TV(o1), TV(o2)).value)

def some_A_are_not_B(I, A, B):
    o1 = 0
    o2 = 0
    for x in I.domain:
        a = I.get(A, x)
        b = I.get(B, x)
        if a.value == 1:
            if b.value == 0 and o1 < 1:
                o1 = 1
            else:
                o2 = 1
    return TV(l_and(TV(o1), TV(o2)).value)