from enum import Enum

# Three-valued logic
class TV(Enum):
    TRUE = 1
    FALSE = 0
    UNKNOWN = 0.5


# Simple logical operations
def l_not(v):
    return TV(1.0-v.value)

def l_and(a, b):
    return TV(min(a.value, b.value))

def l_or(a, b):
    return TV(max(a.value, b.value))

def l_implies(a ,b):
    return TV(min(1.0, 1.0 - a.value + b.value))


# Interpretation with given domain
class Interpretation:
    def __init__(self, domain):
        self.domain = domain
        self.predicates = {} 

    def set(self, pred, obj, value):
        self.predicates.setdefault(pred, {})[obj] = value

    def get(self, pred, obj):
        return self.predicates.get(pred, {}).get(obj, TV.UNKNOWN)