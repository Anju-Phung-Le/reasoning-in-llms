from wcs.encoders import build_program_for_form
from wcs.leastmodel import least_model

def all_forms():
    moods = ["A","E","I","O"]
    return [f"{m1}{m2}{fig}" for m1 in moods for m2 in moods for fig in [1,2,3,4]]

for form in all_forms():
    P, dom = build_program_for_form(form)
    I = least_model(P, domain=dom)
print("Built + solved all 64 syllogisms!")