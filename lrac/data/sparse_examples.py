"""Exact rational toy signals, separate from optimization and experiment output."""
from fractions import Fraction as F
import random

from lrac.sparse_search import transform


def search_examples(seed=0):
    """Seven exact eight-dimensional examples used by the bounded search study."""
    impulse = (F(1),)+(F(0),)*7
    two = (F(1),F(0),F(0),F(2))+(F(0),)*4
    rng = random.Random(seed)
    return dict(
        zero=(F(0),)*8,
        impulse=impulse,
        rotated_one=transform(impulse,((0,1,1),)),
        rotated_two=transform(two,((0,1,1),(1,2,1))),
        dense_rational=tuple(F(v,25) for v in [25,19,-11,7,5,-13,17,23]),
        harmonic_cycle=tuple(F(v) for v in [0,1,0,-1,0,1,0,-1]),
        noise_rational=tuple(F(rng.randint(-25,25),25) for _ in range(8)))


def cue_examples(seed=20261029):
    """The 96-case cue benchmark; the seven historical examples stay fixed."""
    cases=[dict(name=name,group="previous_examples",x=x) for name,x in search_examples().items()]
    cases.append(dict(name="two_step_trap",group="adversarial",
        x=(F(-7,25),F(24,25))+(F(0),)*6))
    cases.append(dict(name="geometric_growth",group="recurrence",x=tuple(F(2**i,25) for i in range(8))))
    cases.append(dict(name="linear_ramp",group="recurrence",x=tuple(F(i,25) for i in range(8))))
    rng=random.Random(seed)
    actions=[(i,j,s) for i in range(8) for j in range(i+1,8) for s in (-1,1)]
    planted=[]
    for k in [1,2,3]:
        for index in range(16):
            z=[F(0)]*8
            for i in rng.sample(range(8),k):
                z[i]=F(rng.choice([-3,-2,-1,1,2,3]))
            x=tuple(z)
            word=[]
            for _ in range(2):
                candidates=[op for op in actions if x[op[0]] or x[op[1]]]
                op=rng.choice(candidates)
                word.append(op)
                x=transform(x,(op,))
            case=dict(name=f"planted_{k}_{index:02d}",group=f"planted_{k}",x=x,
                      generating_word=word,generating_code=z)
            planted.append(case)
            cases.append(case)
    for index in range(18):
        cases.append(dict(name=f"dense_{index:02d}",group="dense",
                          x=tuple(F(rng.randint(-25,25),25) for _ in range(8))))
    for index in range(20):
        parent=rng.choice(planted)
        noise=tuple(F(rng.choice([-1,0,1]),25) for _ in range(8))
        cases.append(dict(name=f"perturbed_{index:02d}",group="perturbed",
                          x=tuple(a+b for a,b in zip(parent["x"],noise)),parent=parent["name"]))
    assert len(cases)==96
    return cases


