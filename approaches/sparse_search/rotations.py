"""Exact joint basis/support search in a deliberately bounded rational family.

Each domain is a word of plane rotations with cosine 3/5 and sine +/-4/5.
The family is specified independently of the target. Orthogonality eliminates
the coefficient/support subproblem analytically; all permitted words are searched.
This is a tiny conditional global oracle, not an all-domain audio optimizer.
"""
from fractions import Fraction
from itertools import combinations, product


def rational(value):
    if isinstance(value, float):
        raise TypeError("Use integers, Fractions or decimal strings, not binary floats")
    return Fraction(value)


def transform(values, word, *, inverse=False):
    """Apply Q_word or Q_word.T exactly; the decoder needs only word and values."""
    result = [rational(v) for v in values]
    word = tuple(word)
    steps = [(i,j,-sign) for i,j,sign in reversed(word)] if inverse else word
    c = Fraction(3,5)
    for i,j,sign in steps:
        if not 0 <= i < j < len(result) or sign not in (-1,1):
            raise ValueError("Invalid rotation instruction")
        s = Fraction(4*sign,5)
        a,b = result[i],result[j]
        result[i],result[j] = c*a-s*b,s*a+c*b
    return tuple(result)


def words(dimension, max_depth):
    if dimension < 2 or max_depth < 0:
        raise ValueError("Expected dimension >=2 and nonnegative maximum depth")
    actions = [(i,j,sign) for i,j in combinations(range(dimension),2)
               for sign in (-1,1)]
    yield ()
    for depth in range(1,max_depth+1):
        yield from product(actions,repeat=depth)


def search(x, *, max_depth=2, coefficient_grid=625, coefficient_bound=8):
    """Minimum squared-error curves for EVERY support budget and allowed depth.

    Preconditions ensure each exact Q.T@x is itself in the finite coefficient
    alphabet. Thus real-valued orthogonal best-k solutions are also admissible
    finite-precision solutions. Unsupported inputs fail rather than silently
    dropping a candidate domain from the claimed global search.
    """
    x = tuple(rational(v) for v in x)
    n = len(x)
    energy = sum(v*v for v in x)
    bound = rational(coefficient_bound)
    if coefficient_grid <= 0 or bound <= 0 or energy > bound*bound:
        raise ValueError("Invalid coefficient alphabet or input exceeds norm bound")
    best = [[None]*(n+1) for _ in range(max_depth+1)]
    candidates = 0
    for word in words(n,max_depth):
        candidates += 1
        a = transform(x,word,inverse=True)
        if sum(v*v for v in a) != energy:
            raise AssertionError("Rotation did not preserve energy exactly")
        if any((v*coefficient_grid).denominator != 1 or abs(v)>bound for v in a):
            raise ValueError("A candidate projection is outside the declared coefficient alphabet")
        order = sorted(range(n),key=lambda i:(-a[i]*a[i],i))
        tail = energy
        for k in range(n+1):
            if k:
                tail -= a[order[k-1]]**2
            for depth in range(len(word),max_depth+1):
                previous = best[depth][k]
                if previous is None or tail < previous["error_sq"]:
                    keep = set(order[:k])
                    z = tuple(v if i in keep else Fraction(0) for i,v in enumerate(a))
                    best[depth][k] = dict(error_sq=tail,word=word,z=z,
                        nonzeros=sum(v!=0 for v in z))
    for curve in best:
        for entry in curve:
            y = transform(entry["z"],entry["word"])
            if sum((a-b)**2 for a,b in zip(x,y)) != entry["error_sq"]:
                raise AssertionError("Witness reconstruction failed")
    return dict(x=x,energy=energy,curves=best,domain_words=candidates)


def at_tolerance(result, nmse_limit, max_depth):
    """Return a witness and an exact lower bound ruling out every smaller support."""
    eta = rational(nmse_limit)
    if eta < 0:
        raise ValueError("NMSE limit must be nonnegative")
    target = eta*result["energy"]
    curve = result["curves"][max_depth]
    for k,entry in enumerate(curve):
        if entry["error_sq"] <= target:
            if entry["nonzeros"] != k:
                raise AssertionError("First feasible budget should equal actual support")
            return dict(k=k,**entry,target_error_sq=target,
                smaller_support_min_error_sq=curve[k-1]["error_sq"] if k else None,
                certified_lower_bound=k,certified_upper_bound=k)
    return None
