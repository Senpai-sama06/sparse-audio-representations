"""Geometry-guided alternatives to the frozen finite rotation-word optimizer.

All methods obey the same depth, precision and reconstruction rules as the exact
oracle. Only branch_bound claims completion-based global optimality. Heuristics
may certify a solution when their witness meets the universal root lower bound.
"""
from collections import defaultdict
from fractions import Fraction as F
import heapq
from itertools import combinations
from math import atan2,cos,hypot,sin
from time import perf_counter
from lrac.sparse_search import rational,transform


def required_support(values, eta):
    """Return the exact best-k error curve and least feasible support."""
    values=tuple(values)
    order=tuple(sorted(range(len(values)),key=lambda i:(-values[i]*values[i],i)))
    energy=sum(v*v for v in values)
    tails=[energy]
    for i in order:
        tails.append(tails[-1]-values[i]**2)
    target=eta*energy
    k=next(k for k,error in enumerate(tails) if error<=target)
    return dict(a=values,k=k,tails=tuple(tails),order=order)


def lower_support_bound(k,remaining,energy,eta):
    """A pairwise orthogonal rotation can improve required support by at most 1."""
    floor=0 if energy==0 or eta>=1 else 1
    return max(floor,k-remaining)


def cancellation_actions(values):
    """Construct immediate exact-zero actions from value-ratio hash lookups."""
    lookup=defaultdict(list)
    for j,v in enumerate(values):
        if v:
            lookup[v].append(j)
    actions=set()
    for i,a in enumerate(values):
        if not a:
            continue
        for sign in (-1,1):
            # Inverse rotation: a'=(3a+4 sign b)/5, b'=(-4 sign a+3b)/5.
            for b in (F(4*sign,3)*a,-F(3,4*sign)*a):
                for j in lookup.get(b,()):
                    if j>i:
                        actions.add((i,j,sign))
    return sorted(actions)


def _quality(state):
    k=state["k"]
    return (k,state["tails"][k-1] if k else F(0),state["tails"][k])


def _key(state):
    return (*_quality(state),len(state["word"]),state["word"])


def solve(x,eta,*,method,max_depth=2,width=8,coefficient_grid=625,coefficient_bound=8):
    """Find a feasible sparse code with the selected deterministic strategy."""
    start=perf_counter()
    x=tuple(rational(v) for v in x)
    eta=rational(eta)
    energy=sum(v*v for v in x)
    if method not in ("ratio_greedy","tail_greedy","beam","branch_bound"):
        raise ValueError("Unknown method")
    if len(x)<2 or eta<0 or max_depth<0 or width<1:
        raise ValueError("Invalid search dimensions, tolerance, depth or width")
    if energy>coefficient_bound**2:
        raise ValueError("Input exceeds coefficient norm bound")
    actions=tuple((i,j,s) for i,j in combinations(range(len(x)),2) for s in (-1,1))
    stats=dict(candidate_states=1,scored_states=0,rotation_updates=0,bound_pruned=0,
               duplicate_pruned=0,width_discarded=0)

    def state(a,word):
        if any((v*coefficient_grid).denominator!=1 or abs(v)>coefficient_bound for v in a):
            raise ValueError("Candidate outside declared coefficient alphabet")
        stats["scored_states"]+=1
        return dict(**required_support(a,eta),word=word)

    def values(parent,op):
        stats["candidate_states"]+=1
        stats["rotation_updates"]+=1
        return transform(parent["a"],(op,),inverse=True)

    def child(parent,op):
        return state(values(parent,op),(op,)+parent["word"])

    root=state(x,())
    best=root
    root_lower=lower_support_bound(root["k"],max_depth,energy,eta)
    complete=False
    if best["k"]==root_lower:
        complete=True
    elif method in ("ratio_greedy","tail_greedy"):
        current=root
        for _ in range(max_depth):
            choices=cancellation_actions(current["a"]) if method=="ratio_greedy" else actions
            candidates={}
            for op in choices:
                a=values(current,op)
                if a in candidates:
                    stats["duplicate_pruned"]+=1
                    continue
                candidates[a]=state(a,(op,)+current["word"])
            if not candidates:
                break
            proposal=min(candidates.values(),key=_key)
            if _quality(proposal)>=_quality(current):
                break
            current=proposal
            if _key(current)<_key(best):
                best=current
            if best["k"]==root_lower:
                break
    elif method=="beam":
        frontier=[root]
        for depth in range(1,max_depth+1):
            candidates={}
            for parent in frontier:
                remaining=max_depth-len(parent["word"])
                if lower_support_bound(parent["k"],remaining,energy,eta)>=best["k"]:
                    stats["bound_pruned"]+=1
                    continue
                for op in actions:
                    a=values(parent,op)
                    if a in candidates:
                        stats["duplicate_pruned"]+=1
                        continue
                    candidate=state(a,(op,)+parent["word"])
                    candidates[a]=candidate
                    if _key(candidate)<_key(best):
                        best=candidate
            if best["k"]==root_lower:
                break
            ordered=sorted(candidates.values(),key=_key)
            stats["width_discarded"]+=max(0,len(ordered)-width)
            frontier=ordered[:width]
            if not frontier:
                break
    elif method=="branch_bound":
        serial=0
        queue=[(root_lower,_key(root),serial,root)]
        seen={root["a"]:0}
        while queue and best["k"]>root_lower:
            bound,_,_,parent=heapq.heappop(queue)
            if bound>=best["k"]:
                stats["bound_pruned"]+=1
                continue
            depth=len(parent["word"])+1
            for op in actions:
                a=values(parent,op)
                if seen.get(a,max_depth+1)<=depth:
                    stats["duplicate_pruned"]+=1
                    continue
                seen[a]=depth
                candidate=state(a,(op,)+parent["word"])
                if _key(candidate)<_key(best):
                    best=candidate
                bound=lower_support_bound(candidate["k"],max_depth-depth,energy,eta)
                if bound<best["k"]:
                    serial+=1
                    heapq.heappush(queue,(bound,_key(candidate),serial,candidate))
                else:
                    stats["bound_pruned"]+=1
                if best["k"]==root_lower:
                    break
        complete=True  # All improving branches exhausted or witness meets root bound.
    else:
        raise ValueError("Unknown method")
    k=best["k"]
    keep=set(best["order"][:k])
    z=tuple(v if i in keep else F(0) for i,v in enumerate(best["a"]))
    y=transform(z,best["word"])
    error=sum((a-b)**2 for a,b in zip(x,y))
    if error!=best["tails"][k] or error>eta*energy:
        raise AssertionError("Inconsistent reconstruction witness")
    certified=complete or k==root_lower
    return dict(method=method,k=k,word=best["word"],z=z,error_sq=error,energy=energy,
        root_lower_bound=root_lower,certified=certified,
        lower_bound=k if certified else root_lower,upper_bound=k,
        seconds=perf_counter()-start,**stats)


def hankel_rank(x):
    """Exact rank of the 4x5 Hankel matrix for an eight-sample diagnostic."""
    x=tuple(rational(v) for v in x)
    if len(x)!=8:
        raise ValueError("This diagnostic expects eight samples")
    matrix=[[x[i+j] for j in range(5)] for i in range(4)]
    rank=0
    for column in range(5):
        pivot=next((i for i in range(rank,4) if matrix[i][column]),None)
        if pivot is None:
            continue
        matrix[rank],matrix[pivot]=matrix[pivot],matrix[rank]
        scale=matrix[rank][column]
        matrix[rank]=[v/scale for v in matrix[rank]]
        for i in range(4):
            if i!=rank:
                scale=matrix[i][column]
                matrix[i]=[a-scale*b for a,b in zip(matrix[i],matrix[rank])]
        rank+=1
        if rank==4:
            break
    return rank


def construct_continuous(x,eta,rotations):
    """Ideal continuous-angle optimum, illustrated numerically in float64.

    This changes the admissible family: angle/coefficients need not belong to the
    frozen rational alphabet. Mathematical error is the original q-term tail;
    observed error additionally includes floating-point roundoff.
    """
    x=tuple(rational(v) for v in x)
    eta=rational(eta)
    if eta<0 or rotations<0:
        raise ValueError("Invalid tolerance or rotation budget")
    initial=required_support(x,eta)
    q=initial["k"]
    used=min(rotations,max(0,q-1))
    a=[float(v) for v in x]
    instructions=[]
    retained=[]
    if q:
        pivot=initial["order"][0]
        for j in initial["order"][1:used+1]:
            angle=atan2(a[j],a[pivot])
            a[pivot]=hypot(a[pivot],a[j])
            a[j]=0.
            instructions.append((pivot,j,angle))
        retained=[pivot,*initial["order"][used+1:q]]
    z=[v if i in retained else 0. for i,v in enumerate(a)]
    y=z.copy()
    for i,j,angle in reversed(instructions):
        c,s=cos(angle),sin(angle)
        a,b=y[i],y[j]
        y[i],y[j]=c*a-s*b,s*a+c*b
    observed_error=sum((float(a)-b)**2 for a,b in zip(x,y))
    return dict(k=q-used,q=q,rotations=used,instructions=instructions,z=z,y=y,
        angle_parameters=used,scalar_values_and_angles=q,
        ideal_error_sq=initial["tails"][q],observed_error_sq=observed_error)
