"""Equal-budget expression search and stronger refinement of its discoveries."""
import numpy as np
from .expressions import fit_expression,frontier
from .model import random_expression


def search(y,model,*,seed,candidates=24,budgets=(7,15,31),restarts=2,max_nfev=96,refine=2):
    """Count every proposal, including duplicates/invalid fits, against budget.

All arms use the same constant optimizer, requested node-budget schedule and
final refinement allowance. Duplicate proposals reuse a fit but do not receive
free extra attempts. This makes mode collapse visible in the search logs.
"""
    rng=np.random.default_rng(seed);gain=max(float(np.sqrt(np.mean(y*y))),1e-12);normal=y/gain
    programs=[]
    for i in range(candidates):
        budget=budgets[i%len(budgets)]
        tokens=random_expression(rng,budget) if model is None else model.propose(normal,budget,1,rng)[0]
        programs.append(tokens)
    records=[];seen=set();invalid=0
    for i,tokens in enumerate(programs):
        key=tuple(tokens)
        if key in seen:continue
        seen.add(key)
        try:
            fit=fit_expression(tokens,y,seed=seed+i*17,restarts=restarts,max_nfev=max_nfev)
            predicted=fit.waveform(np.linspace(-1,1,len(y)))
            magnitude=np.abs(np.fft.rfft(y));error=np.abs(np.fft.rfft(predicted))-magnitude
            spectral=float(np.sum(error**2)/max(np.sum(magnitude**2),1e-24))
            records.append(dict(**fit.record(),spectral_nmse=spectral,proposal=i,budget=budgets[i%len(budgets)],refined=False))
        except (ValueError,FloatingPointError,OverflowError):invalid+=1
    # Refine several frontier expressions, not just the unconstrained best fit.
    selected=sorted(frontier(records),key=lambda r:r['nmse'])[:refine]
    for i,row in enumerate(selected):
        fit=fit_expression(row['tokens'],y,seed=seed+100000+i,restarts=restarts+1,
                           max_nfev=max_nfev*4,initial=row['constants'])
        predicted=fit.waveform(np.linspace(-1,1,len(y)))
        magnitude=np.abs(np.fft.rfft(y));error=np.abs(np.fft.rfft(predicted))-magnitude
        spectral=float(np.sum(error**2)/max(np.sum(magnitude**2),1e-24))
        records.append(dict(**fit.record(),spectral_nmse=spectral,proposal=row['proposal'],budget=row['budget'],refined=True))
    return dict(records=records,frontier=frontier(records),proposals=candidates,unique=len(seen),
                invalid_fits=invalid,refinements=len(selected))


def reward(record):
    """Declared training preference; final evaluation preserves both axes."""
    return float(-np.log10(max(record['nmse'],1e-8))-.0005*record['description_bits'])
