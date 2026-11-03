"""Exact finite-library allocation for a conservative whole-block error bound."""
import numpy as np
from .isolated_models import decode


def select(candidates_by_mode,reference,tolerances=(.1,.03,.01,.003,.001)):
    """Minimize parameter count under sum-of-error-norms bounds using DP.

    Triangle inequality bounds ||sum errors|| by sum ||error||. The dynamic
    program is exact over available fitted candidates for this conservative
    proxy, not a global optimum for true summed-waveform/perceptual distortion.
    Input components must sum to reference, with any decomposition residual
    represented as an ordinary component. Missing fits are never fabricated.
    """
    candidates=[]
    for group in candidates_by_mode:
        best={}
        for c in group:
            cost=c['parameter_count']
            if cost not in best or c['error_energy']<best[cost]['error_energy']:best[cost]=c
        candidates.append(list(best.values()))
    limit=sum(max(c['parameter_count'] for c in group) for group in candidates)
    state=np.full(limit+1,np.inf);state[0]=0;back=[]
    for group in candidates:
        updated=np.full_like(state,np.inf);choice=np.full(limit+1,-1,int)
        for ci,c in enumerate(group):
            cost=c['parameter_count'];end=limit+1-cost
            values=state[:end]+np.sqrt(c['error_energy']);mask=values<updated[cost:]
            updated[cost:][mask]=values[mask];choice[cost:][mask]=ci
        state=updated;back.append(choice)
    reference_energy=float(reference@reference);rows=[]
    for tol in tolerances:
        feasible=np.flatnonzero(state<=np.sqrt(tol*reference_energy))
        if not len(feasible):
            rows.append(dict(tolerance=tol,status='not_achieved_by_available_fits'));continue
        cost=int(feasible[0]);remaining=cost;selected=[]
        for group,choice in zip(reversed(candidates),reversed(back)):
            chosen=group[choice[remaining]];selected.append(chosen);remaining-=chosen['parameter_count']
        selected.reverse();models=[c['model'] for c in selected]
        waveform=sum((decode(m,len(reference)) for m in models),np.zeros_like(reference))
        rows.append(dict(tolerance=tol,status='achieved',parameter_count=cost,
                         conservative_nmse_bound=float(state[cost]**2/reference_energy),
                         nmse=float(np.sum((reference-waveform)**2)/reference_energy),models=models,
                         criterion='minimum parameter count among fitted candidates under triangle bound'))
    return rows
