"""Explicit candidate spaces and fitting evidence for single-block studies.

Linear fits report numerical rank/condition. Nonlinear fits report best found
solutions, convergence and multiple starts; they never certify global capacity.
There are no zero-crossing-based order choices or silent minimum architectures.
"""
from functools import lru_cache
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.fft import dct,idct
from scipy.optimize import least_squares
from scipy.signal import hilbert
import torch
from .models import _network, fit_mode


def candidates():
    """Declared finite library. Counts, not geometric heuristics, define orders."""
    values=[]
    for degree in (0,1,2,4,8,16,32,64,96,128):
        values.append(dict(family='polynomial',degree=degree))
    for knots in (2,4,8,16,32,64,128,256):
        values.append(dict(family='spline',knots=knots))
    for coefficients in (1,2,4,8,16,32,64,128,256,640):
        values.append(dict(family='dct',coefficients=coefficients))
    for components in (1,2,4,8,16):
        values.append(dict(family='sinusoid',components=components))
    for amplitude_degree in (0,1,2,4):
        for phase_degree in (1,2,4):
            values.append(dict(family='amfm',amplitude_degree=amplitude_degree,phase_degree=phase_degree))
    for family in ('sine_network','snakebeta_network'):
        for width in (1,2,4,8): values.append(dict(family=family,width=width))
    return values


def identifier(spec):
    return '_'.join(str(v) for v in spec.values())


@lru_cache(maxsize=64)
def linear_space(family,order,n):
    t=np.linspace(-1,1,n)
    matrix=np.polynomial.chebyshev.chebvander(t,order) if family=='polynomial' else CubicSpline(np.linspace(-1,1,order),np.eye(order),bc_type='natural')(t)
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    cutoff=max(matrix.shape)*np.finfo(float).eps*s[0]; rank=int(np.sum(s>cutoff))
    inverse=(v[:rank].T/s[:rank])@u[:,:rank].T
    return inverse,rank,float(s[0]/s[-1]),matrix.shape[1]


def decode(model,n):
    """Decoder uses only the serialized family specification and coefficients."""
    spec=model['spec']; p=np.asarray(model['parameters'],float);family=spec['family'];t=np.linspace(-1,1,n)
    if family=='zero': return np.zeros(n)
    if family=='polynomial':return np.polynomial.chebyshev.chebval(t,p)
    if family=='spline':return CubicSpline(np.linspace(-1,1,len(p)),p,bc_type='natural')(t)
    if family=='dct':return idct(np.pad(p,(0,n-len(p))),norm='ortho')
    if family=='sinusoid':
        a,f,phase=p.reshape(-1,3).T
        return np.sum(a[:,None]*np.cos(f[:,None]*t+phase[:,None]),axis=0)
    if family=='amfm':
        split=spec['amplitude_degree']+1
        return np.maximum(0,np.polynomial.chebyshev.chebval(t,p[:split]))*np.cos(np.polynomial.chebyshev.chebval(t,p[split:]))
    with torch.no_grad():
        return _network(torch.tensor(p),torch.tensor(t),family.startswith('snake')).numpy()


def fit(x,spec,*,restarts=3,max_steps=1600,seed=42):
    """Return a finite candidate plus numerical evidence, not a capacity claim."""
    started=time.perf_counter();x=np.asarray(x,float);n=len(x);t=np.linspace(-1,1,n)
    family=spec['family'];energy=float(x@x);scale=max(float(np.sqrt(np.mean(x*x))),1e-12)
    y=x/scale;history=[];records=[];evidence={};basis_certificate='best_found_only'
    if not np.any(x):
        # Keep the requested model size explicit. The separate zero-cost model
        # is available to the allocator, rather than silently dropping a mode.
        if family=='polynomial':count=spec['degree']+1
        elif family=='spline':count=spec['knots']
        elif family=='dct':count=spec['coefficients']
        elif family=='sinusoid':count=3*spec['components']
        elif family=='amfm':count=spec['amplitude_degree']+spec['phase_degree']+2
        else:
            h=spec['width'];count=h*h+(8 if family.startswith('snake') else 4)*h+1
        return dict(model=dict(spec=spec,parameters=[0.]*count),parameter_count=count,error_energy=0.,nmse=None,
                    evidence='exact_zero_target',linear_diagnostics={},restarts=[],history=[],seconds=time.perf_counter()-started)
    if family in ('polynomial','spline'):
        order=spec.get('degree',spec.get('knots'));inverse,rank,condition,columns=linear_space(family,order,n)
        p=inverse@x;evidence=dict(rank=rank,columns=columns,condition_number=condition)
        basis_certificate='linear_least_squares_full_rank' if rank==columns else 'numerically_truncated_space_only'
    elif family=='dct':
        p=dct(x,norm='ortho')[:spec['coefficients']];basis_certificate='orthonormal_projection_fixed_prefix'
    elif family in ('amfm','sinusoid'):
        if family=='amfm':
            na,np_=spec['amplitude_degree']+1,spec['phase_degree']+1
            va=np.polynomial.chebyshev.chebvander(t,na-1);vp=np.polynomial.chebyshev.chebvander(t,np_-1)
            analytic=hilbert(y);phase=np.unwrap(np.angle(analytic));amplitude=np.abs(analytic)
            interior=(np.abs(t)<.85);weights=np.sqrt(np.maximum(amplitude[interior],.01))
            initial=np.r_[np.linalg.lstsq(va[interior],amplitude[interior],rcond=None)[0],
                          np.linalg.lstsq(vp[interior]*weights[:,None],phase[interior]*weights,rcond=None)[0]]
            def evaluate(p):
                amp=va@p[:na];ph=vp@p[na:];return np.maximum(amp,0)*np.cos(ph)
            def jacobian(p):
                amp=va@p[:na];ph=vp@p[na:]
                return np.column_stack([va*((amp>0)*np.cos(ph))[:,None],vp*(-np.maximum(amp,0)*np.sin(ph))[:,None]])
            starts=[initial]
            dominant=np.argmax(np.abs(np.fft.rfft(y)))*np.pi*(n-1)/n
            alternate=np.zeros(na+np_);alternate[0]=np.sqrt(2);alternate[na+1]=dominant
            starts.append(alternate)
        else:
            initial=fit_mode(y,'sinusoid',3*spec['components']).parameters
            if len(initial)!=3*spec['components']:raise ValueError('Sinusoidal initializer returned wrong order')
            def evaluate(p):
                a,f,ph=p.reshape(-1,3).T
                return np.sum(a[:,None]*np.cos(f[:,None]*t+ph[:,None]),axis=0)
            def jacobian(p):
                a,f,ph=p.reshape(-1,3).T;z=f[:,None]*t+ph[:,None]
                return np.stack([np.cos(z),-a[:,None]*t*np.sin(z),-a[:,None]*np.sin(z)],axis=2).transpose(1,0,2).reshape(n,-1)
            starts=[initial]
        rng=np.random.default_rng(seed)
        while len(starts)<restarts:
            starts.append(initial+rng.normal(0,.15,len(initial)))
        best_loss=float('inf')
        for ri,start in enumerate(starts[:restarts]):
            counter=[0]
            def residual(p):
                e=evaluate(p)-y;counter[0]+=1
                if counter[0]==1 or counter[0]%10==0:history.append(dict(restart=ri,step=counter[0],nmse=float(e@e/n)))
                return e
            result=least_squares(residual,start,jac=jacobian,max_nfev=max_steps,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            loss=float(np.mean((evaluate(result.x)-y)**2))
            history.append(dict(restart=ri,step=counter[0],nmse=loss))
            records.append(dict(restart=ri,evaluations=int(result.nfev),termination=str(result.message),local_converged=bool(result.success),nmse=loss))
            if loss<best_loss:best_loss=loss;p=result.x.copy()
        if family=='amfm':p[:na]*=scale
        else:p[::3]*=scale
    else:
        h=spec['width'];snake=family.startswith('snake');count=h*h+(8 if snake else 4)*h+1
        target=torch.tensor(y,dtype=torch.float64);tt=torch.tensor(t,dtype=torch.float64)
        best_loss=float('inf');dominant=np.argmax(np.abs(np.fft.rfft(y)))*np.pi*(n-1)/n
        for ri in range(restarts):
            rng=np.random.default_rng(seed+ri);initial=rng.normal(0,.2,count)
            initial[:h]=np.linspace(max(1,dominant*.5),max(2,dominant*1.5),h)
            if snake:initial[-4*h:]=0
            param=torch.tensor(initial,dtype=torch.float64,requires_grad=True)
            optimizer=torch.optim.Adam([param],lr=.01);local_best=float('inf');last_improvement=0
            for step in range(max_steps+1):
                optimizer.zero_grad();prediction=_network(param,tt,snake);loss=(prediction-target).square().mean()
                if not torch.isfinite(loss):break
                value=float(loss.detach())
                if step%20==0:history.append(dict(restart=ri,step=step,nmse=value))
                if value<best_loss:best_loss=value;p=param.detach().numpy().copy()
                if value<local_best*(1-1e-4):local_best=value;last_improvement=step
                if step>=400 and step-last_improvement>=250:break
                loss.backward();torch.nn.utils.clip_grad_norm_([param],10);optimizer.step()
            records.append(dict(restart=ri,steps=step,termination='plateau' if step<max_steps else 'step_limit',local_converged=False,nmse=local_best))
        offset=h*h+3*h;p[offset:offset+h+1]*=scale
    if not np.isfinite(p).all():raise FloatingPointError('Nonfinite parameters')
    model=dict(spec=spec,parameters=np.asarray(p,float).tolist());reconstruction=decode(model,n)
    error=float(np.sum((x-reconstruction)**2))
    return dict(model=model,parameter_count=len(p),error_energy=error,nmse=error/energy if energy>1e-24 else None,
                evidence=basis_certificate,linear_diagnostics=evidence,restarts=records,history=history,
                seconds=time.perf_counter()-started)
