"""Safe prefix expressions, exact constant accounting and differentiable fitting.

The grammar specifies elementary operations, not complete function templates.
Every C occurrence is an independent fitted constant. An explicit output gain
normalizes optimization; it is included in parameter and description counts.
No Python eval, source samples or neural weights are needed to evaluate a fit.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.optimize import least_squares


TOKENS = ('t', 'C', '+', '-', '*', 'sin', 'cos', 'tanh', 'square')
ARITY = dict(zip(TOKENS, (0, 0, 2, 2, 2, 1, 1, 1, 1)))


def validate(tokens, max_nodes=127):
    """Reject incomplete/trailing programs and bound recursion/evaluation cost."""
    if not 0 < len(tokens) <= max_nodes:raise ValueError('Expression size outside limits')
    slots=1
    for token in tokens:
        if slots==0 or token not in ARITY:raise ValueError('Invalid prefix expression')
        slots += ARITY[token]-1
    if slots:raise ValueError('Incomplete prefix expression')


def evaluate(tokens, constants, t, jacobian=False):
    """Evaluate a prefix tree and optionally its analytic constant Jacobian."""
    validate(tokens);constants=np.asarray(constants,dtype=np.float64);t=np.asarray(t,dtype=np.float64)
    k=tokens.count('C')
    if len(constants)!=k:raise ValueError('Constant count does not match expression')
    position=0;constant=0
    def visit():
        nonlocal position,constant
        op=tokens[position];position+=1
        if op=='t':return t,np.zeros((len(t),k)) if jacobian else None
        if op=='C':
            value=np.full_like(t,constants[constant]);d=np.zeros((len(t),k)) if jacobian else None
            if jacobian:d[:,constant]=1
            constant+=1;return value,d
        a,da=visit()
        if ARITY[op]==2:
            b,db=visit()
            if op=='+':return a+b,da+db if jacobian else None
            if op=='-':return a-b,da-db if jacobian else None
            return a*b,da*b[:,None]+db*a[:,None] if jacobian else None
        if op=='sin':return np.sin(a),np.cos(a)[:,None]*da if jacobian else None
        if op=='cos':return np.cos(a),-np.sin(a)[:,None]*da if jacobian else None
        if op=='tanh':
            value=np.tanh(a);return value,(1-value**2)[:,None]*da if jacobian else None
        return a*a,2*a[:,None]*da if jacobian else None
    with np.errstate(over='ignore',invalid='ignore'):
        value,jac=visit()
    if not np.isfinite(value).all() or (jacobian and not np.isfinite(jac).all()):raise FloatingPointError('Non-finite expression')
    return (value,jac) if jacobian else value


def formula(tokens, constants=None):
    """Produce a readable formula, preserving every fitted constant slot."""
    position=0;constant=0
    def visit():
        nonlocal position,constant
        op=tokens[position];position+=1
        if op=='t':return 't'
        if op=='C':
            value=f'c{constant}' if constants is None else f'({constants[constant]:.9g})';constant+=1;return value
        a=visit()
        if ARITY[op]==2:return f'({a} {op} {visit()})'
        return f'({a})^2' if op=='square' else f'{op}({a})'
    validate(tokens);return visit()


def description(tokens):
    """Fixed-width reference accounting, not an entropy-coded bitrate claim.

Four bits per operator/leaf, a 16-bit token-count header, and float64 constants
plus output gain. JSON file size is recorded separately. Time grid is fixed.
"""
    k=tokens.count('C')+1
    return dict(parameters=k,nodes=len(tokens)+2,description_bits=16+4*len(tokens)+64*k)


@dataclass
class Fit:
    tokens:list
    constants:list
    gain:float
    nmse:float
    nfev:int
    converged:bool
    optimality:float
    attempts:int

    def record(self):
        return dict(**asdict(self),**description(self.tokens),formula=f'{self.gain:.9g} * {formula(self.tokens,self.constants)}')

    def waveform(self,t):return self.gain*evaluate(self.tokens,self.constants,t)


def initial_constants(tokens,y,t,rng):
    """FFT frequencies seed coefficient optimization only, never fix topology."""
    window=np.hanning(len(y));spectrum=np.abs(np.fft.rfft(y*window));spectrum[0]=0
    peak=np.argsort(spectrum)[-8:];omega=2*np.pi*peak/(len(y)*(t[1]-t[0]))
    result=[];position=0
    def visit(inside_trig=False):
        nonlocal position
        op=tokens[position];position+=1
        if op=='C':
            result.append(float(rng.choice(omega)) if inside_trig and rng.random()<.6 else float(rng.normal()))
        for _ in range(ARITY[op]):visit(inside_trig or op in ('sin','cos'))
    visit();return np.asarray(result)


def fit_expression(tokens,y,*,seed=0,restarts=3,max_nfev=160,initial=None):
    """Fit constants with analytic derivatives and keep the best restart.

Success flags and first-order optimality describe local optimization only. No
global guarantee is inferred from a least_squares termination flag.
"""
    validate(tokens);y=np.asarray(y,dtype=np.float64);t=np.linspace(-1,1,len(y))
    scale=max(float(np.sqrt(np.mean(y*y))),1e-12);target=y/scale
    k=tokens.count('C');rng=np.random.default_rng(seed);best=None;total=0
    if k==0:
        prediction=evaluate(tokens,[],t)
        amplitude=float(np.dot(prediction,target)/max(np.dot(prediction,prediction),1e-24))
        return Fit(list(tokens),[],scale*amplitude,float(np.mean((amplitude*prediction-target)**2)),0,True,0.,1)
    # Bounds keep trigonometric and polynomial searches numerically tractable;
    # they are declared in the manifest, rather than silently implied.
    for attempt in range(restarts):
        x0=np.asarray(initial,dtype=float) if attempt==0 and initial is not None else initial_constants(tokens,target,t,rng)
        x0=np.clip(x0,-2047,2047)
        try:
            # Output scale is a genuine fitted coefficient too. Jointly fit it
            # rather than fixing RMS normalization and handicapping trees that
            # have no amplitude constant inside their structure.
            shape=evaluate(tokens,x0,t)
            amplitude=np.dot(shape,target)/max(np.dot(shape,shape),1e-24)
            x0=np.r_[x0,np.clip(amplitude,-2047,2047)]
            def residual(c):return c[-1]*evaluate(tokens,c[:-1],t)-target
            def jac(c):
                shape,d=evaluate(tokens,c[:-1],t,True)
                return np.column_stack([c[-1]*d,shape])
            result=least_squares(residual,x0,jac=jac,bounds=(-2048,2048),max_nfev=max_nfev,
                                 ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale='jac')
            total+=result.nfev;nmse=float(np.mean(result.fun**2))
            fit=Fit(list(tokens),result.x[:-1].tolist(),scale*float(result.x[-1]),nmse,total,bool(result.success),float(result.optimality),attempt+1)
            if best is None or nmse<best.nmse:best=fit
            if nmse<1e-12:break
        except (ValueError,FloatingPointError,OverflowError):continue
    if best is None:raise ValueError('All constant-fitting restarts failed')
    best.nfev=total;best.attempts=attempt+1
    return best


def frontier(records):
    """Non-dominated error/complete-description-size fits, not a global optimum."""
    ordered=sorted(records,key=lambda r:(r['description_bits'],r['nmse']));result=[];best=math.inf
    for row in ordered:
        if row['nmse']<best:result.append(row);best=row['nmse']
    return result
