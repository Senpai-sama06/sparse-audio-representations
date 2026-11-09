"""Signal-conditioned generation of complete variable-length expression trees."""
import numpy as np
import torch
from torch import nn
from .expressions import TOKENS,ARITY,validate,evaluate


PRIOR=np.array([.15,.22,.12,.06,.22,.10,.06,.03,.04])


def allowed(slots,used,budget):
    """A token is legal iff enough room remains to complete the entire tree."""
    return np.array([used+1+slots-1+ARITY[token]<=budget for token in TOKENS])


def random_expression(rng,budget):
    tokens=[];slots=1
    while slots:
        probability=PRIOR*allowed(slots,len(tokens),budget);probability/=probability.sum()
        token=TOKENS[int(rng.choice(len(TOKENS),p=probability))]
        tokens.append(token);slots+=ARITY[token]-1
    return tokens


def synthetic_example(rng,max_nodes=31,samples=640):
    """Random compositions span the grammar; no closed family labels are used."""
    t=np.linspace(-1,1,samples)
    for _ in range(200):
        tokens=random_expression(rng,int(rng.choice([7,15,max_nodes])))
        if len(tokens)<3 or 't' not in tokens or 'C' not in tokens:continue
        constants=rng.normal(size=tokens.count('C'));position=0;index=0
        def initialize(trig=False):
            nonlocal position,index
            token=tokens[position];position+=1
            if token=='C':
                if trig and rng.random()<.6:constants[index]=rng.uniform(2,650)
                index+=1
            for _ in range(ARITY[token]):initialize(trig or token in ('sin','cos'))
        initialize()
        try:y=evaluate(tokens,constants,t)
        except FloatingPointError:continue
        scale=np.sqrt(np.mean(y*y))
        if not np.isfinite(scale) or scale<1e-5 or np.std(y)<1e-5 or np.max(np.abs(y))>1e8:continue
        return (y/scale).astype(np.float32),tokens
    raise RuntimeError('Could not generate a finite nonconstant synthetic function')


class ExpressionGenerator(nn.Module):
    """A neural proposer of formulas, not a neural waveform reconstructor.

An MLP reads all 640 IMF samples. A GRU emits prefix operation/constant tokens.
The signal and a requested node budget initialize its state. Fitted numerical
constants are optimized separately; this pilot does not predict constants.
"""
    def __init__(self,hidden=128):
        super().__init__();self.hidden=hidden
        self.encoder=nn.Sequential(nn.Linear(640,256),nn.GELU(),nn.Linear(256,hidden),nn.Tanh())
        self.budget=nn.Linear(1,hidden);self.embedding=nn.Embedding(len(TOKENS)+1,32)
        self.gru=nn.GRU(32,hidden,batch_first=True);self.head=nn.Linear(hidden,len(TOKENS))
        with torch.no_grad():self.head.bias.copy_(torch.tensor(np.log(PRIOR),dtype=torch.float32))

    def initial(self,y,budgets):
        return (self.encoder(y)+self.budget(torch.log2(budgets[:,None])/6)).unsqueeze(0)

    def log_prob(self,y,programs,budgets):
        """Grammar-normalized log likelihood, shared by pretraining and RL."""
        length=max(map(len,programs));batch=len(programs);device=y.device
        inputs=torch.full((batch,length),len(TOKENS),dtype=torch.long,device=device)
        targets=torch.zeros((batch,length),dtype=torch.long,device=device)
        masks=torch.zeros((batch,length,len(TOKENS)),dtype=torch.bool,device=device)
        active=torch.zeros((batch,length),dtype=torch.bool,device=device)
        for i,tokens in enumerate(programs):
            validate(tokens);slots=1
            for j,token in enumerate(tokens):
                targets[i,j]=TOKENS.index(token);active[i,j]=True
                masks[i,j]=torch.tensor(allowed(slots,j,int(budgets[i])),device=device)
                if j+1<length:inputs[i,j+1]=TOKENS.index(token)
                slots+=ARITY[token]-1
            masks[i,len(tokens):,0]=True
        h,_=self.gru(self.embedding(inputs),self.initial(y,budgets))
        logits=self.head(h).masked_fill(~masks,-1e9)
        lp=torch.log_softmax(logits,-1).gather(-1,targets[:,:,None]).squeeze(-1)
        return (lp*active).sum(1)

    @torch.no_grad()
    def propose(self,y,budget,count,rng):
        """Sample grammar-valid expressions; NumPy RNG is checkpointable."""
        self.eval();device=next(self.parameters()).device
        signal=torch.tensor(y,dtype=torch.float32,device=device).repeat(count,1)
        hidden=self.initial(signal,torch.full((count,),float(budget),device=device))
        previous=torch.full((count,1),len(TOKENS),dtype=torch.long,device=device)
        programs=[[] for _ in range(count)];slots=[1]*count
        for step in range(budget):
            out,hidden=self.gru(self.embedding(previous),hidden);logits=self.head(out[:,0]).cpu().numpy()
            choices=[]
            for i in range(count):
                if slots[i]==0:choices.append(0);continue
                mask=allowed(slots[i],step,budget);p=logits[i].astype(np.float64);p-=p[mask].max()
                p=np.where(mask,np.exp(np.clip(p,-700,0)),0);p/=p.sum()
                choice=int(rng.choice(len(TOKENS),p=p));choices.append(choice)
                token=TOKENS[choice];programs[i].append(token);slots[i]+=ARITY[token]-1
            previous=torch.tensor(choices,device=device)[:,None]
            if not any(slots):break
        return programs
