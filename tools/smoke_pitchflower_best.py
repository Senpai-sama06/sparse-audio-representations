import os,sys,json
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch,soundfile as sf,pyworld
from scipy.signal import resample_poly
from pesq import pesq
from pystoi import stoi
from vocos import Vocos
from approaches.pitchflower.training.campaign import TrainingModule

torch.set_num_threads(1)
run=Path('runs/pitchflower_hardware'); out=Path('runs/pitchflower_rvq_best_smoke'); out.mkdir(exist_ok=True)
settings=json.loads((run/'settings.json').read_text()); checkpoint=run/'rvq/checkpoints/best_step=91500.ckpt'
payload=torch.load(checkpoint,map_location='cpu',weights_only=False); model=TrainingModule(settings,'rvq'); model.load_state_dict(payload['state_dict'],strict=True); del payload
model.cuda().eval(); vocoder=Vocos.from_pretrained('charactr/vocos-mel-24khz').cuda().eval(); results=[]
with torch.inference_mode():
 for i in range(2):
  x,sr=sf.read(run/'rvq/audio'/f'{i}_original.wav',dtype='float64')
  f0,t=pyworld.harvest(x,24000,frame_period=256/24); f0=pyworld.stonemask(x,f0,t,24000)
  sp=pyworld.cheaptrick(x,f0,t,24000); ap=pyworld.d4c(x,f0,t,24000); voiced=f0>0
  flat=np.where(voiced,f0[voiced].mean() if voiced.any() else 0,0)
  perturbed=pyworld.synthesize(flat,sp,ap,24000,frame_period=256/24)[:len(x)]; perturbed=np.pad(perturbed,(0,max(0,len(x)-len(perturbed))))
  perturbed=perturbed/max(np.max(np.abs(perturbed)),1e-8)
  pitch=np.zeros_like(f0); pitch[voiced]=np.log2(f0[voiced])
  torch.manual_seed(i)
  mel=model.net.reconstruct_mel(torch.tensor(perturbed,dtype=torch.float32,device='cuda')[None],torch.tensor(pitch,dtype=torch.float32,device='cuda')[None],steps=10)
  y=vocoder.decode(mel).flatten().cpu().numpy()
  oracle=vocoder.decode(model.net.mel(torch.tensor(x,dtype=torch.float32,device='cuda')[None]).clamp_min(1e-7).log()).flatten().cpu().numpy()
  rec={'sample':i,'voiced_fraction':float(voiced.mean())}
  for name,z in [('model',y),('vocoder_reference',oracle)]:
   n=min(len(x),len(z)); a,b=resample_poly(x[:n],2,3),resample_poly(z[:n],2,3)
   rec[name+'_stoi']=float(stoi(a,b,16000)); rec[name+'_pesq_wb']=float(pesq(16000,a,b,'wb'))
   rec[name+'_finite']=bool(np.isfinite(z).all()); rec[name+'_peak']=float(np.max(np.abs(z)))
  for name,z in [('original',x),('reconstructed',y),('vocoder_reference',oracle)]: sf.write(out/f'{i}_{name}.wav',z,24000,subtype='FLOAT')
  results.append(rec); print(rec,flush=True)
summary={'checkpoint':str(checkpoint),'scope':'Two previously exported validation crops (1.504 s each). Fresh WORLD analysis on cropped waveforms, flat F0 without shift; not the original cached full-utterance perturbations. Smoke test only.', 'integration_steps':10,'results':results}
(out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)
