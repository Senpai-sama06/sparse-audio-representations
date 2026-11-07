import os,sys,json,csv
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch,soundfile as sf
from scipy.signal import resample_poly
from pesq import pesq
from pystoi import stoi
from vocos import Vocos
from approaches.pitchflower.data import PitchDataset
from approaches.pitchflower.training.campaign import TrainingModule

def main():
 torch.set_num_threads(1)
 run=Path('runs/pitchflower_hardware'); out=Path('runs/pitchflower_rvq_best_evaluation'); out.mkdir(exist_ok=True); (out/'audio').mkdir(exist_ok=True)
 settings=json.loads((run/'settings.json').read_text()); audit=json.loads((run/'data_audit.json').read_text())
 checkpoint=run/'rvq/checkpoints/best_step=91500.ckpt'
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model=TrainingModule(settings,'rvq'); model.load_state_dict(payload['state_dict'],strict=True); del payload
 model.cuda().eval(); vocoder=Vocos.from_pretrained('charactr/vocos-mel-24khz').cuda().eval()
 rows=[json.loads(s) for s in Path('pitchflower_gs_standalone/data/manifests/validation.jsonl').read_text().splitlines() if s.strip()]
 excluded=set(audit['validation_ids']+audit['calibration_ids']); fresh={}
 for r in sorted(rows,key=lambda r:r['id']):
  if r['id'] not in excluded: fresh.setdefault(str(r['speaker']),r)
 fresh=list(fresh.values()); dataset=PitchDataset(fresh,settings['cache_dir'],settings['samples'],settings['seed'],False)
 results=[]
 with torch.inference_mode():
  for i,row in enumerate(fresh):
   original,perturbed,pitch=dataset[i]; x=original.numpy(); torch.manual_seed(settings['seed']+i)
   mel=model.net.reconstruct_mel(perturbed[None].cuda(),pitch[None].cuda(),steps=10)
   y=vocoder.decode(mel).flatten().cpu().numpy()
   oracle=vocoder.decode(model.net.mel(original[None].cuda()).clamp_min(1e-7).log()).flatten().cpu().numpy()
   record={'id':row['id'],'speaker':str(row['speaker'])}
   for name,z in [('model',y),('vocoder_reference',oracle)]:
    n=min(len(x),len(z)); ref,est=resample_poly(x[:n],2,3),resample_poly(z[:n],2,3)
    record[name+'_stoi']=float(stoi(ref,est,16000,extended=False))
    try: record[name+'_pesq_wb']=float(pesq(16000,ref,est,'wb'))
    except Exception as e: record[name+'_pesq_wb']=None; record[name+'_error']=str(e)
   if i<6:
    for name,z in [('original',x),('reconstructed',y),('vocoder_reference',oracle)]: sf.write(out/'audio'/f'{i:02d}_{name}.wav',z,24000,subtype='FLOAT')
   results.append(record); (out/'progress.json').write_text(json.dumps({'completed':len(results),'total':len(fresh),'results':results},indent=2)); print(record,flush=True)
 keys=['model_stoi','model_pesq_wb','vocoder_reference_stoi','vocoder_reference_pesq_wb']
 summary={'checkpoint':str(checkpoint),'examples':len(fresh),'speakers':len(fresh),'samples':settings['samples'],'sample_rate':24000,'integration_steps':10,'scope':'Unused dev-clean utterances, one per speaker, first 1.504-second crop; not full utterances or test-clean. No alignment or gain fitting.', 'metrics':{k:{'mean':float(np.mean([r[k] for r in results if r[k] is not None])),'median':float(np.median([r[k] for r in results if r[k] is not None])),'valid':sum(r[k] is not None for r in results)} for k in keys}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)); (out/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
