"""Independent decoding from a shared decoder asset and per-block code vectors."""
import json
from pathlib import Path
import torch
from .learned_models import SharedDecoder
from util.provenance import file_hash


def write_payload(path,asset,codes,means,gains):
    value=dict(version=1,sample_rate=16000,samples=640,decoder_sha256=file_hash(asset),
               codes=codes.tolist(),means=means.reshape(-1).tolist(),gains=gains.reshape(-1).tolist())
    Path(path).write_text(json.dumps(value,separators=(',',':'),allow_nan=False))
    return dict(parameters=sum(len(c) for c in value['codes'])+2*len(value['codes']),payload_bytes=Path(path).stat().st_size)


def decode_payload(asset,path):
    value=json.loads(Path(path).read_text())
    if value['version']!=1 or value['decoder_sha256']!=file_hash(asset):raise ValueError('Decoder identity mismatch')
    shared=torch.load(asset,map_location='cpu',weights_only=True)
    code=torch.tensor(value['codes'],dtype=torch.float32)
    with torch.no_grad():
        if shared['kind']=='pca':
            shape=code@shared['basis'][:,:code.shape[-1]].T
        else:
            decoder=SharedDecoder(**shared['spec']);decoder.load_state_dict(shared['state']);decoder.eval()
            shape=decoder(torch.nn.functional.pad(code,(0,shared['spec']['max_dim']-code.shape[-1])))
        audio=(shape*torch.tensor(value['gains'])[:,None]+torch.tensor(value['means'])[:,None]).sum(0)
    if not torch.isfinite(audio).all() or audio.numel()!=640:raise ValueError('Invalid decoded block')
    return audio.numpy(),16000


if __name__=='__main__':
    import argparse
    import soundfile as sf
    p=argparse.ArgumentParser();p.add_argument('decoder');p.add_argument('payload');p.add_argument('wav');a=p.parse_args()
    audio,rate=decode_payload(a.decoder,a.payload);sf.write(a.wav,audio,rate,subtype='FLOAT')
