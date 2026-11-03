"""Versioned, independently decodable streams with actual packed integer bytes."""
import json
import struct
from pathlib import Path
import numpy as np
from .models import synthesize


def write_payload(path, frames, *, samples, window, hop, bits=32, sample_rate=16000):
    """Store all model values plus timing, model identity and quantizer metadata.

    Integer quantizers use per-mode min/range (two float32 values). Twelve-bit
    integers are bit-packed, not stored in sixteen-bit containers.
    """
    if bits not in (8,12,16,32):
        raise ValueError('Supported precisions: 8, 12, 16, 32')
    records, body, scalars = [], bytearray(), 0
    for start, models in frames:
        group = []
        for model in models:
            p = model.parameters.astype('<f4'); n = len(p)
            if not np.isfinite(p).all():
                raise ValueError('Nonfinite model parameters')
            scalars += n
            if bits == 32 or n == 0:
                raw = p.tobytes()
            else:
                low = np.float32(p.min()); span = np.float32(float(p.max())-float(low))
                integers = np.rint((p.astype(float)-low)/max(float(span),1e-30)*((1<<bits)-1)).clip(0,(1<<bits)-1).astype(np.uint32)
                binary = ((integers[:,None] >> np.arange(bits)) & 1).astype(np.uint8)
                raw = struct.pack('<ff', low, span) + np.packbits(binary.ravel(), bitorder='little').tobytes()
                scalars += 2
            group.append([model.family,n,len(raw)]); body.extend(raw)
        records.append([start,group])
    header = json.dumps(dict(version=1,samples=samples,window=window,hop=hop,
                            sample_rate=sample_rate,bits=bits,frames=records), separators=(',',':')).encode()
    Path(path).write_bytes(b'PAR1'+struct.pack('<I',len(header))+header+body)
    return dict(parameters=scalars, model_parameters=sum(len(m.parameters) for _, ms in frames for m in ms),
                payload_bytes=8+len(header)+len(body), metadata_bytes=8+len(header))


def decode_payload(path):
    """Decode with only the file and the public mathematical model definitions."""
    raw = Path(path).read_bytes()
    if raw[:4] != b'PAR1':
        raise ValueError('Invalid parameter stream')
    length = struct.unpack('<I',raw[4:8])[0]
    header = json.loads(raw[8:8+length]); pos = 8+length
    if header['version'] != 1:
        raise ValueError('Unsupported stream version')
    n, size, bits = header['samples'],header['window'],header['bits']
    audio, weight = np.zeros(n), np.zeros(n)
    weights = np.hanning(size+2)[1:-1]
    for start, records in header['frames']:
        frame = np.zeros(size)
        for family,count,byte_count in records:
            chunk = raw[pos:pos+byte_count]; pos += byte_count
            if len(chunk) != byte_count:
                raise ValueError('Truncated stream')
            if bits == 32 or count == 0:
                p = np.frombuffer(chunk,dtype='<f4').astype(float)
            else:
                low,span = struct.unpack('<ff',chunk[:8])
                binary = np.unpackbits(np.frombuffer(chunk[8:],dtype=np.uint8),bitorder='little')[:count*bits].reshape(count,bits)
                integers = binary @ (2.**np.arange(bits))
                p = low + span*integers/((1<<bits)-1)
            frame += synthesize(family,p,size)
        left,right = max(0,start),min(n,start+size)
        sl = slice(left-start,right-start)
        audio[left:right] += frame[sl]*weights[sl]; weight[left:right] += weights[sl]
    if pos != len(raw) or np.any(weight <= 0) or not np.isfinite(audio).all():
        raise ValueError('Invalid stream coverage, length or decoded values')
    return audio/weight, header['sample_rate']


if __name__ == '__main__':
    import argparse
    import soundfile as sf
    parser = argparse.ArgumentParser(); parser.add_argument('payload'); parser.add_argument('wav')
    args = parser.parse_args(); audio, rate = decode_payload(args.payload)
    sf.write(args.wav,audio,rate,subtype='FLOAT')
