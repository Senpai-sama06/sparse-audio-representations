"""Speaker-disjoint fixed 40 ms examples from existing local LibriSpeech files."""
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from util.data.manifests import librispeech_split_root
from util.provenance import file_hash


def select_blocks(corpus_root,seed=42,train_speakers=64,eval_speakers=8,train_utterances=8,eval_utterances=4,blocks_per_utterance=8):
    """Freeze source paths/starts before model training, without quality selection."""
    rows=[];sets={}
    for role,split,count,utterances in [('train','train-clean-360',train_speakers,train_utterances),
                                        ('validation','dev-clean',eval_speakers,eval_utterances),
                                        ('test','test-clean',eval_speakers,eval_utterances)]:
        folder=librispeech_split_root(corpus_root,split)
        speakers=sorted([p for p in folder.iterdir() if p.is_dir()],key=lambda p:hashlib.sha256(f'{seed}:{role}:{p.name}'.encode()).hexdigest())[:count]
        if len(speakers)!=count:raise ValueError('Insufficient local speakers')
        sets[role]={p.name for p in speakers}
        for speaker in speakers:
            paths=sorted(speaker.rglob('*.flac'),key=lambda p:hashlib.sha256(f'{seed}:{p.stem}'.encode()).hexdigest())
            selected=0
            for path in paths:
                info=sf.info(path)
                if info.samplerate!=16000 or info.channels!=1:raise ValueError('Expected mono 16 kHz LibriSpeech')
                slots=info.frames//640
                if slots<blocks_per_utterance:continue
                digest=file_hash(path)
                starts=np.linspace(0,slots-1,blocks_per_utterance).astype(int)*640
                for start in starts:
                    rows.append(dict(role=role,split=split,speaker=speaker.name,path=str(path.resolve()),
                                     utterance=path.stem,start=int(start),samples=640,source_sha256=digest))
                selected+=1
                if selected==utterances:break
            if selected<utterances:raise ValueError(f'Insufficient utterances for speaker {speaker.name}')
    if any(sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Speaker overlap across partitions')
    return rows
