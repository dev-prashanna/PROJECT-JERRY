"""Atomic, versioned SLAM checkpoints without Python object deserialization."""
import json
import os
from pathlib import Path

import numpy as np


def write_memory(path,state):
    arrays={}
    def encode(value):
        if isinstance(value,np.ndarray):
            key=f'a{len(arrays)}';arrays[key]=value
            return {'array':key}
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {'dict':[[encode(k),encode(v)] for k,v in value.items()]}
        if isinstance(value,tuple):return {'tuple':[encode(v) for v in value]}
        if isinstance(value,list):return [encode(v) for v in value]
        return value
    metadata=json.dumps({'version':1,'state':encode(state)}).encode()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,metadata=np.frombuffer(metadata,dtype=np.uint8),**arrays)
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_memory(path):
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(archive['metadata'].tobytes())
        if metadata['version']!=1:raise ValueError('Unsupported SLAM memory version')
        def decode(value):
            if isinstance(value,list):return [decode(v) for v in value]
            if isinstance(value,dict):
                if 'array' in value:return archive[value['array']].copy()
                if 'dict' in value:return {decode(k):decode(v) for k,v in value['dict']}
                if 'tuple' in value:return tuple(decode(v) for v in value['tuple'])
            return value
        return decode(metadata['state'])
