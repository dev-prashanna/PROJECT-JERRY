"""Isolated AURA model interpreter. Bounded JSON-lines IPC; no ROS or actuator access."""
from __future__ import annotations
import base64
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'AURA'))
os.environ.setdefault('STT_DEVICE','cpu')
os.environ.setdefault('STT_COMPUTE_TYPE','int8')
os.environ.setdefault('STT_ALLOW_DOWNLOAD','0')
os.environ.setdefault('TTS_DEVICE','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')


def main():
    # Capture stdout before third-party model code can print progress into the protocol.
    wire = sys.stdout
    sys.stdout = sys.stderr
    from config.settings import get_settings
    settings = get_settings()
    vlm = stt = tts = None
    for line in sys.stdin:
        try:
            if len(line)>4_000_000:
                raise ValueError('IPC request too large')
            request = json.loads(line)
            if request.get('version') != 1:
                raise ValueError('Unsupported worker protocol')
            op = request['operation']
            if op == 'observe':
                from PIL import Image
                if vlm is None:
                    from src.perception.moondream_engine import MoondreamEngine
                    # Only the local, explicitly configured engine receives camera images.
                    vlm = MoondreamEngine(settings.vlm)
                image = Image.open(io.BytesIO(base64.b64decode(request['image'],validate=True))).convert('RGB')
                evidence = vlm.ask(image,request['question'][:500],frame_id=request['frame_id'],max_new_tokens=160)
                output = evidence.model_dump(mode='json')
            elif op == 'transcribe':
                import numpy as np
                if stt is None:
                    from src.speech.whisper_engine import WhisperEngine
                    stt = WhisperEngine(settings.stt)
                raw = base64.b64decode(request['audio'],validate=True)
                if len(raw)>16000*4*20:
                    raise ValueError('Utterance exceeds 20 seconds')
                result = stt.transcribe(np.frombuffer(raw,dtype='<f4'),16000)
                if result.error:
                    raise RuntimeError(result.error)
                output = dict(text=result.text)
            elif op == 'synthesize':
                if tts is None:
                    from src.speech.kokoro_engine import KokoroEngine
                    tts = KokoroEngine(settings.tts)
                audio = tts.synthesize(request['text'][:200])
                output = dict(audio=base64.b64encode(audio.astype('<f4').tobytes()).decode(),sample_rate=tts.sample_rate)
            elif op == 'capture':
                from src.speech.audio_capture import AudioCapture
                with AudioCapture(settings.stt) as capture:
                    recording = capture.record_utterance(max_seconds=15.)
                output = dict(audio=base64.b64encode(recording.samples.astype('<f4').tobytes()).decode())
            elif op == 'status':
                output = dict(planner='external_codex_mcp',vlm_path_exists=Path(settings.vlm.model_path).exists(),tts_path_exists=Path(settings.tts.model_path).exists(),
                    stt_model=settings.stt.model)
            else:
                raise ValueError('Unknown worker operation')
            reply = dict(version=1,id=request['id'],ok=True,result=output)
        except Exception as exc:
            reply = dict(version=1,id=locals().get('request',{}).get('id',''),ok=False,error=type(exc).__name__+': '+str(exc)[:500])
        wire.write(json.dumps(reply,allow_nan=False)+'\n')
        wire.flush()


if __name__ == '__main__':
    main()
