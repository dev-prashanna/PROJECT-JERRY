"""Local-model check using synthesized speech; no microphone or speaker is opened."""
import base64
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from jerry.model_client import ModelClient


def main():
    worker=ModelClient('speech-roundtrip')
    try:
        result=worker.call('synthesize',text='Stop the robot.',timeout=60.)
        audio=np.frombuffer(base64.b64decode(result['audio']),dtype='<f4')
        pcm=resample_poly(audio,16000,result['sample_rate']).astype('<f4')
        transcript=worker.call('transcribe',audio=base64.b64encode(pcm.tobytes()).decode(),timeout=60.)['text']
        assert 'stop' in transcript.lower() and 'robot' in transcript.lower(),transcript
        report=dict(passed=True,synthesis='local Kokoro CPU',transcription='local Whisper CPU int8',
                    sample_rate=result['sample_rate'],samples=len(audio),transcript=transcript,
                    microphone_opened=False,speaker_opened=False)
        output=Path('simulation/artifacts/jerry/speech-acceptance.json')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:worker.close()


if __name__=='__main__':main()
