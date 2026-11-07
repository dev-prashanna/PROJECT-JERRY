"""Interruptible speech output; inference and device playback are separate processes."""
import base64
import subprocess
import sys
import threading
import time
from .model_client import ModelClient


class Speech:
    def __init__(self):
        self.worker = ModelClient('speech')
        self.process = None
        self.generation = 0
        self.thread = None
        self.lock = threading.RLock()
        self.last_error = ''

    def stop(self):
        with self.lock:
            self.generation += 1
            if self.process and self.process.poll() is None:
                self.process.terminate()
            self.process = None

    def say(self,text):
        self.stop()
        self.last_error = ''
        generation = self.generation
        if self.thread and self.thread.is_alive():
            # The old inference sees its cancelled generation and terminates its worker.
            return False
        def run():
            try:
                # Clause-sized output has bounded synthesis and playback duration.
                chunks = [text[i:i+200] for i in range(0,min(len(text),1000),200)]
                for chunk in chunks:
                    result = self.worker.call('synthesize',text=chunk,timeout=60.,cancelled=lambda:self.generation!=generation)
                    with self.lock:
                        if self.generation!=generation:return
                        process = subprocess.Popen([sys.executable,'-m','jerry.speech','--play',str(result['sample_rate'])],stdin=subprocess.PIPE)
                        self.process = process
                    raw = base64.b64decode(result['audio'],validate=True)
                    process.communicate(raw,timeout=20.)
                    if process.returncode:
                        raise RuntimeError('Audio playback device failed; text response remains available')
            except Exception as exc:
                self.last_error = str(exc)
                print('Speech unavailable: '+self.last_error,file=sys.stderr,flush=True)
            finally:
                with self.lock:
                    if self.process and self.process.poll() is None:
                        self.process.kill()
                        self.process.wait(timeout=2.)
                    self.process = None
        self.thread = threading.Thread(target=run,daemon=True,name='jerry-speech')
        self.thread.start()
        return True

    def close(self):
        self.stop()
        if self.thread:self.thread.join(timeout=3.)
        self.worker.close()


if __name__ == '__main__':
    import numpy as np
    import sounddevice as sd
    audio = np.frombuffer(sys.stdin.buffer.read(16000*4*40),dtype='<f4')
    with sd.OutputStream(samplerate=int(sys.argv[2]),channels=1,dtype='float32',blocksize=480) as stream:
        for start in range(0,len(audio),480):
            stream.write(audio[start:start+480])
