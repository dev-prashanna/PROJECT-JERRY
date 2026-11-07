"""Bounded subprocess model requests with enforced deadlines and cancellation."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time
from uuid import uuid4


class ModelClient:
    def __init__(self, name='models'):
        self.name = name
        self.process = None
        self.lock = threading.Lock()
        self.lifecycle = threading.RLock()
        self.selector = None
        self.buffer = bytearray()

    def start(self):
        with self.lifecycle:
            if self.process is not None and self.process.poll() is None:
                return
            root = Path(__file__).resolve().parents[1]
            self.process = subprocess.Popen([sys.executable,'-m','jerry.model_worker'],cwd=root,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,bufsize=0)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.process.stdout,selectors.EVENT_READ)
            self.buffer.clear()

    def call(self, operation, *, timeout=45., cancelled=lambda:False, **payload):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Model worker busy; request rejected')
        try:
            if cancelled():
                raise RuntimeError('Request cancelled')
            self.start()
            identity = uuid4().hex
            request = json.dumps(dict(version=1,id=identity,operation=operation,**payload),allow_nan=False).encode()+b'\n'
            if len(request)>4_000_000:
                raise ValueError('Request exceeds IPC bound')
            # A single bounded request is written while the child is reading, never an unbounded queue.
            self.process.stdin.write(request)
            end = time.monotonic()+timeout
            while time.monotonic()<end:
                if cancelled():
                    raise RuntimeError('Request cancelled')
                if self.process is None or self.process.poll() is not None:
                    raise RuntimeError('Model worker exited')
                if not self.selector.select(.05):
                    continue
                data = os.read(self.process.stdout.fileno(),65536)
                if not data:
                    raise RuntimeError('Model worker closed its response pipe')
                self.buffer.extend(data)
                if len(self.buffer)>8_000_000:
                    raise RuntimeError('Worker response exceeds IPC bound')
                if b'\n' in self.buffer:
                    line,_,remainder = self.buffer.partition(b'\n')
                    self.buffer = bytearray(remainder)
                    response = json.loads(line)
                    if response.get('version')!=1 or response.get('id')!=identity:
                        raise RuntimeError('Worker response association mismatch')
                    if not response.get('ok'):
                        raise RuntimeError(response.get('error','Model failed'))
                    return response['result']
            raise TimeoutError('Model request deadline exceeded')
        except Exception:
            self.close()
            raise
        finally:
            self.lock.release()

    def close(self):
        with self.lifecycle:
            process,self.process = self.process,None
            if process:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=2.)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2.)
                for stream in (process.stdin,process.stdout):
                    stream.close()
            if self.selector:
                self.selector.close()
                self.selector = None
