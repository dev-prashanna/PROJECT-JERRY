"""ROS single-writer memory service; no model or control dependencies."""
from pathlib import Path
import os
from rclpy.node import Node
from jerry_interfaces.srv import Journal as JournalService
from jerry_interfaces.msg import MemoryRecord
from .journal import Journal


class Memory(Node):
    def __init__(self):
        super().__init__('jerry_memory')
        path = Path(os.environ.get('JERRY_MEMORY_DB','simulation/artifacts/jerry/memory.sqlite3'))
        path.parent.mkdir(parents=True,exist_ok=True)
        self.store = Journal(path)
        self.store.interrupt_previous()
        self.create_service(JournalService,'/jerry/memory/journal',self.handle_request)

    def handle_request(self,req,res):
        try:
            if req.operation == 'QUERY':
                res.records = [MemoryRecord(**r) for r in self.store.query(req.text,req.limit)]
            elif req.operation == 'ADMIT':
                # The record text is a canonical, bounded task document produced by the executive.
                import json
                payload = json.loads(req.text)
                if not isinstance(payload,dict) or len(req.text)>8000 or payload.get('session_id')!=req.session_id:
                    raise ValueError('Invalid task journal document')
                res.record_id,res.reason = self.store.admit(req.request_id,payload)
            elif req.operation == 'TRANSITION':
                self.store.transition(req.task_id,req.kind,req.text)
            elif req.operation == 'APPEND':
                res.record_id = self.store.append(req.request_id,req.kind,req.text,req.task_id,req.evidence_id,
                    req.session_id,req.epoch_id,req.map_revision)
            else:
                raise ValueError('Unknown memory operation')
            res.accepted = True
        except Exception as exc:
            res.reason = type(exc).__name__+': '+str(exc)[:500]
        return res

    def close(self):
        self.store.close()
