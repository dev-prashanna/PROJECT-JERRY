"""Single-writer durable task/event journal; derived search is rebuildable."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import sqlite3
from uuid import uuid4


class Journal:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA busy_timeout=1000')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS records (
            record_id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
            kind TEXT NOT NULL, task_id TEXT NOT NULL, text TEXT NOT NULL,
            evidence_id TEXT NOT NULL, session_id TEXT NOT NULL,
            epoch_id INTEGER NOT NULL, map_revision INTEGER NOT NULL, created_at TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
            payload TEXT NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '');
          CREATE INDEX IF NOT EXISTS record_task ON records(task_id);
          PRAGMA user_version=1;
        ''')

    def append(self, request_id, kind, text, task_id='', evidence_id='', session_id='', epoch_id=0, map_revision=0):
        if not request_id or len(request_id)>128 or not kind or len(text)>8000:
            raise ValueError('Invalid bounded journal entry')
        old = self.db.execute('SELECT record_id,kind,task_id,text,evidence_id,session_id,epoch_id,map_revision FROM records WHERE request_id=?', (request_id,)).fetchone()
        values = (kind, task_id, text, evidence_id, session_id, epoch_id, map_revision)
        if old:
            if tuple(old[1:]) != values:
                raise ValueError('Idempotency key reused with different content')
            return old[0]
        identity = uuid4().hex
        with self.db:
            self.db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',
                (identity, request_id, *values, datetime.now(timezone.utc).isoformat()))
        return identity

    def admit(self, request_id, payload):
        canonical = json.dumps(payload, sort_keys=True, allow_nan=False)
        old = self.db.execute('SELECT task_id,payload,status FROM tasks WHERE request_id=?', (request_id,)).fetchone()
        if old:
            if old[1] != canonical:
                raise ValueError('Idempotency conflict')
            return old[0], old[2]
        identity = uuid4().hex
        with self.db:
            self.db.execute('INSERT INTO tasks(task_id,request_id,payload,status) VALUES (?,?,?,?)',
                            (identity, request_id, canonical, 'ADMITTED'))
            self.db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',
                (uuid4().hex, 'admit:'+request_id, 'task', identity, canonical, '',
                 payload['session_id'], payload['epoch_id'], payload['map_revision'], datetime.now(timezone.utc).isoformat()))
        return identity, 'NEW'

    def transition(self, task_id, status, reason):
        if status not in ('RUNNING','SUCCEEDED','FAILED','CANCELLED','EXPIRED'):
            raise ValueError('Invalid task transition')
        with self.db:
            row = self.db.execute('SELECT status FROM tasks WHERE task_id=?', (task_id,)).fetchone()
            if not row:
                raise ValueError('Unknown task')
            if row[0] in ('SUCCEEDED','FAILED','CANCELLED','EXPIRED'):
                if row[0] != status:
                    raise ValueError('Terminal state is immutable')
                return
            self.db.execute('UPDATE tasks SET status=?,reason=? WHERE task_id=?', (status, reason[:2000], task_id))

    def query(self, text='', limit=10):
        limit = min(30, max(1, int(limit)))
        columns = 'record_id,kind,task_id,text,evidence_id,session_id,epoch_id,map_revision,created_at'
        query = f'SELECT {columns} FROM records'
        args = []
        if text:
            query += " WHERE text LIKE ? ESCAPE '\\'"
            args.append('%'+text[:500].replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%')
        query += ' ORDER BY rowid DESC LIMIT ?'
        rows = self.db.execute(query, (*args,limit)).fetchall()
        keys = columns.split(',')
        return [dict(zip(keys,row)) for row in rows]

    def interrupt_previous(self):
        with self.db:
            self.db.execute("UPDATE tasks SET status='FAILED',reason='Runtime restarted; prior authority not restored' WHERE status IN ('ADMITTED','RUNNING')")

    def close(self):
        self.db.close()
