"""Background memory job scheduler."""

from __future__ import annotations

import threading
import time

import pytest

from src.memory.exceptions import MemoryOperationCancelled
from src.memory.resource_scheduler import MemoryJobPriority, MemoryJobScheduler


def test_jobs_run_and_order_by_priority():
    order: list[str] = []
    gate = threading.Event()
    lock = threading.Lock()

    def handler(job):
        # First job holds the worker until released so later jobs queue
        # behind it; then priority ordering kicks in.
        if job.operation == "first":
            gate.wait(timeout=2)
        with lock:
            order.append(job.operation)
        return job.operation

    sched = MemoryJobScheduler(handler, max_pending=10)
    sched.start()
    try:
        first = sched.submit("first", priority=MemoryJobPriority.MAINTENANCE)
        time.sleep(0.05)
        low = sched.submit("low", priority=MemoryJobPriority.MAINTENANCE)
        high = sched.submit("high", priority=MemoryJobPriority.INTERACTIVE)
        normal = sched.submit("normal", priority=MemoryJobPriority.BACKGROUND)
        gate.set()
        for j in (first, low, high, normal):
            sched.wait(j, timeout=2)
        assert order[0] == "first"
        assert order[1] == "high"
        assert order[2] == "normal"
        assert order[-1] == "low"
    finally:
        sched.stop(timeout_s=2)


def test_cancelled_job_not_executed():
    executed: list[str] = []
    gate = threading.Event()
    lock = threading.Lock()

    def handler(job):
        gate.wait(timeout=2)
        with lock:
            executed.append(job.operation)
        return job.operation

    sched = MemoryJobScheduler(handler, max_pending=10)
    sched.start()
    try:
        blocker = sched.submit("blocker", priority=MemoryJobPriority.INTERACTIVE)
        time.sleep(0.05)
        victim = sched.submit("victim", priority=MemoryJobPriority.MAINTENANCE)
        sched.cancel(victim)
        gate.set()
        sched.wait(blocker, timeout=2)
        with pytest.raises(MemoryOperationCancelled):
            sched.wait(victim, timeout=2)
        time.sleep(0.1)
        assert "victim" not in executed
    finally:
        sched.stop(timeout_s=2)


def test_wait_returns_result():
    sched = MemoryJobScheduler(lambda job: 42, max_pending=4)
    sched.start()
    try:
        job = sched.submit("j")
        assert sched.wait(job, timeout=2) == 42
    finally:
        sched.stop(timeout_s=2)


def test_handler_error_propagates():
    def boom(job):
        raise RuntimeError("kaput")

    sched = MemoryJobScheduler(boom, max_pending=4)
    sched.start()
    try:
        job = sched.submit("j")
        with pytest.raises(RuntimeError, match="kaput"):
            sched.wait(job, timeout=2)
        assert sched.failed == 1
    finally:
        sched.stop(timeout_s=2)


def test_queue_full_rejected():
    from src.memory.exceptions import MemoryQueueFull

    gate = threading.Event()
    sched = MemoryJobScheduler(lambda job: gate.wait(timeout=2), max_pending=1)
    sched.start()
    try:
        sched.submit("blocker", priority=MemoryJobPriority.INTERACTIVE)
        time.sleep(0.05)
        with pytest.raises(MemoryQueueFull):
            for _ in range(5):
                sched.submit("extra", priority=MemoryJobPriority.INTERACTIVE)
    finally:
        gate.set()
        sched.stop(timeout_s=2)


def test_status_shape():
    sched = MemoryJobScheduler(lambda job: None, max_pending=4)
    sched.start()
    try:
        status = sched.status()
        assert "running" in status and "pending" in status
    finally:
        sched.stop(timeout_s=2)


def test_stop_without_start_is_safe():
    sched = MemoryJobScheduler(lambda job: None)
    sched.stop(timeout_s=0.1)
