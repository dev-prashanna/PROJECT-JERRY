"""Pure, receiver-clock motion authorization. No ROS or model dependencies."""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
import math
import secrets
import time


@dataclass(frozen=True)
class Binding:
    session_id: str
    epoch_id: int
    owner_id: str
    task_id: str
    generation: int
    map_revision: int


class AuthorityGate:
    def __init__(self, clock=time.monotonic, token_ttl=.15, lease_ttl=.30):
        if not 0 < token_ttl <= lease_ttl <= 1:
            raise ValueError('invalid authority durations')
        self.clock, self.token_ttl, self.lease_ttl = clock, token_ttl, lease_ttl
        self.binding = None
        self.expiry = 0.
        self.tokens = OrderedDict()
        self.sequence = -1
        self.generation = 0
        self.reason = 'Disarmed'

    def acquire(self, session, epoch, owner, task, revision=0):
        if self.live:
            raise ValueError('Motion already owned')
        if not all(isinstance(s, str) and 0 < len(s) <= 128 for s in (session, owner, task)):
            raise ValueError('Invalid authority identity')
        self.generation += 1
        self.binding = Binding(session, epoch, owner, task, self.generation, revision)
        self.expiry = self.clock() + self.lease_ttl
        self.tokens.clear()
        self.sequence = -1
        self.reason = 'Authorized'
        return self.binding

    @property
    def live(self):
        return self.binding is not None and self.clock() < self.expiry

    def matches(self, obj):
        return self.live and all(getattr(obj, name, None) == getattr(self.binding, name)
                                 for name in Binding.__dataclass_fields__)

    def renew(self, obj):
        if not self.matches(obj):
            raise ValueError('Expired or mismatched lease')
        self.expiry = self.clock() + self.lease_ttl

    def issue(self, obj, maximum_validity_s=None):
        if not self.matches(obj):
            raise ValueError('No live matching authority')
        validity = self.token_ttl if maximum_validity_s is None else maximum_validity_s
        if not math.isfinite(validity) or validity <= 0:
            raise ValueError('Invalid token validity budget')
        now = self.clock()
        self.tokens = OrderedDict((t, end) for t, end in self.tokens.items() if end > now)
        token = secrets.token_hex(16)
        end = min(self.expiry, now + self.token_ttl, now + validity)
        self.tokens[token] = end
        while len(self.tokens) > 8:
            self.tokens.popitem(last=False)
        return token, end - now

    def accept(self, command, maximum_deadline=None):
        if not self.matches(command):
            raise ValueError('Wrong session, epoch, owner, task, generation or geometry')
        end = min(self.expiry, self.tokens.get(command.token, 0.))
        if maximum_deadline is not None:
            if not math.isfinite(maximum_deadline):raise ValueError('Invalid shared-clock deadline')
            end=min(end,maximum_deadline)
        if self.clock() >= end:
            raise ValueError('Expired or unknown receiver token')
        if command.sequence <= self.sequence:
            raise ValueError('Duplicate or reordered command')
        v, w = command.linear_m_s, command.angular_rad_s
        if not all(math.isfinite(x) for x in (v, w)):
            raise ValueError('Non-finite command')
        if command.mode not in ('TRACK', 'PIVOT', 'REVERSE', 'HOLD'):
            raise ValueError('Invalid motion mode')
        if command.mode in ('PIVOT', 'HOLD') and v != 0.:
            raise ValueError('Stationary mode attempted translation')
        if command.mode == 'HOLD' and w != 0.:
            raise ValueError('Hold attempted rotation')
        if v < 0 and command.mode != 'REVERSE':
            raise ValueError('Reverse requires explicit mode')
        if not -.035 <= v <= .08 or abs(w) > .4:
            raise ValueError('Command exceeds profile limits')
        self.sequence = command.sequence
        return v, w, end

    def revoke(self, reason='Stopped'):
        self.generation += 1
        self.binding = None
        self.expiry = 0.
        self.tokens.clear()
        self.reason = reason


def collision_gate(v, w, ranges, now, sim_time, *, max_age=.8):
    """Conservative flat-lab bounds; ranges are (value, receipt, stamp, max_range)."""
    if not all(math.isfinite(x) for x in (v, w, now, sim_time)):
        return 0., 0., 'Invalid control/clock value'
    clearance = {}
    for side in ('front', 'left', 'rear', 'right'):
        if side not in ranges:
            return 0., 0., f'{side} range missing'
        value, receipt, stamp, maximum = ranges[side]
        if (not all(math.isfinite(x) for x in (receipt, stamp, maximum)) or maximum <= 0
                or not 0 <= now-receipt <= max_age or not -.1 <= sim_time-stamp <= .65
                or math.isnan(value) or value == -math.inf or (math.isfinite(value) and not .02 <= value <= maximum)):
            return 0., 0., f'{side} range invalid/stale'
        clearance[side] = maximum if value == math.inf else value
    # Includes the baseline observed delays and a provisional simulator braking margin.
    front = .18 + max(v, 0.) * .8 + max(v, 0.)**2 / .4
    rear = .12 + max(-v, 0.) * .8 + max(-v, 0.)**2 / .4
    if min(clearance['left'], clearance['right']) < .10:
        return 0., 0., 'Side clearance'
    if v > 0 and clearance['front'] < front:
        return 0., 0., 'Forward stopping clearance'
    if v < 0 and clearance['rear'] < rear:
        return 0., 0., 'Reverse stopping clearance'
    if w and min(clearance.values()) < .18:
        return 0., 0., 'Swept pivot clearance'
    return v, w, 'Clear'


def host_clock_id():
    """CLOCK_MONOTONIC has one origin across processes on this Linux boot only."""
    from pathlib import Path
    value=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if len(value)!=36:raise RuntimeError('Validated local-host clock identity unavailable')
    return value
