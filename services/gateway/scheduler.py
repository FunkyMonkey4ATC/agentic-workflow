"""Workload scheduler: priorities, bounded queues, coalescing, reserved capacity.

Deliberately separate from capability resolution (gateway.py). This module knows
nothing about models or HTTP; it only decides *when* a unit of work may run.

Priorities: P0 is handled by rules/templates and never reaches here.
P1 = parser traffic (reserved slots), P2 = enrichment (coalescable), P3 = best effort.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

PRIORITIES = ("P1", "P2", "P3")


class QueueFull(Exception):
    pass


class Superseded(Exception):
    """Raised to a waiter whose request was replaced by a newer one with the same coalesce key."""


@dataclass(order=True)
class _Ticket:
    prio: int
    seq: int
    key: Optional[str] = field(compare=False, default=None)
    event: asyncio.Event = field(compare=False, default_factory=asyncio.Event)
    superseded: bool = field(compare=False, default=False)
    enqueued_at: float = field(compare=False, default_factory=time.monotonic)


class CapabilityScheduler:
    def __init__(self, name: str, max_concurrent: int, queue_limits: Dict[str, int],
                 reserved_for_p1: int = 0, coalesce: bool = False, default_timeout_s: float = 30):
        self.name = name
        self.max_concurrent = max_concurrent
        self.queue_limits = {p: int(queue_limits.get(p, 0)) for p in PRIORITIES}
        self.reserved_for_p1 = reserved_for_p1
        self.coalesce = coalesce
        self.default_timeout_s = default_timeout_s
        self._running = 0
        self._seq = 0
        self._waiting: list[_Ticket] = []
        self._by_key: Dict[str, _Ticket] = {}
        self._lock = asyncio.Lock()
        # metrics
        self.stats = {"admitted": 0, "rejected_full": 0, "superseded": 0, "timeouts": 0,
                      "wait_ms_sum": 0.0, "wait_ms_max": 0.0, "completed": 0}

    # -- admission -----------------------------------------------------------
    def _slots_for(self, prio: str) -> int:
        free = self.max_concurrent - self._running
        if prio == "P1":
            return free
        return free - self.reserved_for_p1

    def _queued(self, prio: str) -> int:
        p = PRIORITIES.index(prio)
        return sum(1 for t in self._waiting if t.prio == p and not t.superseded)

    async def _dispatch_locked(self) -> None:
        """Wake the best waiting ticket(s) that fit the free slots."""
        self._waiting.sort()
        for t in list(self._waiting):
            if t.superseded:
                self._waiting.remove(t)
                continue
            if self._slots_for(PRIORITIES[t.prio]) > 0:
                self._waiting.remove(t)
                self._running += 1
                t.event.set()
            else:
                break

    async def run(self, prio: str, fn: Callable[[], Awaitable[Any]],
                  coalesce_key: Optional[str] = None, timeout_s: Optional[float] = None) -> Any:
        if prio not in PRIORITIES:
            prio = "P2"
        timeout_s = timeout_s or self.default_timeout_s
        async with self._lock:
            self._seq += 1
            ticket = _Ticket(prio=PRIORITIES.index(prio), seq=self._seq, key=coalesce_key)
            if self._slots_for(prio) > 0 and not self._waiting:
                self._running += 1
                ticket.event.set()
            else:
                if self.coalesce and coalesce_key and coalesce_key in self._by_key:
                    old = self._by_key[coalesce_key]
                    if old in self._waiting:
                        old.superseded = True
                        old.event.set()
                        self.stats["superseded"] += 1
                if self._queued(prio) >= self.queue_limits[prio]:
                    self.stats["rejected_full"] += 1
                    raise QueueFull(f"{self.name}: {prio} queue full")
                self._waiting.append(ticket)
                if coalesce_key:
                    self._by_key[coalesce_key] = ticket
        try:
            try:
                await asyncio.wait_for(ticket.event.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                async with self._lock:
                    if ticket in self._waiting:
                        self._waiting.remove(ticket)
                    self.stats["timeouts"] += 1
                raise
            if ticket.superseded:
                raise Superseded(f"{self.name}: superseded by newer request for {coalesce_key}")
            wait_ms = (time.monotonic() - ticket.enqueued_at) * 1000
            self.stats["admitted"] += 1
            self.stats["wait_ms_sum"] += wait_ms
            self.stats["wait_ms_max"] = max(self.stats["wait_ms_max"], wait_ms)
            try:
                return await fn()
            finally:
                self.stats["completed"] += 1
                async with self._lock:
                    self._running -= 1
                    await self._dispatch_locked()
        finally:
            if coalesce_key and self._by_key.get(coalesce_key) is ticket:
                self._by_key.pop(coalesce_key, None)

    def snapshot(self) -> Dict[str, Any]:
        admitted = max(1, self.stats["admitted"])
        return {
            "running": self._running,
            "waiting": {p: self._queued(p) for p in PRIORITIES},
            "limits": self.queue_limits,
            "max_concurrent": self.max_concurrent,
            "reserved_for_p1": self.reserved_for_p1,
            "wait_ms_avg": round(self.stats["wait_ms_sum"] / admitted, 1),
            "wait_ms_max": round(self.stats["wait_ms_max"], 1),
            **{k: v for k, v in self.stats.items() if not k.startswith("wait_ms")},
        }
