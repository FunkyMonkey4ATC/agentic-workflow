"""Tiny async client for the hub: publish events over HTTP, subscribe over WebSocket."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator, Dict, Iterable, Optional

import httpx
import websockets

HUB_URL = os.environ.get("HUB_URL", "http://127.0.0.1:7000")


class Bus:
    def __init__(self, hub_url: str = HUB_URL):
        self.hub_url = hub_url.rstrip("/")
        self.ws_url = self.hub_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._client = httpx.AsyncClient(timeout=5.0)

    async def publish(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            r = await self._client.post(self.hub_url + "/events", json=event)
            return r.json() if r.status_code < 300 else None
        except Exception:  # hub down: never block the producer
            return None

    async def subscribe(self, types: Optional[Iterable[str]] = None, since: Optional[int] = None) -> AsyncIterator[Dict[str, Any]]:
        """Yield events forever; reconnects with backoff. `types` filters server-side."""
        q = ""
        params = []
        if types:
            params.append("types=" + ",".join(types))
        if since is not None:
            params.append(f"since={since}")
        if params:
            q = "?" + "&".join(params)
        delay = 0.5
        while True:
            try:
                async with websockets.connect(self.ws_url + q, max_size=8 * 1024 * 1024) as ws:
                    delay = 0.5
                    async for msg in ws:
                        yield json.loads(msg)
            except (OSError, websockets.exceptions.WebSocketException):
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)

    async def aclose(self) -> None:
        await self._client.aclose()
