from __future__ import annotations

import asyncio
from collections import defaultdict
from uuid import UUID

from fastapi import WebSocket


class FrameConnectionManager:
    def __init__(self) -> None:
        self._queues: dict[UUID, set[asyncio.Queue[bytes]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, job_id: UUID) -> asyncio.Queue[bytes]:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        async with self._lock:
            self._queues[job_id].add(queue)
        return queue

    async def disconnect(self, job_id: UUID, queue: asyncio.Queue[bytes]) -> None:
        async with self._lock:
            queues = self._queues.get(job_id)
            if not queues:
                return
            queues.discard(queue)
            if not queues:
                self._queues.pop(job_id, None)

    async def publish(self, job_id: UUID, frame: bytes) -> None:
        async with self._lock:
            queues = list(self._queues.get(job_id, set()))
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    async def frame_sender(self, job_id: UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        queue = await self.connect(job_id)
        try:
            while True:
                frame = await queue.get()
                await websocket.send_bytes(frame)
        finally:
            await self.disconnect(job_id, queue)
