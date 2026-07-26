from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app import main as app_module
from backend.app.schemas import WsMessage


def test_websocket_message_schema():
    message = WsMessage(type="progress", data={"progress": 0.5})
    assert message.model_dump() == {"type": "progress", "data": {"progress": 0.5}}
    risk = WsMessage(
        type="risk",
        data={
            "available": True,
            "track_id": 3,
            "level": "HIGH",
            "probabilities": {"LOW": 0.01, "MEDIUM": 0.09, "HIGH": 0.9},
        },
    )
    assert risk.type == "risk"


def test_websocket_connect_disconnect_lifecycle():
    job_id = uuid4()
    with TestClient(app_module.app) as client:
        with client.websocket_connect(f"/ws/jobs/{job_id}") as websocket:
            websocket.send_text("ping")
