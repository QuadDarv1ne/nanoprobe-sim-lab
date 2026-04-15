"""
SSTV Session Manager — менеджер сессий записи и WebSocket подключений
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class SSTVSessionManager:
    """Менеджер SSTV сессий"""

    def __init__(self):
        self.active_sessions: Dict[str, dict] = {}
        self.recording_sessions: Dict[str, dict] = {}
        self.websocket_connections: List[WebSocket] = []
        self.stats = {
            "total_sessions": 0,
            "total_recordings": 0,
            "total_images_decoded": 0,
            "total_data_received_mb": 0,
        }

    async def add_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.websocket_connections.append(websocket)
        logger.info(f"WebSocket подключено. Всего: {len(self.websocket_connections)}")

    async def remove_websocket(self, websocket: WebSocket):
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
        logger.info(f"WebSocket отключено. Всего: {len(self.websocket_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for ws in self.websocket_connections[:]:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.debug(f"WebSocket send failed: {e}")
                disconnected.append(ws)
        for ws in disconnected:
            await self.remove_websocket(ws)

    def start_recording(
        self, session_id: str, frequency: float, duration: float, gain: float
    ) -> dict:
        session = {
            "id": session_id,
            "frequency": frequency,
            "duration": duration,
            "gain": gain,
            "status": "recording",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "samples_received": 0,
            "file_path": None,
        }
        self.recording_sessions[session_id] = session
        self.stats["total_recordings"] += 1
        return session

    def finish_recording(self, session_id: str, file_path: Optional[str] = None):
        if session_id in self.recording_sessions:
            session = self.recording_sessions[session_id]
            session["status"] = "completed"
            session["file_path"] = file_path
            session["completed_at"] = datetime.now(timezone.utc).isoformat()

    def get_stats(self) -> dict:
        return {
            "active_recordings": len(
                [s for s in self.recording_sessions.values() if s["status"] == "recording"]
            ),
            "total_sessions": self.stats["total_sessions"],
            "total_recordings": self.stats["total_recordings"],
            "total_images_decoded": self.stats["total_images_decoded"],
            "total_data_received_mb": self.stats["total_data_received_mb"],
            "websocket_clients": len(self.websocket_connections),
            "recent_sessions": list(self.recording_sessions.values())[-10:],
        }


# Singleton
_session_manager = SSTVSessionManager()


def get_session_manager() -> SSTVSessionManager:
    return _session_manager
