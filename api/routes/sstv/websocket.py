"""
SSTV WebSocket endpoints

Real-time обновления позиции МКС через WebSocket.
"""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.routes.sstv.helpers import get_satellite_tracker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/iss")
async def iss_websocket(websocket: WebSocket):
    """
    WebSocket для real-time обновлений позиции МКС.
    Отправляет позицию каждые 5 секунд.
    """
    await websocket.accept()

    tracker = get_satellite_tracker()
    if not tracker:
        await websocket.send_json({"error": "Satellite tracker not available"})
        await websocket.close()
        return

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)

                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "stop":
                    break

            except asyncio.TimeoutError:
                try:
                    position = tracker.get_current_position("iss")
                    if position:
                        await websocket.send_json(
                            {
                                "type": "position",
                                "data": position,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                except Exception as exc:
                    logger.debug("WebSocket send position error: %s", exc)
                    pass

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error("WebSocket ISS tracking error: %s", e)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception as e:
            logger.debug("WebSocket error response failed: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logger.debug("WebSocket close failed: %s", e)
