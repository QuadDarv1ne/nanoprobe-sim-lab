"""
Продвинутый SSTV API endpoint с WebSocket для real-time streaming
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from api.error_handlers import ServiceUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter()

# SSTV Receiver
try:
    from api.sstv.rtl_sstv_receiver import get_decoder, get_receiver

    RECEIVER_AVAILABLE = True
except ImportError:
    RECEIVER_AVAILABLE = False


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
        """Добавить WebSocket подключение"""
        await websocket.accept()
        self.websocket_connections.append(websocket)
        logger.info(f"WebSocket подключено. Всего: {len(self.websocket_connections)}")

    async def remove_websocket(self, websocket: WebSocket):
        """Удалить WebSocket подключение"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
        logger.info(f"WebSocket отключено. Всего: {len(self.websocket_connections)}")

    async def broadcast(self, message: dict):
        """Отправить сообщение всем WebSocket клиентам"""
        disconnected = []
        for ws in self.websocket_connections[:]:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        # Удаляем отключенные
        for ws in disconnected:
            await self.remove_websocket(ws)

    def start_recording(
        self, session_id: str, frequency: float, duration: float, gain: float
    ) -> dict:
        """Начать запись SSTV"""
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
        """Завершить запись"""
        if session_id in self.recording_sessions:
            session = self.recording_sessions[session_id]
            session["status"] = "completed"
            session["file_path"] = file_path
            session["completed_at"] = datetime.now(timezone.utc).isoformat()

    def get_stats(self) -> dict:
        """Получить статистику"""
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
    """Получить менеджер сессий"""
    return _session_manager


# ============================================
# API Endpoints
# ============================================


@router.get("/api/v1/sstv/status", summary="Статус SSTV системы")
async def get_sstv_status():
    """Полный статус SSTV системы"""
    receiver = get_receiver() if RECEIVER_AVAILABLE else None
    receiver_info = receiver.get_device_info() if receiver else {"available": False}

    return {
        "receiver": receiver_info,
        "session_manager": get_session_manager().get_stats(),
        "frequencies": {
            "iss_sstv": 145.800,
            "noaa_apt": 137.100,
            "meteor_m2": 137.100,
        },
        "modes": (
            list(
                RECEIVER_AVAILABLE
                and __import__(
                    "api.sstv.rtl_sstv_receiver", fromlist=["SSTV_MODES"]
                ).SSTV_MODES.keys()
            )
            if RECEIVER_AVAILABLE
            else []
        ),
    }


@router.post("/api/v1/sstv/record/start", summary="Начать запись SSTV")
async def start_sstv_recording(
    frequency: float = Query(145.800, description="Частота МГц"),
    duration: float = Query(60.0, description="Длительность секунд"),
    gain: float = Query(49.6, description="Усиление дБ"),
    session_id: Optional[str] = Query(None, description="ID сессии"),
):
    """Начать запись SSTV сигнала"""
    import uuid

    if not RECEIVER_AVAILABLE:
        raise ServiceUnavailableError("SSTV receiver not available")

    sid = session_id or str(uuid.uuid4())[:8]
    receiver = get_receiver()

    if not receiver.initialize():
        raise ServiceUnavailableError("Failed to initialize RTL-SDR")

    # Настраиваем приемник
    receiver.frequency = frequency
    receiver.gain = gain

    session_manager = get_session_manager()
    session_manager.start_recording(sid, frequency, duration, gain)

    # Запускаем запись в фоне с обработкой ошибок
    async def _record_with_error_handling():
        try:
            await _record_sstv_background(sid, receiver, duration)
        except Exception as exc:
            logging.getLogger(__name__).error("Background SSTV recording failed: %s", exc)
            session_manager.finish_recording(sid)

    record_task = asyncio.create_task(_record_with_error_handling())
    session_manager._task = record_task  # сохраняем ссылку

    return {
        "status": "started",
        "session_id": sid,
        "frequency": frequency,
        "duration": duration,
        "message": f"Запись начата на частоте {frequency} МГц",
    }


@router.post("/api/v1/sstv/record/stop", summary="Остановить запись SSTV")
async def stop_sstv_recording(session_id: str = Query(None)):
    """Остановить запись SSTV"""
    if not RECEIVER_AVAILABLE:
        raise ServiceUnavailableError("SSTV receiver not available")

    receiver = get_receiver()
    receiver.is_running = False

    session_manager = get_session_manager()
    if session_id:
        session_manager.finish_recording(session_id)

    return {"status": "stopped", "message": "Запись остановлена"}


@router.get("/api/v1/sstv/recordings", summary="Список записей SSTV")
async def list_sstv_recordings(limit: int = Query(50, ge=1, le=500)):
    """Получить список записей"""
    session_manager = get_session_manager()
    sessions = list(session_manager.recording_sessions.values())[-limit:]

    return {
        "recordings": sessions,
        "total": len(sessions),
        "limit": limit,
    }


@router.get("/api/v1/sstv/spectrum", summary="Спектр сигнала")
async def get_sstv_spectrum(
    frequency: float = Query(145.800, description="Частота МГц"),
    span: float = Query(2.0, description="Диапазон МГц"),
    points: int = Query(1024, description="Количество точек"),
):
    """Получить спектр сигнала"""
    if not RECEIVER_AVAILABLE:
        raise ServiceUnavailableError("SSTV receiver not available")

    receiver = get_receiver()
    # Инициализируем только если устройство ещё не открыто
    if receiver.sdr is None and not receiver.initialize():
        raise ServiceUnavailableError("Failed to initialize RTL-SDR")

    # Настраиваем частоту если нужно
    if abs(receiver.frequency - frequency) > 0.001:
        receiver.set_frequency(frequency)

    freqs, power = receiver.get_spectrum(num_points=points)

    if freqs is None or power is None:
        raise ServiceUnavailableError("Failed to get spectrum")

    return {
        "frequency_mhz": frequency,
        "span_mhz": span,
        "points": len(freqs),
        "frequencies": freqs.tolist(),
        "power_db": power.tolist(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/sstv/signal-strength", summary="Сила сигнала")
async def get_signal_strength():
    """Получить текущую силу сигнала"""
    if not RECEIVER_AVAILABLE:
        raise ServiceUnavailableError("SSTV receiver not available")

    receiver = get_receiver()
    # Инициализируем только если устройство ещё не открыто
    if receiver.sdr is None and not receiver.initialize():
        raise ServiceUnavailableError("Failed to initialize RTL-SDR")

    strength = receiver.get_signal_strength()

    return {
        "strength_percent": strength,
        "frequency_mhz": receiver.frequency,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.websocket("/ws/sstv/stream")
async def sstv_websocket_stream(websocket: WebSocket):
    """WebSocket stream для SSTV данных"""
    session_manager = get_session_manager()
    await session_manager.add_websocket(websocket)

    # Инициализируем receiver один раз при подключении
    receiver = None
    if RECEIVER_AVAILABLE:
        receiver = get_receiver()
        if receiver.sdr is None:
            receiver.initialize()

    try:
        while True:
            # Получаем данные от приемника
            if receiver and receiver.sdr:
                # Отправляем спектр
                freqs, power = receiver.get_spectrum(num_points=512)
                if freqs is not None and power is not None:
                    await websocket.send_json(
                        {
                            "type": "spectrum",
                            "frequencies": freqs.tolist()[-100:],  # Последние 100 точек
                            "power_db": power.tolist()[-100:],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                # Отправляем силу сигнала
                strength = receiver.get_signal_strength()
                await websocket.send_json(
                    {
                        "type": "signal_strength",
                        "strength": strength,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            await asyncio.sleep(1.0)  # Обновление каждую секунду

    except WebSocketDisconnect:
        logger.info("SSTV WebSocket отключен")
    except Exception as e:
        logger.error(f"SSTV WebSocket ошибка: {e}")
    finally:
        await session_manager.remove_websocket(websocket)


# ============================================
# Background Tasks
# ============================================


async def _record_sstv_background(session_id: str, receiver, duration: float):
    """Фоновая задача записи SSTV"""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"data/sstv/recording_{session_id}_{ts}.wav"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Записываем
        audio_data = receiver.record_audio(
            duration=duration, sample_rate=48000, output_file=output_file
        )

        session_manager = get_session_manager()
        session_manager.finish_recording(session_id, output_file)

        # Уведомляем клиентов
        await session_manager.broadcast(
            {
                "type": "recording_completed",
                "session_id": session_id,
                "file_path": output_file,
                "duration": duration,
            }
        )

        # Пытаемся декодировать
        if audio_data is not None:
            decoder = get_decoder()
            image = decoder.decode_audio(audio_data)

            if image is not None:
                image_path = f"data/sstv/decoded_{session_id}.png"
                image.save(image_path)
                session_manager.stats["total_images_decoded"] += 1

                await session_manager.broadcast(
                    {
                        "type": "image_decoded",
                        "session_id": session_id,
                        "image_path": image_path,
                        "image_size": decoder.decode_stats["last_image_size"],
                    }
                )

    except Exception as e:
        logger.error(f"Ошибка записи SSTV: {e}")
        session_manager = get_session_manager()
        session_manager.finish_recording(session_id)

        await session_manager.broadcast(
            {
                "type": "recording_error",
                "session_id": session_id,
                "error": str(e),
            }
        )


# ============================================
# Startup/Shutdown
# ============================================


@router.on_event("startup")
async def startup_sstv():
    """Инициализация при старте"""
    logger.info("SSTV API initialized")


@router.on_event("shutdown")
async def shutdown_sstv():
    """Очистка при завершении"""
    receiver = get_receiver() if RECEIVER_AVAILABLE else None
    if receiver:
        receiver.close()
    logger.info("SSTV API shutdown")
