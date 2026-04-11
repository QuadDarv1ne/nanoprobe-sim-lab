"""
SSTV Recording endpoints

Управление записью SSTV через RTL-SDR,
список записей, скачивание, удаление.
"""

import asyncio
import json
import logging
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from subprocess import DEVNULL

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import FileResponse

from api.error_handlers import NotFoundError, ServiceUnavailableError, ValidationError
from api.routes.sstv.helpers import SSTV_AVAILABLE, get_app_state, get_sstv_decoder, set_app_state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/record/start")
async def start_sstv_recording(
    frequency: float = 145.800,
    sample_rate: int = 2048000,
    gain: int = 30,
    duration: int = 600,
    ppm: int = 0,
):
    """
    Запуск записи с RTL-SDR для приёма SSTV.

    Args:
        frequency: Частота в MHz (по умолчанию 145.800 для МКС)
        sample_rate: Частота дискретизации
        gain: Усиление RTL-SDR в dB (0-50)
        duration: Длительность записи в секундах
        ppm: Коррекция частоты в ppm
    """
    recording_process = get_app_state("recording_process")
    recording_start_time = get_app_state("recording_start_time")

    if recording_process is not None:
        return {
            "status": "already_recording",
            "started_at": (recording_start_time.isoformat() if recording_start_time else None),
            "message": "Запись уже идёт",
        }

    output_dir = Path("output/sstv/recordings")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sstv_{frequency}MHz_{timestamp}.wav"

    gain_tenths = gain * 10

    cmd = [
        "rtl_fm",
        "-f",
        str(int(frequency * 1e6)),
        "-s",
        str(sample_rate),
        "-g",
        str(gain_tenths),
        "-F",
        "9",
        "-o",
        "4",
        "-p",
        str(ppm),
        "-M",
        "fm",
        str(output_file),
    ]

    try:
        subprocess.run(["rtl_fm", "-h"], capture_output=True, timeout=5)
    except FileNotFoundError:
        logger.warning("rtl_fm not found - simulation mode")
        recording_start_time = datetime.now(timezone.utc)
        recording_metadata = {
            "frequency": frequency,
            "sample_rate": sample_rate,
            "gain": gain,
            "duration": duration,
            "output_file": str(output_file),
            "simulated": True,
        }
        set_app_state("recording_start_time", recording_start_time)
        set_app_state("recording_metadata", recording_metadata)

        return {
            "status": "recording_simulated",
            "frequency_mhz": frequency,
            "sample_rate": sample_rate,
            "output_file": str(output_file),
            "started_at": recording_start_time.isoformat(),
            "message": ("RTL-SDR не найден. Запись симулируется для тестирования."),
        }

    try:
        proc = subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL)
        recording_start_time = datetime.now(timezone.utc)
        recording_metadata = {
            "frequency": frequency,
            "sample_rate": sample_rate,
            "gain": gain,
            "duration": duration,
            "output_file": str(output_file),
            "pid": proc.pid,
        }
        set_app_state("recording_process", proc)
        set_app_state("recording_start_time", recording_start_time)
        set_app_state("recording_metadata", recording_metadata)

        async def _stop_after(delay: int):
            try:
                await asyncio.sleep(delay)
                await stop_sstv_recording()
            except Exception as exc:
                logger.error("Scheduled recording stop error: %s", exc)

        stop_task = asyncio.create_task(_stop_after(duration))
        set_app_state("recording_stop_task", stop_task)

        return {
            "status": "recording_started",
            "frequency_mhz": frequency,
            "sample_rate": sample_rate,
            "gain": gain,
            "output_file": str(output_file),
            "started_at": recording_start_time.isoformat(),
            "pid": proc.pid,
            "message": f"Запись началась. Остановка через {duration} секунд.",
        }

    except Exception as e:
        logger.error(f"Recording start error: {e}")
        if recording_process:
            try:
                recording_process.kill()
                recording_process.wait(timeout=2)
            except Exception:
                pass
        raise ServiceUnavailableError(f"Не удалось начать запись: {str(e)}")


@router.post("/record/stop")
async def stop_sstv_recording():
    """Остановка записи SSTV."""
    recording_process = get_app_state("recording_process")
    recording_metadata = get_app_state("recording_metadata", {})

    if recording_process is None and not recording_metadata.get("simulated"):
        return {
            "status": "not_recording",
            "message": "Запись не идёт",
        }

    if recording_metadata.get("simulated"):
        recording_start_time = get_app_state("recording_start_time")
        duration = (
            (datetime.now(timezone.utc) - recording_start_time).total_seconds()
            if recording_start_time
            else 0
        )
        set_app_state("recording_process", None)
        metadata = recording_metadata.copy()
        set_app_state("recording_metadata", {})

        return {
            "status": "recording_stopped_simulated",
            "duration_seconds": round(duration, 2),
            "output_file": metadata.get("output_file"),
            "message": "Симуляция записи остановлена",
        }

    try:
        if recording_process is None:
            raise ServiceUnavailableError("Recording process не найден в state")

        recording_process.send_signal(signal.SIGINT)
        try:
            recording_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully, killing")
            recording_process.kill()
            recording_process.wait(timeout=2)

        recording_start_time = get_app_state("recording_start_time")
        duration = (
            (datetime.now(timezone.utc) - recording_start_time).total_seconds()
            if recording_start_time
            else 0
        )
        output_file = recording_metadata.get("output_file")

        set_app_state("recording_process", None)
        set_app_state("recording_metadata", {})

        return {
            "status": "recording_stopped",
            "duration_seconds": round(duration, 2),
            "output_file": output_file,
            "message": "Запись остановлена",
        }

    except Exception as e:
        logger.error(f"Recording stop error: {e}")
        if recording_process:
            try:
                recording_process.kill()
                recording_process.wait(timeout=2)
            except Exception:
                pass
        set_app_state("recording_process", None)
        set_app_state("recording_metadata", {})

        raise ServiceUnavailableError(f"Не удалось остановить запись: {str(e)}")


@router.get("/record/status")
async def get_recording_status():
    """Получить статуса записи."""
    recording_process = get_app_state("recording_process")
    recording_metadata = get_app_state("recording_metadata", {})
    recording_start_time = get_app_state("recording_start_time")

    if recording_process is not None or recording_metadata.get("simulated"):
        duration = (
            (datetime.now(timezone.utc) - recording_start_time).total_seconds()
            if recording_start_time
            else 0
        )

        return {
            "status": "recording",
            "recording": True,
            "started_at": (recording_start_time.isoformat() if recording_start_time else None),
            "duration_seconds": round(duration, 2),
            "metadata": recording_metadata,
        }
    else:
        return {
            "status": "idle",
            "recording": False,
            "message": "Запись не идёт",
        }


@router.get("/recordings")
async def list_recordings(limit: int = 20):
    """Получить список записей SSTV."""
    output_dir = Path("output/sstv/recordings")

    if not output_dir.exists():
        return {"recordings": []}

    recordings = []
    for file in sorted(output_dir.glob("*.wav"), reverse=True)[:limit]:
        stat = file.stat()
        png = file.with_suffix(".png")
        meta_json = file.with_suffix(".json")

        metadata = {}
        if meta_json.exists():
            try:
                metadata = json.loads(meta_json.read_text())
            except Exception:
                pass

        entry = {
            "filename": file.name,
            "path": str(file),
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            "has_image": png.exists(),
            "image_filename": png.name if png.exists() else None,
            "frequency": metadata.get("frequency", "145.800"),
            "metadata": metadata,
        }
        recordings.append(entry)

    return {"count": len(recordings), "recordings": recordings}


@router.get("/recordings/{filename}")
async def download_recording(filename: str):
    """Скачать запись SSTV (WAV файл)."""
    file_path = Path("output/sstv/recordings") / filename

    if not file_path.exists():
        raise NotFoundError(f"Запись не найдена: {filename}")

    return FileResponse(str(file_path), media_type="audio/wav", filename=filename)


@router.delete("/recordings/{filename}")
async def delete_recording(filename: str):
    """Удалить запись SSTV."""
    wav_path = Path("output/sstv/recordings") / filename

    if not wav_path.exists():
        raise NotFoundError(f"Запись не найдена: {filename}")

    try:
        wav_path.unlink()

        png_path = wav_path.with_suffix(".png")
        if png_path.exists():
            png_path.unlink()

        json_path = wav_path.with_suffix(".json")
        if json_path.exists():
            json_path.unlink()

        return {
            "status": "success",
            "message": f"Запись {filename} удалена",
            "deleted_files": [
                str(wav_path),
                str(png_path) if png_path.exists() else None,
                str(json_path) if json_path.exists() else None,
            ],
        }
    except Exception as e:
        logger.error(f"Delete recording error: {e}")
        raise ServiceUnavailableError(f"Ошибка удаления записи: {str(e)}")


@router.post("/sstv/decode-recording/{filename}")
async def decode_existing_recording(
    filename: str,
    mode: str = "auto",
    background_tasks: BackgroundTasks = None,
):
    """
    Декодирует уже записанный WAV файл.
    Сохраняет результат рядом с WAV как PNG.
    """
    if not SSTV_AVAILABLE:
        raise ServiceUnavailableError("SSTV декодер недоступен")

    wav_path = Path("output/sstv/recordings") / filename
    if not wav_path.exists():
        raise NotFoundError(f"Файл не найден: {filename}")

    decoder = get_sstv_decoder()
    if not decoder:
        raise ServiceUnavailableError("SSTV декодер не инициализирован")

    try:
        decoder.mode = mode
        image = decoder.decode_from_audio(str(wav_path))

        if not image:
            raise ValidationError("Не удалось декодировать SSTV из файла")

        output_path = wav_path.with_suffix(".png")
        image.save(str(output_path), "PNG")

        metadata = decoder.get_metadata()

        try:
            from api.state import get_db_manager

            db = get_db_manager()
            db.add_scan_result(
                scan_type="sstv",
                file_path=str(output_path),
                metadata={
                    "source_wav": filename,
                    "sstv_mode": metadata.get("mode", "unknown"),
                    "image_size": list(image.size),
                },
            )
        except Exception as e:
            logger.debug(f"Could not save SSTV scan to DB: {e}")

        return {
            "status": "success",
            "image_path": str(output_path),
            "image_size": image.size,
            "mode": metadata.get("mode", "unknown"),
            "download_url": f"/api/v1/sstv/download/{output_path.name}",
        }

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Decode recording error: {e}")
        raise ServiceUnavailableError(f"Ошибка декодирования: {str(e)}")
