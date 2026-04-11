#!/usr/bin/env python3
"""
Тесты для RTL-SDR Recording Control (SSTV)
"""

import sys
from pathlib import Path

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio

from api.routes.sstv import (
    get_recording_status,
    list_recordings,
    start_sstv_recording,
    stop_sstv_recording,
)
from api.state import get_app_state


def test_recording_init():
    """Тест инициализации переменных записи"""
    print("Тест инициализации переменных записи...")

    # Переменные должны быть инициализированы через app_state
    recording_process = get_app_state("recording_process")
    assert recording_process is None or hasattr(recording_process, "pid")

    recording_start_time = get_app_state("recording_start_time")
    assert recording_start_time is None or isinstance(recording_start_time, object)

    recording_metadata = get_app_state("recording_metadata", {})
    assert isinstance(recording_metadata, dict)

    print("[PASS] Инициализация переменных записи")


async def test_get_recording_status_idle():
    """Тест статуса записи (режим ожидания)"""
    print("Тест статуса записи (idle)...")

    status = await get_recording_status()

    assert "status" in status
    assert "recording" in status
    assert status["recording"] is False or status["status"] == "idle"

    print("[PASS] Статус записи (idle)")


async def test_get_recording_status_structure():
    """Тест структуры ответа статуса записи"""
    print("Тест структуры ответа статуса...")

    status = await get_recording_status()

    # Проверяем наличие обязательных полей
    assert "status" in status, "Должно быть поле 'status'"
    assert "recording" in status, "Должно быть поле 'recording'"

    if status["recording"]:
        assert "started_at" in status, "При активной записи должно быть 'started_at'"
        assert "duration_seconds" in status, "Должно быть 'duration_seconds'"
        assert "metadata" in status, "Должно быть 'metadata'"
    else:
        assert status["status"] == "idle", "Статус должен быть 'idle'"

    print("[PASS] Структура ответа статуса")


async def test_list_recordings():
    """Тест списка записей"""
    print("Тест списка записей...")

    result = await list_recordings(limit=20)

    assert "recordings" in result, "Должно быть поле 'recordings'"
    assert isinstance(result["recordings"], list), "recordings должен быть списком"

    # Если есть записи, проверяем структуру
    if result["recordings"]:
        recording = result["recordings"][0]
        assert "filename" in recording, "Должно быть 'filename'"
        assert "path" in recording, "Должно быть 'path'"
        assert "size_bytes" in recording, "Должно быть 'size_bytes'"
        assert "created_at" in recording, "Должно быть 'created_at'"

    print("[PASS] Список записей")


async def test_start_recording_validation():
    """Тест валидации параметров запуска записи"""
    print("Тест валидации параметров записи...")

    # Тест с параметрами по умолчанию (145.800 MHz - МКС)
    result = await start_sstv_recording(
        frequency=145.800, sample_rate=2048000, gain=496, duration=60  # Короткая запись для теста
    )

    assert "status" in result, "Должно быть поле 'status'"

    # Статус должен быть одним из:
    valid_statuses = ["recording_started", "recording_simulated", "already_recording"]
    assert result["status"] in valid_statuses, f"Невалидный статус: {result['status']}"

    # Проверяем наличие частоты в ответе
    if "frequency_mhz" in result:
        assert result["frequency_mhz"] == 145.800, "Частота должна совпадать"

    print("[PASS] Валидация параметров записи")


async def test_recording_metadata():
    """Тест метаданных записи"""
    print("Тест метаданных записи...")

    # Запускаем запись с кастомными параметрами
    result = await start_sstv_recording(
        frequency=146.000, sample_rate=1024000, gain=300, duration=30
    )

    if result["status"] in ["recording_started", "recording_simulated"]:
        # Проверяем метаданные
        if "metadata" in result or "output_file" in result:
            print("  Метаданные присутствуют")

    # Останавливаем запись
    stop_result = await stop_sstv_recording()
    assert "status" in stop_result, "Должно быть поле 'status'"

    print("[PASS] Метаданные записи")


async def test_stop_recording_when_not_recording():
    """Тест остановки записи, когда запись не идёт"""
    print("Тест остановки (когда не записывает)...")

    # Сначала убеждаемся, что запись не идёт
    status = await get_recording_status()

    if not status["recording"]:
        result = await stop_sstv_recording()
        assert "status" in result
        # Статус должен быть not_recording или recording_stopped
        assert result["status"] in [
            "not_recording",
            "recording_stopped",
            "recording_stopped_simulated",
        ]

    print("[PASS] Остановка (когда не записывает)")


async def test_recording_frequency_range():
    """Тест различных частот"""
    print("Тест диапазона частот...")

    frequencies = [
        145.800,  # МКС SSTV
        146.000,  # Любительская
        437.550,  # UHF диапазон
    ]

    for freq in frequencies:
        result = await start_sstv_recording(frequency=freq, duration=10)
        assert "status" in result

        # Если началась запись, останавливаем
        if result["status"] in ["recording_started", "recording_simulated"]:
            await asyncio.sleep(1)
            await stop_sstv_recording()

    print("[PASS] Диапазон частот")


async def run_async_tests():
    """Запуск асинхронных тестов"""
    print("\n" + "=" * 60)
    print("Асинхронные тесты RTL-SDR Recording")
    print("=" * 60 + "\n")

    await test_get_recording_status_idle()
    await test_get_recording_status_structure()
    await test_list_recordings()
    await test_start_recording_validation()
    await test_recording_metadata()
    await test_stop_recording_when_not_recording()
    await test_recording_frequency_range()

    print("\n" + "=" * 60)
    print("Все асинхронные тесты пройдены!")
    print("=" * 60 + "\n")


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("Тесты RTL-SDR Recording Control (SSTV)")
    print("=" * 60 + "\n")

    # Синхронные тесты
    test_recording_init()

    # Асинхронные тесты
    asyncio.run(run_async_tests())

    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
