"""
WebSocket Connection Manager - централизованное управление WebSocket подключениями
Thread-safe операции, broadcast, subscribe/unsubscribe, валидация каналов
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)

# Whitelist допустимых каналов
ALLOWED_CHANNELS = {
    "metrics",      # Системные метрики
    "sstv",         # SSTV обновления
    "iss",          # Позиция МКС
    "simulations",  # Симуляции
    "scans",        # Сканы
    "alerts",       # Оповещения
    "realtime",     # Real-time обновления
}

# Максимальный размер сообщения (64 KB)
MAX_MESSAGE_SIZE = 64 * 1024


class ConnectionManager:
    """Централизованный менеджер WebSocket подключений"""

    def __init__(self):
        # Активные подключения: websocket -> set(channels)
        self.active_connections: Dict[WebSocket, Set[str]] = {}
        # Подписчики по каналам: channel -> set(websocket)
        self.channel_subscribers: Dict[str, Set[WebSocket]] = {
            channel: set() for channel in ALLOWED_CHANNELS
        }
        # Блокировка для thread-safe операций
        self._lock = asyncio.Lock()
        # Счётчик подключений для мониторинга
        self.connection_count = 0
        self.max_connections = 100  # Максимальное количество подключений

    async def connect(self, websocket: WebSocket, channels: Optional[Set[str]] = None) -> bool:
        """
        Принимает новое WebSocket подключение.
        
        Args:
            websocket: WebSocket соединение
            channels: Начальные каналы подписки
            
        Returns:
            bool: True если подключение успешно
        """
        async with self._lock:
            # Проверка лимита подключений
            if len(self.active_connections) >= self.max_connections:
                logger.warning(f"Max connections limit reached ({self.max_connections})")
                await websocket.close(code=1013, reason="Too many connections")
                return False

            # Принятие подключения
            await websocket.accept()
            
            # Инициализация подписок
            subscribed_channels = channels or {"realtime", "metrics"}
            self.active_connections[websocket] = subscribed_channels
            
            # Добавление в каналы
            for channel in subscribed_channels:
                if channel in self.channel_subscribers:
                    self.channel_subscribers[channel].add(websocket)
            
            self.connection_count += 1
            logger.info(
                f"WebSocket connected. Total: {len(self.active_connections)}, "
                f"Channels: {subscribed_channels}"
            )
            return True

    async def disconnect(self, websocket: WebSocket):
        """
        Отключает WebSocket клиента и очищает подписки.
        
        Args:
            websocket: WebSocket соединение
        """
        async with self._lock:
            if websocket not in self.active_connections:
                return

            # Удаление из всех каналов
            channels = self.active_connections[websocket]
            for channel in channels:
                if channel in self.channel_subscribers:
                    self.channel_subscribers[channel].discard(websocket)

            # Удаление подключения
            del self.active_connections[websocket]

            # Закрытие соединения
            try:
                await websocket.close()
            except Exception:
                pass

            logger.info(
                f"WebSocket disconnected. Total: {len(self.active_connections)}"
            )

    async def subscribe(self, websocket: WebSocket, channel: str) -> bool:
        """
        Подписывает клиента на канал.
        
        Args:
            websocket: WebSocket соединение
            channel: Название канала
            
        Returns:
            bool: True если подписка успешна
        """
        # Валидация канала
        if channel not in ALLOWED_CHANNELS:
            logger.warning(f"Attempt to subscribe to invalid channel: {channel}")
            return False

        async with self._lock:
            if websocket not in self.active_connections:
                return False

            # Добавление канала
            self.active_connections[websocket].add(channel)
            self.channel_subscribers[channel].add(websocket)
            logger.debug(f"Subscribed to channel: {channel}")
            return True

    async def unsubscribe(self, websocket: WebSocket, channel: str) -> bool:
        """
        Отписывает клиента от канала.
        
        Args:
            websocket: WebSocket соединение
            channel: Название канала
            
        Returns:
            bool: True если отписка успешна
        """
        async with self._lock:
            if websocket not in self.active_connections:
                return False

            # Удаление канала
            self.active_connections[websocket].discard(channel)
            self.channel_subscribers[channel].discard(websocket)
            logger.debug(f"Unsubscribed from channel: {channel}")
            return True

    async def send_to_channel(self, channel: str, data: Any, exclude: Optional[Set[WebSocket]] = None):
        """
        Отправляет данные всем подписчикам канала.
        
        Args:
            channel: Название канала
            data: Данные для отправки
            exclude: Исключить указанные подключения
        """
        if channel not in self.channel_subscribers:
            return

        exclude = exclude or set()
        message = json.dumps(data, default=str) if not isinstance(data, str) else data

        async with self._lock:
            subscribers = self.channel_subscribers[channel].copy()

        disconnected = []
        for websocket in subscribers:
            if websocket in exclude:
                continue

            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error sending to channel {channel}: {e}")
                disconnected.append(websocket)

        # Очистка отключенных клиентов
        for ws in disconnected:
            await self.disconnect(ws)

    async def send_personal(self, websocket: WebSocket, data: Any) -> bool:
        """
        Отправляет данные конкретному клиенту.
        
        Args:
            websocket: WebSocket соединение
            data: Данные для отправки
            
        Returns:
            bool: True если отправка успешна
        """
        try:
            message = json.dumps(data, default=str) if not isinstance(data, str) else data
            await websocket.send_text(message)
            return True
        except WebSocketDisconnect:
            await self.disconnect(websocket)
            return False
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await self.disconnect(websocket)
            return False

    async def broadcast(self, data: Any, exclude: Optional[Set[WebSocket]] = None):
        """
        Отправляет данные ВСЕМ подключённым клиентам.
        
        Args:
            data: Данные для отправки
            exclude: Исключить указанные подключения
        """
        exclude = exclude or set()
        message = json.dumps(data, default=str) if not isinstance(data, str) else data

        async with self._lock:
            connections = list(self.active_connections.keys())

        disconnected = []
        for websocket in connections:
            if websocket in exclude:
                continue

            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(websocket)

        # Очистка отключенных клиентов
        for ws in disconnected:
            await self.disconnect(ws)

    def validate_message(self, data: str) -> dict:
        """
        Валидирует входящее WebSocket сообщение.
        
        Args:
            data: JSON строка сообщения
            
        Returns:
            dict: Распарсенное сообщение
            
        Raises:
            ValueError: При невалидном сообщении
        """
        # Проверка размера
        if len(data) > MAX_MESSAGE_SIZE:
            raise ValueError(f"Message too large: {len(data)} bytes (max: {MAX_MESSAGE_SIZE})")

        # Парсинг JSON
        try:
            message = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # Проверка типа
        if not isinstance(message, dict):
            raise ValueError("Message must be a JSON object")

        if "type" not in message:
            raise ValueError("Message must have 'type' field")

        # Валидация канала
        if message.get("type") in ("subscribe", "unsubscribe"):
            channel = message.get("channel")
            if not channel:
                raise ValueError("subscribe/unsubscribe requires 'channel' field")
            if channel not in ALLOWED_CHANNELS:
                raise ValueError(
                    f"Invalid channel: {channel}. Allowed: {ALLOWED_CHANNELS}"
                )

        return message

    def get_stats(self) -> dict:
        """
        Получает статистику подключений.
        
        Returns:
            dict: Статистика
        """
        return {
            "total_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            "channels": {
                channel: len(subscribers)
                for channel, subscribers in self.channel_subscribers.items()
            },
            "connection_count": self.connection_count,
        }


# Singleton экземпляр
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Получает singleton экземпляр ConnectionManager"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


def reset_connection_manager():
    """Сбрасывает ConnectionManager (для тестов)"""
    global _connection_manager
    _connection_manager = None
