"""
Base Widget Class

Базовый класс для всех виджетов dashboard.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from enum import Enum
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class WidgetPriority(Enum):
    """
    Приоритеты виджетов для определения видимости в разных режимах.

    CRITICAL — всегда виден (CPU, память, критичные ошибки)
    HIGH — виден в standard и enhanced режимах
    NORMAL — виден только в enhanced режиме
    LOW — опциональный виджет
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class WidgetMode(Enum):
    """Режимы отображения dashboard"""
    MINIMAL = "minimal"      # Только CRITICAL виджеты
    STANDARD = "standard"    # CRITICAL + HIGH
    ENHANCED = "enhanced"    # Все виджеты


@dataclass
class WidgetData:
    """Данные виджета"""
    title: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


class Widget(ABC):
    """
    Базовый класс виджета.

    Каждый виджет должен реализовать:
    - refresh() — обновление данных
    - render() — отрисовка в консоли

    Attributes:
        name: Уникальное имя виджета
        title: Заголовок для отображения
        priority: Приоритет видимости
        refresh_interval: Интервал автообновления (секунды)
        enabled: Статус включения
        position: Позиция (row, col)
        size: Размер (height, width)
    """

    def __init__(
        self,
        name: str,
        title: str,
        priority: WidgetPriority = WidgetPriority.NORMAL,
        refresh_interval: int = 5,
        enabled: bool = True,
        position: tuple = (0, 0),
        size: tuple = (1, 1),
    ):
        self.name = name
        self.title = title
        self.priority = priority
        self.refresh_interval = refresh_interval
        self.enabled = enabled
        self.position = position  # (row, col)
        self.size = size  # (height, width)

        # State
        self.last_update: Optional[datetime] = None
        self.data: Optional[WidgetData] = None
        self.error: Optional[str] = None

    @abstractmethod
    async def refresh(self) -> WidgetData:
        """
        Обновить данные виджета.

        Returns:
            WidgetData: Обновлённые данные
        """
        pass

    @abstractmethod
    def render(self, width: int = 40) -> str:
        """
        Отрисовать виджет.

        Args:
            width: Ширина виджета в символах

        Returns:
            str: Строковое представление виджета
        """
        pass

    def is_visible(self, mode: WidgetMode) -> bool:
        """
        Проверить видимость виджета в текущем режиме.

        Args:
            mode: Текущий режим dashboard

        Returns:
            bool: Видим ли виджет
        """
        if not self.enabled:
            return False

        if mode == WidgetMode.MINIMAL:
            return self.priority == WidgetPriority.CRITICAL
        elif mode == WidgetMode.STANDARD:
            return self.priority.value <= WidgetPriority.HIGH.value
        else:  # ENHANCED
            return True

    async def _safe_refresh(self) -> WidgetData:
        """Безопасное обновление с обработкой ошибок"""
        try:
            self.data = await self.refresh()
            self.last_update = datetime.now(timezone.utc)
            self.error = None
            return self.data
        except Exception as e:
            self.error = str(e)
            logger.error(f"Widget {self.name} refresh error: {e}")
            return WidgetData(
                title=self.title,
                content=None,
                error=str(e)
            )

    def get_status_icon(self) -> str:
        """Получить иконку статуса"""
        if self.error:
            return "❌"
        elif self.last_update is None:
            return "⏳"
        else:
            return "✅"
