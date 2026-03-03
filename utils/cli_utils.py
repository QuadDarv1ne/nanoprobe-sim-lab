# -*- coding: utf-8 -*-
"""
Утилиты для улучшения CLI
Progress bar, цвета, форматирование вывода
"""

import sys
import time
from typing import Optional, Iterable, Any
from datetime import datetime


class Colors:
    """ANSI цвета для терминала."""
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Цвета текста
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Яркие цвета
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    
    # Фоны
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    @classmethod
    def disable(cls):
        """Отключает цвета (для Windows без colorama)."""
        cls.RESET = ''
        cls.BOLD = ''
        cls.RED = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.BLUE = ''
        cls.MAGENTA = ''
        cls.CYAN = ''
        cls.WHITE = ''


class ProgressBar:
    """Progress bar для отображения прогресса операций."""

    def __init__(
        self,
        total: int,
        desc: str = '',
        bar_length: int = 40,
        show_eta: bool = True,
        color: str = Colors.GREEN
    ):
        """
        Инициализирует progress bar.

        Args:
            total: Общее количество итераций
            desc: Описание операции
            bar_length: Длина полосы прогресса
            show_eta: Показывать ли оставшееся время
            color: Цвет progress bar
        """
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.show_eta = show_eta
        self.color = color
        self.current = 0
        self.start_time = None
        self._enabled = sys.stdout.isatty()

    def _format_time(self, seconds: float) -> str:
        """Форматирует время в читаемый формат."""
        if seconds < 60:
            return f"{seconds:.0f}с"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}м{secs}с"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}ч{mins}м"

    def update(self, n: int = 1):
        """Обновляет прогресс на n шагов."""
        if self.start_time is None:
            self.start_time = time.time()
        
        self.current += n
        self._render()

    def _render(self):
        """Рендерит progress bar."""
        if not self._enabled:
            return

        percent = self.current / self.total if self.total > 0 else 0
        filled_length = int(self.bar_length * percent)
        bar = '█' * filled_length + '░' * (self.bar_length - filled_length)

        # Рассчитываем ETA
        eta_str = ''
        if self.show_eta and self.start_time:
            elapsed = time.time() - self.start_time
            if self.current > 0:
                eta = elapsed * (self.total - self.current) / self.current
                eta_str = f" | ETA: {self._format_time(eta)}"

        # Формируем строку
        sys.stdout.write(f'\r{self.desc} {self.color}{bar}{Colors.RESET} '
                        f'{percent*100:5.1f}% ({self.current}/{self.total}){eta_str}')
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def __iter__(self):
        """Итератор для использования в цикле for."""
        self.start_time = time.time()
        for item in range(self.total):
            yield item
            self.update(1)

    def __enter__(self):
        """Контекстный менеджер."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Завершение контекстного менеджера."""
        if self.current < self.total:
            self.current = self.total
            self._render()


def colorize(text: str, color: str) -> str:
    """Окрашивает текст."""
    return f"{color}{text}{Colors.RESET}"


def print_success(message: str):
    """Выводит сообщение об успехе."""
    print(f"{Colors.BRIGHT_GREEN}✓{Colors.RESET} {message}")


def print_error(message: str):
    """Выводит сообщение об ошибке."""
    print(f"{Colors.BRIGHT_RED}✗{Colors.RESET} {message}")


def print_warning(message: str):
    """Выводит предупреждение."""
    print(f"{Colors.BRIGHT_YELLOW}⚠{Colors.RESET} {message}")


def print_info(message: str):
    """Выводит информационное сообщение."""
    print(f"{Colors.BRIGHT_BLUE}ℹ{Colors.RESET} {message}")


def print_step(step_num: int, total: int, message: str):
    """Выводит шаг выполнения."""
    step_str = f"[{step_num}/{total}]"
    print(f"{Colors.CYAN}{step_str}{Colors.RESET} {message}")


def print_header(title: str, width: int = 60):
    """Выводит заголовок."""
    border = '═' * width
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}╔{border}╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET} {Colors.BOLD}{title.center(width)}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╚{border}╝{Colors.RESET}\n")


def print_table(headers: list, rows: list, col_widths: list = None):
    """
    Выводит таблицу.

    Args:
        headers: Заголовки столбцов
        rows: Строки данных
        col_widths: Ширина столбцов (авто если None)
    """
    if col_widths is None:
        col_widths = [max(len(str(row[i])) if i < len(row) else 0 
                         for row in [headers] + rows) + 2 
                     for i in range(len(headers))]

    # Разделитель
    separator = '+' + '+'.join('─' * w for w in col_widths) + '+'
    
    print(f"{Colors.CYAN}{separator}{Colors.RESET}")
    
    # Заголовки
    header_line = '|' + '|'.join(str(h).center(w) for h, w in zip(headers, col_widths)) + '|'
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{header_line}{Colors.RESET}")
    print(f"{Colors.CYAN}{separator}{Colors.RESET}")
    
    # Данные
    for row in rows:
        row_line = '|' + '|'.join(str(row[i]).ljust(w) if i < len(row) else ''.ljust(w) 
                                  for i, w in enumerate(col_widths)) + '|'
        print(row_line)
    
    print(f"{Colors.CYAN}{separator}{Colors.RESET}")


class Spinner:
    """Спиннер для отображения активности."""

    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, message: str = ''):
        """Инициализирует спиннер."""
        self.message = message
        self._running = False
        self._thread = None

    def _animate(self):
        """Анимирует спиннер."""
        import itertools
        for frame in itertools.cycle(self.FRAMES):
            if not self._running:
                break
            sys.stdout.write(f'\r{frame} {self.message}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        sys.stdout.flush()

    def start(self, message: str = None):
        """Запускает спиннер."""
        if message:
            self.message = message
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True):
        """Останавливает спиннер."""
        self._running = False
        if self._thread:
            self._thread.join()
        
        if success:
            print(f"{Colors.BRIGHT_GREEN}✓{Colors.RESET} {self.message}")
        else:
            print(f"{Colors.BRIGHT_RED}✗{Colors.RESET} {self.message}")

    def __enter__(self):
        """Контекстный менеджер."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Завершение контекстного менеджера."""
        self.stop(success=exc_type is None)


# Проверка поддержки цветов в Windows
if sys.platform == 'win32':
    try:
        import colorama
        colorama.init()
    except ImportError:
        # Если colorama нет, отключаем цвета
        Colors.disable()
