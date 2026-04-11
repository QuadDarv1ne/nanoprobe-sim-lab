"""
Dashboard Layouts

Раскладки для отображения виджетов.
"""

from .enhanced import EnhancedLayout
from .minimal import MinimalLayout
from .standard import StandardLayout

__all__ = [
    "StandardLayout",
    "EnhancedLayout",
    "MinimalLayout",
]
