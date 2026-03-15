"""
API Clients

Клиенты для внешних API:
- NASA API
- External services
"""

from .nasa_api_client import NASAAPIClient, get_nasa_client, close_nasa_client

__all__ = [
    'NASAAPIClient',
    'get_nasa_client',
    'close_nasa_client',
]
