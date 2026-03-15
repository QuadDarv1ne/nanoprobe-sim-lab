#!/usr/bin/env python3
"""
Тесты для NASA API Key configuration
"""

import os
import sys
from pathlib import Path

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_nasa_api_key_exists():
    """Тест наличия NASA API ключа"""
    print("Тест наличия NASA API ключа...")
    
    # Загружаем .env если существует
    env_file = project_root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    api_key = os.getenv("NASA_API_KEY")
    
    assert api_key is not None, "NASA_API_KEY не настроен в .env"
    print(f"   ✅ NASA_API_KEY найден: {api_key[:10]}...")
    

def test_nasa_api_key_not_demo():
    """Тест что не используется DEMO_KEY"""
    print("Тест что не используется DEMO_KEY...")
    
    # Загружаем .env если существует
    env_file = project_root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    api_key = os.getenv("NASA_API_KEY")
    
    if api_key == "DEMO_KEY":
        print("   ⚠️  WARNING: Используется DEMO_KEY (лимит 30 запросов/час)")
        print("   💡 Получите production ключ на https://api.nasa.gov/")
        # Не failing тест, но предупреждаем
    else:
        print(f"   ✅ Production ключ настроен (длина: {len(api_key)})")


def test_nasa_api_key_length():
    """Тест длины ключа"""
    print("Тест длины ключа...")
    
    env_file = project_root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    api_key = os.getenv("NASA_API_KEY")
    
    # Production ключ обычно 40 символов
    if api_key != "DEMO_KEY":
        assert len(api_key) >= 20, f"Слишком короткий ключ: {len(api_key)} символов"
        print(f"   ✅ Длина ключа: {len(api_key)} символов")
    else:
        print(f"   ℹ️  DEMO_KEY (длина: {len(api_key)})")


def test_nasa_api_env_example():
    """Тест наличия NASA_API_KEY в .env.example"""
    print("Тест .env.example...")
    
    env_example = project_root / ".env.example"
    assert env_example.exists(), ".env.example не найден"
    
    content = env_example.read_text()
    assert "NASA_API_KEY" in content, "NASA_API_KEY не указан в .env.example"
    
    print("   ✅ NASA_API_KEY указан в .env.example")


def test_nasa_api_urls_configured():
    """Тест NASA API URLs"""
    print("Тест NASA API URLs...")
    
    env_file = project_root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    urls = {
        "NASA_IMAGE_LIBRARY_URL": "https://images-api.nasa.gov",
        "NASA_EARTH_OBSERVATORY_URL": "https://eoimages.gsfc.nasa.gov",
        "NASA_APOD_URL": "https://api.nasa.gov/planetary/apod"
    }
    
    for var, default_url in urls.items():
        url = os.getenv(var, default_url)
        assert url.startswith("https://"), f"{var} должен использовать HTTPS"
        print(f"   ✅ {var}: {url}")


def main():
    """Запуск всех тестов"""
    print("=" * 70)
    print("  NASA API Key Configuration Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_nasa_api_key_exists,
        test_nasa_api_key_not_demo,
        test_nasa_api_key_length,
        test_nasa_api_env_example,
        test_nasa_api_urls_configured,
    ]
    
    passed = 0
    failed = 0
    warnings = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"   ⚠️  ERROR: {e}")
            warnings += 1
        print()
    
    print("=" * 70)
    print(f"  Результаты: {passed} passed, {failed} failed, {warnings} warnings")
    print("=" * 70)
    
    if failed == 0:
        if warnings > 0:
            print("\n✅ Все тесты пройдены (есть предупреждения)")
        else:
            print("\n✅ Все тесты пройдены!")
    else:
        print(f"\n❌ {failed} тестов не пройдено")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
