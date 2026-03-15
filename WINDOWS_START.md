# Руководство по запуску для Windows

## Используемая версия Python

Проект работает **только** с Python 3.13:
```
C:\Users\maksi\AppData\Local\Programs\Python\Python313\python.exe
```

## Быстрый запуск

### Вариант 1: Через batch-файл (рекомендуется)
```cmd
start.bat
```

С режимом:
```cmd
start.bat flask      # Flask Dashboard
start.bat nextjs     # Next.js Dashboard  
start.bat api        # Backend API only
```

### Вариант 2: Прямой запуск
```cmd
C:\Users\maksi\AppData\Local\Programs\Python\Python313\python.exe start.py
```

## Проверка зависимостей

```cmd
C:\Users\maksi\AppData\Local\Programs\Python\Python313\python.exe -m pip list
```

## Установка зависимостей

Если чего-то не хватает:
```cmd
C:\Users\maksi\AppData\Local\Programs\Python\Python313\python.exe -m pip install -r requirements.txt
```

## Порты сервисов

| Сервис | Порт | URL |
|--------|------|-----|
| Backend (FastAPI) | 8000 | http://localhost:8000/docs |
| Flask Dashboard | 5000 | http://localhost:5000 |
| Next.js Dashboard | 3000 | http://localhost:3000 |

## Остановка сервисов

Нажмите `Ctrl+C` в терминале или закройте окно.
