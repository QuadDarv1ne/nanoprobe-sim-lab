@echo off
REM Batch script to run the Nanoprobe Simulation Lab project

echo ========================================
echo        ЛАБОРАТОРИЯ МОДЕЛИРОВАНИЯ НАНОЗОНДА
echo     Nanoprobe Simulation Lab - Launcher
echo ========================================
echo.

:menu
echo.
echo ДОСТУПНЫЕ ОПЕРАЦИИ:
echo   1. Запустить симулятор СЗМ (C++)
echo   2. Запустить анализатор изображений (Python)
echo   3. Запустить наземную станцию SSTV (Python/C++)
echo   4. Показать информацию о проекте
echo   5. Показать текущую лицензию
echo   0. Выход
echo.

set /p choice=Выберите действие (0-5): 

if "%choice%"=="1" goto run_spm
if "%choice%"=="2" goto run_analyzer
if "%choice%"=="3" goto run_sstv
if "%choice%"=="4" goto show_info
if "%choice%"=="5" goto show_license
if "%choice%"=="0" goto exit

echo Неверный выбор. Пожалуйста, выберите от 0 до 5.
pause
goto menu

:run_spm
echo.
echo --- ЗАПУСК СИМУЛЯТОРА СЗМ ---
echo Для запуска симулятора СЗМ выполните следующие команды:
echo   cd cpp-spm-hardware-sim
echo   mkdir build && cd build
echo   cmake .. && make
echo   .\spm-simulator.exe
echo.
pause
goto menu

:run_analyzer
echo.
echo --- ЗАПУСК АНАЛИЗАТОРА ИЗОБРАЖЕНИЙ ---
echo Для запуска анализатора изображений выполните следующие команды:
echo   cd py-surface-image-analyzer
echo   pip install -r requirements.txt
echo   python src/main.py
echo.
pause
goto menu

:run_sstv
echo.
echo --- ЗАПУСК НАЗЕМНОЙ СТАНЦИИ SSTV ---
echo Для запуска наземной станции SSTV выполните следующие команды:
echo   cd py-sstv-groundstation
echo   pip install -r requirements.txt
echo   python src/main.py
echo.
pause
goto menu

:show_info
echo.
echo --- ИНФОРМАЦИЯ О ПРОЕКТЕ ---
echo Название: Лаборатория моделирования нанозонда
echo Описание: Комплекс инструментов для моделирования наноразмерных измерительных систем
echo          и космических микроскопических систем.
echo.
echo Компоненты проекта:
echo   1. Симулятор СЗМ на C++:
echo      - Моделирует работу сканирующего зондового микроскопа
echo      - Обеспечивает управление зондом и сбор топографических данных
echo.
echo   2. Анализатор изображений на Python:
echo      - Обрабатывает и анализирует изображения поверхности
echo      - Применяет фильтры и визуализирует 3D-модели
echo.
echo   3. Наземная станция SSTV на Python/C++:
echo      - Принимает и декодирует сигналы со спутников
echo      - Работает с реальными космическими данными
echo.
echo Цель проекта: Образовательные и исследовательские цели в нанотехнологиях
echo               и радионауке.
echo.
pause
goto menu

:show_license
echo.
echo --- ЛИЦЕНЗИЯ НА ИСПОЛЬЗОВАНИЕ ПРОГРАММНОГО ОБЕСПЕЧЕНИЯ ---
if exist LICENCE (
    type LICENCE
) else (
    echo Файл лицензии не найден.
)
echo.
pause
goto menu

:exit
echo.
echo Спасибо за использование Лаборатории моделирования нанозонда!
echo До новых встреч!
echo.
pause