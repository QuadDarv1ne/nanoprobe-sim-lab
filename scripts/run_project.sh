#!/bin/bash

# Shell script to run the Nanoprobe Simulation Lab project

# Function to display welcome message
print_welcome() {
    echo "========================================"
    echo "       ЛАБОРАТОРИЯ МОДЕЛИРОВАНИЯ НАНОЗОНДА"
    echo "    Nanoprobe Simulation Lab - Launcher"
    echo "========================================"
    echo
}

# Function to display menu
show_menu() {
    echo
    echo "ДОСТУПНЫЕ ОПЕРАЦИИ:"
    echo "  1. Запустить симулятор СЗМ (C++)"
    echo "  2. Запустить анализатор изображений (Python)"
    echo "  3. Запустить наземную станцию SSTV (Python/C++)"
    echo "  4. Показать информацию о проекте"
    echo "  5. Показать текущую лицензию"
    echo "  0. Выход"
    echo
}

# Function to run SPM simulator
run_spm() {
    echo
    echo "--- ЗАПУСК СИМУЛЯТОРА СЗМ ---"
    echo "Для запуска симулятора СЗМ выполните следующие команды:"
    echo "  cd cpp-spm-hardware-sim"
    echo "  mkdir build && cd build"
    echo "  cmake .. && make"
    echo "  ./spm-simulator"
    echo
    read -p "Нажмите Enter для продолжения..."
}

# Function to run surface analyzer
run_analyzer() {
    echo
    echo "--- ЗАПУСК АНАЛИЗАТОРА ИЗОБРАЖЕНИЙ ---"
    echo "Для запуска анализатора изображений выполните следующие команды:"
    echo "  cd py-surface-image-analyzer"
    echo "  pip install -r requirements.txt"
    echo "  python src/main.py"
    echo
    read -p "Нажмите Enter для продолжения..."
}

# Function to run SSTV ground station
run_sstv() {
    echo
    echo "--- ЗАПУСК НАЗЕМНОЙ СТАНЦИИ SSTV ---"
    echo "Для запуска наземной станции SSTV выполните следующие команды:"
    echo "  cd py-sstv-groundstation"
    echo "  pip install -r requirements.txt"
    echo "  python src/main.py"
    echo
    read -p "Нажмите Enter для продолжения..."
}

# Function to show project info
show_info() {
    echo
    echo "--- ИНФОРМАЦИЯ О ПРОЕКТЕ ---"
    echo "Название: Лаборатория моделирования нанозонда"
    echo "Описание: Комплекс инструментов для моделирования наноразмерных измерительных систем"
    echo "         и космических микроскопических систем."
    echo
    echo "Компоненты проекта:"
    echo "  1. Симулятор СЗМ на C++:"
    echo "     - Моделирует работу сканирующего зондового микроскопа"
    echo "     - Обеспечивает управление зондом и сбор топографических данных"
    echo
    echo "  2. Анализатор изображений на Python:"
    echo "     - Обрабатывает и анализирует изображения поверхности"
    echo "     - Применяет фильтры и визуализирует 3D-модели"
    echo
    echo "  3. Наземная станция SSTV на Python/C++:"
    echo "     - Принимает и декодирует сигналы со спутников"
    echo "     - Работает с реальными космическими данными"
    echo
    echo "Цель проекта: Образовательные и исследовательские цели в нанотехнологиях"
    echo "               и радионауке."
    echo
    read -p "Нажмите Enter для продолжения..."
}

# Function to show license
show_license() {
    echo
    echo "--- ЛИЦЕНЗИЯ НА ИСПОЛЬЗОВАНИЕ ПРОГРАММНОГО ОБЕСПЕЧЕНИЯ ---"
    if [ -f "LICENCE" ]; then
        cat LICENCE
    else
        echo "Файл лицензии не найден."
    fi
    echo
    read -p "Нажмите Enter для продолжения..."
}

# Main program loop
print_welcome

while true; do
    show_menu
    read -p "Выберите действие (0-5): " choice

    case $choice in
        1)
            run_spm
            ;;
        2)
            run_analyzer
            ;;
        3)
            run_sstv
            ;;
        4)
            show_info
            ;;
        5)
            show_license
            ;;
        0)
            echo
            echo "Спасибо за использование Лаборатории моделирования нанозонда!"
            echo "До новых встреч!"
            break
            ;;
        *)
            echo
            echo "Неверный выбор. Пожалуйста, выберите от 0 до 5."
            read -p "Нажмите Enter для продолжения..."
            ;;
    esac
done