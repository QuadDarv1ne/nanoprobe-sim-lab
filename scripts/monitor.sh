#!/bin/bash
# Скрипт мониторинга Nanoprobe Sim Lab
# Использование: ./monitor.sh [--interval SECONDS]

INTERVAL=${1:-5}
LOG_FILE="logs/monitor.log"

# Создание директории логов
mkdir -p logs

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> $LOG_FILE

    case $level in
        "INFO") echo -e "${GREEN}[$timestamp] $message${NC}" ;;
        "WARN") echo -e "${YELLOW}[$timestamp] $message${NC}" ;;
        "ERROR") echo -e "${RED}[$timestamp] $message${NC}" ;;
    esac
}

check_service() {
    local name=$1
    local port=$2

    if nc -z localhost $port 2>/dev/null; then
        log_message "INFO" "✅ $name (порт $port) работает"
        return 0
    else
        log_message "ERROR" "❌ $name (порт $port) НЕ работает"
        return 1
    fi
}

check_disk_space() {
    local threshold=${1:-80}
    local usage=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')

    if [ "$usage" -gt "$threshold" ]; then
        log_message "WARN" "⚠️  Дисковое пространство: ${usage}% (порог ${threshold}%)"
        return 1
    else
        log_message "INFO" "✅ Дисковое пространство: ${usage}%"
        return 0
    fi
}

check_memory() {
    local threshold=${1:-80}
    local usage=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')

    if [ "$usage" -gt "$threshold" ]; then
        log_message "WARN" "⚠️  Использование памяти: ${usage}% (порог ${threshold}%)"
        return 1
    else
        log_message "INFO" "✅ Использование памяти: ${usage}%"
        return 0
    fi
}

check_database() {
    local db_path="data/nanoprobe.db"

    if [ -f "$db_path" ]; then
        local size=$(du -h "$db_path" | cut -f1)
        log_message "INFO" "✅ База данных: $size"
        return 0
    else
        log_message "ERROR" "❌ База данных не найдена"
        return 1
    fi
}

check_logs() {
    local max_size=${1:-104857600}  # 100MB

    for log_file in logs/*.log; do
        if [ -f "$log_file" ]; then
            local size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null)
            if [ "$size" -gt "$max_size" ]; then
                log_message "WARN" "⚠️  Лог $log_file большой: $(du -h $log_file | cut -f1)"
            fi
        fi
    done
}

# Основной цикл мониторинга
log_message "INFO" "🚀 Запуск мониторинга (интервал: ${INTERVAL}s)"

while true; do
    echo ""
    echo "========================================"
    echo "  Мониторинг Nanoprobe Sim Lab"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    # Проверка сервисов
    check_service "FastAPI API" 8000
    check_service "Flask Web" 5000

    # Проверка ресурсов
    check_disk_space 80
    check_memory 80
    check_database
    check_logs

    # Health check API
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null)
    if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
        log_message "INFO" "✅ API Health Check: OK"
    else
        log_message "ERROR" "❌ API Health Check: FAILED"
    fi

    # Пауза
    sleep $INTERVAL
done
