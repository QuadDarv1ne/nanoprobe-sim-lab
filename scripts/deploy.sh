#!/bin/bash
# Скрипт развёртывания Nanoprobe Sim Lab на production сервере
# Использование: ./deploy.sh <server_ip>

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Конфигурация
SERVER_IP=${1:-""}
USER=${2:-"root"}
PROJECT_DIR="/opt/nanoprobe-sim-lab"
BACKUP_DIR="/opt/backups"

print_header() {
    echo -e "\n${GREEN}======================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}======================================${NC}\n"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Проверка аргументов
if [ -z "$SERVER_IP" ]; then
    print_error "Использование: ./deploy.sh <server_ip> [user]"
    exit 1
fi

print_header "Развёртывание Nanoprobe Sim Lab"
print_info "Сервер: $SERVER_IP"
print_info "Пользователь: $USER"
print_info "Директория: $PROJECT_DIR"

# Шаг 1: Проверка подключения
print_info "Проверка подключения к серверу..."
if ! ssh -o ConnectTimeout=5 $USER@$SERVER_IP "echo 'Connection successful'" > /dev/null 2>&1; then
    print_error "Не удалось подключиться к серверу"
    exit 1
fi
print_success "Подключение успешно"

# Шаг 2: Создание резервной копии
print_info "Создание резервной копии..."
ssh $USER@$SERVER_IP "
    if [ -d '$PROJECT_DIR' ]; then
        mkdir -p $BACKUP_DIR
        TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
        tar -czf $BACKUP_DIR/pre_deploy_\$TIMESTAMP.tar.gz -C \$(dirname $PROJECT_DIR) \$(basename $PROJECT_DIR)
        echo 'Backup created: pre_deploy_\$TIMESTAMP.tar.gz'
    else
        echo 'No existing installation found'
    fi
"
print_success "Резервная копия создана"

# Шаг 3: Обновление системы
print_info "Обновление системы..."
ssh $USER@$SERVER_IP "
    apt update && apt upgrade -y
    apt install -y python3.11 python3.11-venv python3-pip git cmake build-essential nginx
"
print_success "Система обновлена"

# Шаг 4: Настройка пользователя
print_info "Настройка пользователя nanoprobe..."
ssh $USER@$SERVER_IP "
    if ! id -u nanoprobe >/dev/null 2>&1; then
        useradd -r -m -s /bin/bash nanoprobe
        echo 'Пользователь nanoprobe создан'
    else
        echo 'Пользователь nanoprobe уже существует'
    fi
"
print_success "Пользователь настроен"

# Шаг 5: Клонирование репозитория
print_info "Клонирование репозитория..."
ssh $USER@$SERVER_IP "
    if [ -d '$PROJECT_DIR' ]; then
        cd $PROJECT_DIR
        git pull origin main
        echo 'Репозиторий обновлён'
    else
        sudo -u nanoprobe -i git clone <REPOSITORY_URL> $PROJECT_DIR
        echo 'Репозиторий склонирован'
    fi
"
print_success "Репозиторий готов"

# Шаг 6: Установка зависимостей
print_info "Установка зависимостей..."
ssh $USER@$SERVER_IP "
    cd $PROJECT_DIR
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-api.txt
    pip install gunicorn
"
print_success "Зависимости установлены"

# Шаг 7: Настройка окружения
print_info "Настройка окружения..."
ssh $USER@$SERVER_IP "
    cd $PROJECT_DIR
    if [ ! -f .env ]; then
        cp .env.example .env
        # Генерация случайного JWT_SECRET
        JWT_SECRET=\$(openssl rand -hex 32)
        sed -i \"s/JWT_SECRET=.*/JWT_SECRET=\$JWT_SECRET/\" .env
        echo 'Файл .env создан'
    else
        echo 'Файл .env уже существует'
    fi
"
print_success "Окружение настроено"

# Шаг 8: Создание директорий
print_info "Создание директорий..."
ssh $USER@$SERVER_IP "
    cd $PROJECT_DIR
    mkdir -p data logs output reports backups
    chown -R nanoprobe:nanoprobe $PROJECT_DIR
"
print_success "Директории созданы"

# Шаг 9: Установка systemd сервисов
print_info "Установка systemd сервисов..."
scp scripts/nanoprobe-api.service $USER@$SERVER_IP:/etc/systemd/system/
scp scripts/nanoprobe-web.service $USER@$SERVER_IP:/etc/systemd/system/

ssh $USER@$SERVER_IP "
    systemctl daemon-reload
    systemctl enable nanoprobe-api nanoprobe-web
    echo 'Сервисы установлены'
"
print_success "Сервисы установлены"

# Шаг 10: Настройка nginx
print_info "Настройка nginx..."
scp scripts/nginx-nanoprobe.conf $USER@$SERVER_IP:/etc/nginx/sites-available/nanoprobe

ssh $USER@$SERVER_IP "
    ln -sf /etc/nginx/sites-available/nanoprobe /etc/nginx/sites-enabled/nanoprobe
    nginx -t
    systemctl reload nginx
    echo 'Nginx настроен'
"
print_success "Nginx настроен"

# Шаг 11: Запуск сервисов
print_info "Запуск сервисов..."
ssh $USER@$SERVER_IP "
    systemctl start nanoprobe-api
    systemctl start nanoprobe-web
    sleep 5
    systemctl status nanoprobe-api --no-pager
    systemctl status nanoprobe-web --no-pager
"
print_success "Сервисы запущены"

# Шаг 12: Проверка
print_info "Проверка работоспособности..."
HEALTH_CHECK=$(ssh $USER@$SERVER_IP "curl -s http://localhost:8000/health")

if echo "$HEALTH_CHECK" | grep -q "healthy"; then
    print_success "API работает корректно"
else
    print_error "Проблемы с API"
    exit 1
fi

WEB_CHECK=$(ssh $USER@$SERVER_IP "curl -s -o /dev/null -w '%{http_code}' http://localhost:5000")

if [ "$WEB_CHECK" = "200" ] || [ "$WEB_CHECK" = "302" ]; then
    print_success "Веб-интерфейс работает корректно"
else
    print_error "Проблемы с веб-интерфейсом (HTTP $WEB_CHECK)"
fi

# Шаг 13: Настройка firewall
print_info "Настройка firewall..."
ssh $USER@$SERVER_IP "
    if command -v ufw &> /dev/null; then
        ufw allow 22/tcp
        ufw allow 80/tcp
        ufw allow 443/tcp
        ufw --force enable
        echo 'Firewall настроен'
    else
        echo 'UFW не установлен'
    fi
"
print_success "Firewall настроен"

# Шаг 14: SSL сертификат (опционально)
print_info "Настройка SSL (Let's Encrypt)..."
read -p "Настроить SSL сертификат? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    DOMAIN=$(read -p "Введите домен: " DOMAIN)

    ssh $USER@$SERVER_IP "
        apt install -y certbot python3-certbot-nginx
        certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@example.com
        echo 'SSL сертификат установлен'
    "
    print_success "SSL настроен"
fi

# Завершение
print_header "Развёртывание завершено!"

echo "
✅ Nanoprobe Sim Lab успешно развёрнут

📍 Сервер: $SERVER_IP
🌐 API: http://$SERVER_IP:8000
🌐 Веб: http://$SERVER_IP:5000
📖 Документация: http://$SERVER_IP:8000/docs

🔧 Команды управления:
  systemctl status nanoprobe-api
  systemctl status nanoprobe-web
  journalctl -u nanoprobe-api -f
  journalctl -u nanoprobe-web -f

🔒 Не забудьте:
  - Настроить резервное копирование
  - Установить SSL сертификат
  - Настроить мониторинг
"

print_success "Готово!"
