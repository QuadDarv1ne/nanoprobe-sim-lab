#!/bin/bash
# -*- coding: utf-8 -*-
# Скрипт для генерации SSL сертификатов
# Использование: ./scripts/generate_ssl_certs.sh [domain]

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Параметры
DOMAIN=${1:-"nanoprobe-lab.local"}
EMAIL=${2:-"admin@localhost"}
CERT_DIR="${3:-./deployment/nginx/ssl}"
DAYS=365

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Генерация SSL сертификатов${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "  Domain: ${DOMAIN}"
echo "  Email: ${EMAIL}"
echo "  Срок действия: ${DAYS} дней"
echo "  Директория: ${CERT_DIR}"
echo ""

# Создание директории
mkdir -p "${CERT_DIR}"

# Проверка наличия OpenSSL
if ! command -v openssl &> /dev/null; then
    echo -e "${RED}Ошибка: OpenSSL не найден${NC}"
    echo "Установите OpenSSL:"
    echo "  Ubuntu/Debian: sudo apt-get install openssl"
    echo "  macOS: brew install openssl"
    echo "  Windows: choco install openssl"
    exit 1
fi

# Генерация самоподписанного сертификата (для разработки/тестирования)
echo -e "${YELLOW}Генерация самоподписанного сертификата...${NC}"

openssl req -x509 -nodes -days ${DAYS} -newkey rsa:2048 \
    -keyout "${CERT_DIR}/${DOMAIN}.key" \
    -out "${CERT_DIR}/${DOMAIN}.crt" \
    -subj "/C=RU/ST=Moscow/L=Moscow/O=Nanoprobe Sim Lab/OU=IT/CN=${DOMAIN}/emailAddress=${EMAIL}" \
    -addext "subjectAltName=DNS:${DOMAIN},DNS:localhost,IP:127.0.0.1"

# Установка правильных прав
chmod 600 "${CERT_DIR}/${DOMAIN}.key"
chmod 644 "${CERT_DIR}/${DOMAIN}.crt"

echo ""
echo -e "${GREEN}✓ Сертификаты сгенерированы:${NC}"
echo "  Certificate: ${CERT_DIR}/${DOMAIN}.crt"
echo "  Private Key: ${CERT_DIR}/${DOMAIN}.key"
echo ""

# Проверка сертификата
echo -e "${YELLOW}Информация о сертификате:${NC}"
openssl x509 -in "${CERT_DIR}/${DOMAIN}.crt" -text -noout | head -20

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Готово!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Для production используйте Let's Encrypt:"
echo "  sudo certbot --nginx -d ${DOMAIN}"
echo ""
