#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для генерации SSL сертификатов (Python версия)
Использование: python scripts/generate_ssl_certs.py [domain]
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.GREEN}{'='*50}{Colors.END}")
    print(f"{Colors.GREEN}  {text}{Colors.END}")
    print(f"{Colors.GREEN}{'='*50}{Colors.END}\n")


def check_openssl():
    """Проверка наличия OpenSSL"""
    try:
        subprocess.run(['openssl', 'version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def generate_self_signed_cert(domain, email, cert_dir, days=365):
    """Генерация самоподписанного сертификата"""
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    
    cert_path = cert_dir / f"{domain}.crt"
    key_path = cert_dir / f"{domain}.key"
    
    print(f"{Colors.YELLOW}Генерация самоподписанного сертификата...{Colors.END}")
    print(f"  Domain: {domain}")
    print(f"  Days: {days}")
    print(f"  Output: {cert_dir}")
    
    # OpenSSL команда
    cmd = [
        'openssl', 'req', '-x509', '-nodes', '-days', str(days),
        '-newkey', 'rsa:2048',
        '-keyout', str(key_path),
        '-out', str(cert_path),
        '-subj', f"/C=RU/ST=Moscow/L=Moscow/O=Nanoprobe Sim Lab/OU=IT/CN={domain}/emailAddress={email}",
        '-addext', f"subjectAltName=DNS:{domain},DNS:localhost,IP:127.0.0.1"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Установка правильных прав
        os.chmod(key_path, 0o600)
        os.chmod(cert_path, 0o644)
        
        print(f"\n{Colors.GREEN}✓ Сертификаты сгенерированы:{Colors.END}")
        print(f"  Certificate: {cert_path}")
        print(f"  Private Key: {key_path}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Ошибка генерации: {e.stderr}{Colors.END}")
        return False


def verify_cert(cert_path):
    """Проверка сертификата"""
    try:
        result = subprocess.run(
            ['openssl', 'x509', '-in', str(cert_path), '-text', '-noout'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Вывод информации о сертификате
        lines = result.stdout.split('\n')[:20]
        print(f"\n{Colors.YELLOW}Информация о сертификате:{Colors.END}")
        for line in lines:
            print(f"  {line}")
        
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Генерация SSL сертификатов для Nanoprobe Sim Lab',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python generate_ssl_certs.py
  python generate_ssl_certs.py nanoprobe.local
  python generate_ssl_certs.py myapp.com admin@myapp.com
        """
    )
    
    parser.add_argument(
        '--domain', '-d',
        default='nanoprobe-lab.local',
        help='Доменное имя (по умолчанию: nanoprobe-lab.local)'
    )
    parser.add_argument(
        '--email', '-e',
        default='admin@localhost',
        help='Email для сертификата (по умолчанию: admin@localhost)'
    )
    parser.add_argument(
        '--output', '-o',
        default='./deployment/nginx/ssl',
        help='Директория для сертификатов (по умолчанию: ./deployment/nginx/ssl)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Срок действия в днях (по умолчанию: 365)'
    )
    parser.add_argument(
        '--letsencrypt',
        action='store_true',
        help="Показать инструкцию для Let's Encrypt"
    )
    
    args = parser.parse_args()
    
    print_header("Генерация SSL сертификатов")
    
    # Проверка OpenSSL
    if not check_openssl():
        print(f"{Colors.RED}Ошибка: OpenSSL не найден{Colors.END}")
        print("\nУстановите OpenSSL:")
        print("  Ubuntu/Debian: sudo apt-get install openssl")
        print("  macOS: brew install openssl")
        print("  Windows: choco install openssl")
        return 1
    
    print(f"{Colors.CYAN}OpenSSL найден{Colors.END}\n")
    
    # Генерация сертификата
    success = generate_self_signed_cert(
        domain=args.domain,
        email=args.email,
        cert_dir=args.output,
        days=args.days
    )
    
    if not success:
        return 1
    
    # Проверка сертификата
    cert_path = Path(args.output) / f"{args.domain}.crt"
    verify_cert(cert_path)
    
    print(f"\n{Colors.GREEN}{'='*50}{Colors.END}")
    print(f"{Colors.GREEN}  Готово!{Colors.END}")
    print(f"{Colors.GREEN}{'='*50}{Colors.END}")
    
    if args.letsencrypt:
        print(f"\n{Colors.YELLOW}Для production используйте Let's Encrypt:{Colors.END}")
        print(f"""
  # Установка Certbot
  sudo apt-get install certbot python3-certbot-nginx
  
  # Генерация сертификата
  sudo certbot --nginx -d {args.domain}
  
  # Автоматическое обновление
  sudo certbot renew --dry-run
        """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
