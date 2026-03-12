# -*- coding: utf-8 -*-
"""
Alerting система для Nanoprobe Simulation Lab
Отправка уведомлений в Telegram, Email, Slack
"""

import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json


class AlertManager:
    """
    Менеджер уведомлений для системы алертинга
    Поддерживает Telegram, Email, Slack, Webhook
    """

    def __init__(self, config_path: str = None):
        """
        Инициализация менеджера уведомлений

        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config = self._load_config(config_path)
        self.alert_history: List[Dict] = []
        self.alert_log_path = Path('logs/alerts.log')
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str = None) -> Dict:
        """Загрузка конфигурации"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Конфигурация из переменных окружения
        return {
            'telegram': {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
                'enabled': bool(os.getenv('TELEGRAM_BOT_TOKEN')),
            },
            'email': {
                'smtp_host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'smtp_user': os.getenv('SMTP_USER'),
                'smtp_password': os.getenv('SMTP_PASSWORD'),
                'from_email': os.getenv('FROM_EMAIL', 'alerts@nanoprobe-lab.local'),
                'to_emails': os.getenv('TO_EMAILS', '').split(','),
                'enabled': bool(os.getenv('SMTP_USER')),
            },
            'slack': {
                'webhook_url': os.getenv('SLACK_WEBHOOK_URL'),
                'enabled': bool(os.getenv('SLACK_WEBHOOK_URL')),
            },
            'webhook': {
                'url': os.getenv('ALERT_WEBHOOK_URL'),
                'enabled': bool(os.getenv('ALERT_WEBHOOK_URL')),
            },
        }

    def send_alert(
        self,
        alert_name: str,
        severity: str,
        description: str,
        details: Dict = None,
        channels: List[str] = None
    ):
        """
        Отправка алерта

        Args:
            alert_name: Название алерта
            severity: Уровень серьёзности (critical, warning, info)
            description: Описание
            details: Дополнительные детали
            channels: Каналы для отправки (telegram, email, slack, webhook)
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_name': alert_name,
            'severity': severity,
            'description': description,
            'details': details or {},
            'status': 'firing',
        }

        self.alert_history.append(alert)
        self._log_alert(alert)

        # Определение каналов для отправки
        if channels is None:
            channels = self._get_channels_for_severity(severity)

        # Отправка по каналам
        results = {}

        if 'telegram' in channels and self.config['telegram']['enabled']:
            results['telegram'] = self._send_telegram(alert)

        if 'email' in channels and self.config['email']['enabled']:
            results['email'] = self._send_email(alert)

        if 'slack' in channels and self.config['slack']['enabled']:
            results['slack'] = self._send_slack(alert)

        if 'webhook' in channels and self.config['webhook']['enabled']:
            results['webhook'] = self._send_webhook(alert)

        return results

    def send_recovery(
        self,
        alert_name: str,
        severity: str,
        description: str,
        channels: List[str] = None
    ):
        """
        Отправка уведомления о восстановлении

        Args:
            alert_name: Название алерта
            severity: Уровень серьёзности
            description: Описание
            channels: Каналы для отправки
        """
        # Поиск последнего алерта
        for alert in reversed(self.alert_history):
            if alert['alert_name'] == alert_name and alert['status'] == 'firing':
                alert['status'] = 'resolved'
                alert['resolved_at'] = datetime.now().isoformat()
                self._log_alert(alert, event='resolved')

                if channels is None:
                    channels = self._get_channels_for_severity(severity)

                results = {}

                if 'telegram' in channels and self.config['telegram']['enabled']:
                    results['telegram'] = self._send_telegram(alert, is_recovery=True)

                if 'email' in channels and self.config['email']['enabled']:
                    results['email'] = self._send_email(alert, is_recovery=True)

                if 'slack' in channels and self.config['slack']['enabled']:
                    results['slack'] = self._send_slack(alert, is_recovery=True)

                return results

        return {}

    def _get_channels_for_severity(self, severity: str) -> List[str]:
        """Получение каналов для уровня серьёзности"""
        if severity == 'critical':
            return ['telegram', 'email', 'slack', 'webhook']
        elif severity == 'warning':
            return ['telegram', 'email', 'webhook']
        else:  # info
            return ['webhook']

    def _send_telegram(self, alert: Dict, is_recovery: bool = False) -> bool:
        """Отправка уведомления в Telegram"""
        try:
            config = self.config['telegram']

            emoji = {'critical': '🚨', 'warning': '⚠️', 'info': 'ℹ️'}
            status = '✅ RESOLVED' if is_recovery else '🔥 FIRING'

            message = f"""
{emoji.get(alert['severity'], '📢')} *{status}*

*Alert:* {alert['alert_name']}
*Severity:* {alert['severity'].upper()}
*Time:* {alert['timestamp'][:19]}

*Description:*
{alert['description']}

{f"*Resolved at:* {alert.get('resolved_at', 'N/A')[:19]}" if is_recovery else ""}
            """.strip()

            url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            data = {
                'chat_id': config['chat_id'],
                'text': message,
                'parse_mode': 'Markdown',
            }

            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200

        except Exception as e:
            self._log_error(f"Telegram error: {e}")
            return False

    def _send_email(self, alert: Dict, is_recovery: bool = False) -> bool:
        """Отправка Email уведомления"""
        try:
            config = self.config['email']

            subject = f"{'✅ RESOLVED' if is_recovery else '🚨 ALERT'}: {alert['alert_name']} ({alert['severity'].upper()})"

            body = f"""
<html>
<body>
<h2>{'✅ Alert Resolved' if is_recovery else '🚨 Alert Triggered'}</h2>

<table>
<tr><td><b>Alert Name:</b></td><td>{alert['alert_name']}</td></tr>
<tr><td><b>Severity:</b></td><td>{alert['severity'].upper()}</td></tr>
<tr><td><b>Time:</b></td><td>{alert['timestamp'][:19]}</td></tr>
<tr><td><b>Description:</b></td><td>{alert['description']}</td></tr>
{f"<tr><td><b>Resolved at:</b></td><td>{alert.get('resolved_at', 'N/A')[:19]}</td></tr>" if is_recovery else ""}
</table>

<h3>Details</h3>
<pre>{json.dumps(alert.get('details', {}), indent=2)}</pre>

<hr>
<p><i>Nanoprobe Simulation Lab Alerting System</i></p>
</body>
</html>
            """

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])

            msg.attach(MIMEText(body, 'html', 'utf-8'))

            server = smtplib.SMTP(config['smtp_host'], config['smtp_port'])
            server.starttls()
            server.login(config['smtp_user'], config['smtp_password'])
            server.send_message(msg)
            server.quit()

            return True

        except Exception as e:
            self._log_error(f"Email error: {e}")
            return False

    def _send_slack(self, alert: Dict, is_recovery: bool = False) -> bool:
        """Отправка уведомления в Slack"""
        try:
            config = self.config['slack']

            color = {'critical': 'danger', 'warning': 'warning', 'info': 'good'}
            status = '✅ RESOLVED' if is_recovery else '🔥 FIRING'

            payload = {
                'attachments': [
                    {
                        'color': color.get(alert['severity'], 'gray'),
                        'title': f"{status}: {alert['alert_name']}",
                        'fields': [
                            {'title': 'Severity', 'value': alert['severity'].upper(), 'short': True},
                            {'title': 'Time', 'value': alert['timestamp'][:19], 'short': True},
                            {'title': 'Description', 'value': alert['description'], 'short': False},
                        ],
                        'footer': 'Nanoprobe Sim Lab Alerting',
                        'ts': int(datetime.fromisoformat(alert['timestamp']).timestamp()),
                    }
                ]
            }

            response = requests.post(config['webhook_url'], json=payload, timeout=10)
            return response.status_code == 200

        except Exception as e:
            self._log_error(f"Slack error: {e}")
            return False

    def _send_webhook(self, alert: Dict, is_recovery: bool = False) -> bool:
        """Отправка на кастомный webhook"""
        try:
            config = self.config['webhook']

            response = requests.post(config['url'], json=alert, timeout=10)
            return response.status_code == 200

        except Exception as e:
            self._log_error(f"Webhook error: {e}")
            return False

    def _log_alert(self, alert: Dict, event: str = 'firing'):
        """Логирование алерта"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'alert': alert,
        }

        with open(self.alert_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def _log_error(self, message: str):
        """Логирование ошибки"""
        with open(self.alert_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[ERROR] {datetime.now().isoformat()}: {message}\n")

    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Получение истории алертов"""
        return self.alert_history[-limit:]

    def get_alert_statistics(self) -> Dict:
        """Получение статистики алертов"""
        stats = {
            'total': len(self.alert_history),
            'firing': sum(1 for a in self.alert_history if a['status'] == 'firing'),
            'resolved': sum(1 for a in self.alert_history if a['status'] == 'resolved'),
            'by_severity': {},
            'by_name': {},
        }

        for alert in self.alert_history:
            severity = alert['severity']
            name = alert['alert_name']

            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
            stats['by_name'][name] = stats['by_name'].get(name, 0) + 1

        return stats


# Глобальный экземпляр
_alert_manager: Optional[AlertManager] = None


def get_alert_manager(config_path: str = None) -> AlertManager:
    """Получение экземпляра AlertManager"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(config_path)
    return _alert_manager


# Функции быстрого доступа
def send_alert(name: str, severity: str, description: str, details: Dict = None):
    """Быстрая отправка алерта"""
    return get_alert_manager().send_alert(name, severity, description, details)


def send_critical_alert(name: str, description: str, details: Dict = None):
    """Отправка критического алерта"""
    return send_alert(name, 'critical', description, details)


def send_warning_alert(name: str, description: str, details: Dict = None):
    """Отправка предупреждения"""
    return send_alert(name, 'warning', description, details)


# Интеграция с FastAPI (middleware для алертинга)
class AlertingMiddleware:
    """
    Middleware для автоматического алертинга при ошибках
    """

    def __init__(self, app, alert_on_5xx: bool = True, alert_threshold: int = 5):
        self.app = app
        self.alert_on_5xx = alert_on_5xx
        self.alert_threshold = alert_threshold
        self.error_count = 0
        self.alert_manager = get_alert_manager()

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        try:
            await self.app(scope, receive, send)
        except Exception as e:
            self.error_count += 1

            if self.alert_on_5xx and self.error_count >= self.alert_threshold:
                self.alert_manager.send_alert(
                    alert_name='HighErrorRate',
                    severity='critical',
                    description=f'High error rate detected: {self.error_count} errors',
                    details={'error': str(e), 'count': self.error_count},
                )
                self.error_count = 0

            raise


if __name__ == "__main__":
    # Тестирование
    print("=== Тестирование Alerting системы ===\n")

    manager = AlertManager()

    # Тестовый алерт
    print("Отправка тестового алерта...")
    result = manager.send_alert(
        alert_name='TestAlert',
        severity='warning',
        description='Это тестовый алерт для проверки системы уведомлений',
        details={'test': True, 'value': 42},
        channels=['webhook']  # Только webhook для теста
    )

    print(f"Результат: {result}")

    # Статистика
    stats = manager.get_alert_statistics()
    print(f"\nСтатистика: {stats}")
