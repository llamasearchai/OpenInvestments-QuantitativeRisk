"""
Automated alerting system for risk management and model monitoring.

Provides real-time alerts for:
- Risk threshold breaches (VaR, CVaR, drawdown limits)
- Model performance degradation (drift detection)
- Market anomaly detection
- Portfolio rebalancing triggers
- Compliance violations
- System health monitoring
"""

import asyncio
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import slack_sdk
from slack_sdk.webhook import WebhookClient
import requests
from abc import ABC, abstractmethod

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    RISK_THRESHOLD = "risk_threshold"
    MODEL_DRIFT = "model_drift"
    MARKET_ANOMALY = "market_anomaly"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_HEALTH = "system_health"
    DATA_QUALITY = "data_quality"


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: Callable[[Any], bool]
    cooldown_period: timedelta = timedelta(minutes=15)
    enabled: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertChannel(ABC):
    """Abstract base class for alert notification channels."""

    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if channel is properly configured."""
        pass


class EmailChannel(AlertChannel):
    """Email notification channel."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str]
    ):
        """
        Initialize email channel.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: Sender email address
            to_emails: List of recipient email addresses
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange' if alert.severity == AlertSeverity.HIGH else 'blue'};">{alert.title}</h2>
                <p><strong>Type:</strong> {alert.alert_type.value}</p>
                <p><strong>Severity:</strong> {alert.severity.value}</p>
                <p><strong>Timestamp:</strong> {alert.timestamp.isoformat()}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Message:</strong></p>
                <p>{alert.message}</p>
                <hr>
                <h3>Metadata:</h3>
                <pre>{json.dumps(alert.metadata, indent=2)}</pre>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_body, 'html'))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()

            logger.info(f"Alert email sent successfully: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False

    def is_configured(self) -> bool:
        """Check if email channel is configured."""
        return all([
            self.smtp_server,
            self.username,
            self.password,
            self.from_email,
            self.to_emails
        ])


class SlackChannel(AlertChannel):
    """Slack notification channel."""

    def __init__(self, webhook_url: str, channel: str = None, username: str = "Risk Monitor"):
        """
        Initialize Slack channel.

        Args:
            webhook_url: Slack webhook URL
            channel: Slack channel (optional, can be specified in webhook)
            username: Bot username
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.client = WebhookClient(webhook_url) if webhook_url else None

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via Slack."""
        if not self.client:
            return False

        try:
            # Create color based on severity
            color_map = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "#ff0000"
            }

            # Create Slack message
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"ALERT: {alert.title}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Type:*\n{alert.alert_type.value}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:*\n{alert.severity.value.upper()}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Source:*\n{alert.source}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": alert.message
                    }
                }
            ]

            # Add metadata if present
            if alert.metadata:
                metadata_text = "\n".join([f"• *{k}:* {v}" for k, v in alert.metadata.items()])
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Details:*\n{metadata_text}"
                    }
                })

            # Send message
            response = self.client.send(
                text=f"Alert: {alert.title}",
                blocks=blocks
            )

            if response.status_code == 200:
                logger.info(f"Alert sent to Slack successfully: {alert.alert_id}")
                return True
            else:
                logger.error(f"Failed to send Slack alert: {response.body}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def is_configured(self) -> bool:
        """Check if Slack channel is configured."""
        return self.webhook_url is not None


class WebhookChannel(AlertChannel):
    """Generic webhook notification channel."""

    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        """
        Initialize webhook channel.

        Args:
            webhook_url: Webhook URL
            headers: Additional headers for webhook request
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "metadata": alert.metadata,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            }

            headers = {"Content-Type": "application/json", **self.headers}

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=headers,
                timeout=10
            )

            if response.status_code in [200, 201, 202]:
                logger.info(f"Alert sent via webhook successfully: {alert.alert_id}")
                return True
            else:
                logger.error(f"Webhook alert failed with status {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    def is_configured(self) -> bool:
        """Check if webhook channel is configured."""
        return self.webhook_url is not None


class AlertManager:
    """
    Central alert management system.

    Manages alert rules, triggers alerts, and coordinates notification channels.
    """

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, AlertChannel] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_triggered: Dict[str, datetime] = {}

        # Alert counters
        self.alert_counts = {
            AlertSeverity.LOW: 0,
            AlertSeverity.MEDIUM: 0,
            AlertSeverity.HIGH: 0,
            AlertSeverity.CRITICAL: 0
        }

        self.logger = logger

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str) -> None:
        """Remove an alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")

    def add_channel(self, name: str, channel: AlertChannel) -> None:
        """Add a notification channel."""
        self.channels[name] = channel
        self.logger.info(f"Added alert channel: {name}")

    def remove_channel(self, name: str) -> None:
        """Remove a notification channel."""
        if name in self.channels:
            del self.channels[name]
            self.logger.info(f"Removed alert channel: {name}")

    async def check_condition(self, rule: AlertRule, data: Any) -> Optional[Alert]:
        """
        Check if an alert rule condition is met.

        Args:
            rule: Alert rule to check
            data: Data to check against rule condition

        Returns:
            Alert if condition is met, None otherwise
        """
        try:
            # Check cooldown period
            if rule.rule_id in self.last_triggered:
                time_since_last = datetime.now() - self.last_triggered[rule.rule_id]
                if time_since_last < rule.cooldown_period:
                    return None

            # Evaluate condition
            if rule.condition(data):
                # Create alert
                alert = Alert(
                    alert_id=f"{rule.rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    title=rule.name,
                    message=f"Alert condition met: {rule.description}",
                    timestamp=datetime.now(),
                    source="AlertManager",
                    metadata={
                        "rule_id": rule.rule_id,
                        "rule_description": rule.description,
                        "trigger_data": str(data)[:500]  # Limit data size
                    }
                )

                self.last_triggered[rule.rule_id] = datetime.now()
                return alert

        except Exception as e:
            self.logger.error(f"Error checking alert rule {rule.rule_id}: {e}")

        return None

    async def trigger_alert(self, alert: Alert) -> None:
        """
        Trigger an alert and send notifications.

        Args:
            alert: Alert to trigger
        """
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_counts[alert.severity] += 1

        # Send to all configured channels
        sent_count = 0
        for channel_name, channel in self.channels.items():
            if channel.is_configured():
                success = await channel.send_alert(alert)
                if success:
                    sent_count += 1
                    self.logger.info(f"Alert sent via {channel_name}")
                else:
                    self.logger.error(f"Failed to send alert via {channel_name}")

        alert.metadata["channels_notified"] = sent_count
        self.logger.info(f"Alert triggered: {alert.alert_id} (sent to {sent_count} channels)")

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()

            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True

        return False

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()

            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]

            self.logger.info(f"Alert resolved: {alert_id}")
            return True

        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get alert history with optional filtering."""
        alerts = self.alert_history + list(self.active_alerts.values())

        if start_date:
            alerts = [a for a in alerts if a.timestamp >= start_date]
        if end_date:
            alerts = [a for a in alerts if a.timestamp <= end_date]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history) + len(self.active_alerts)
        active_alerts = len(self.active_alerts)

        # Severity breakdown
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in self.alert_history + list(self.active_alerts.values()):
            severity_counts[alert.severity.value] += 1

        # Type breakdown
        type_counts = {alert_type.value: 0 for alert_type in AlertType}
        for alert in self.alert_history + list(self.active_alerts.values()):
            type_counts[alert.alert_type.value] += 1

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "alerts_today": len([a for a in self.alert_history + list(self.active_alerts.values())
                               if a.timestamp.date() == datetime.now().date()])
        }


class RiskAlertRules:
    """Pre-defined risk alert rules."""

    @staticmethod
    def var_breach_threshold(
        threshold: float,
        confidence_level: float = 0.95
    ) -> AlertRule:
        """Create VaR breach alert rule."""
        def condition(data):
            if isinstance(data, dict) and 'var' in data:
                return data['var'] > threshold
            return False

        return AlertRule(
            rule_id="var_breach",
            name="VaR Threshold Breach",
            alert_type=AlertType.RISK_THRESHOLD,
            severity=AlertSeverity.HIGH,
            condition=condition,
            description=f"VaR exceeds {threshold:.1%} threshold at {confidence_level:.1%} confidence"
        )

    @staticmethod
    def drawdown_limit(drawdown_limit: float) -> AlertRule:
        """Create drawdown limit alert rule."""
        def condition(data):
            if isinstance(data, dict) and 'drawdown' in data:
                return abs(data['drawdown']) > drawdown_limit
            return False

        return AlertRule(
            rule_id="drawdown_limit",
            name="Drawdown Limit Exceeded",
            alert_type=AlertType.RISK_THRESHOLD,
            severity=AlertSeverity.CRITICAL,
            condition=condition,
            description=f"Portfolio drawdown exceeds {drawdown_limit:.1%} limit"
        )

    @staticmethod
    def model_performance_drift(
        performance_threshold: float,
        metric: str = "mape"
    ) -> AlertRule:
        """Create model performance drift alert rule."""
        def condition(data):
            if isinstance(data, dict) and metric in data:
                return data[metric] > performance_threshold
            return False

        return AlertRule(
            rule_id="model_drift",
            name="Model Performance Drift",
            alert_type=AlertType.MODEL_DRIFT,
            severity=AlertSeverity.MEDIUM,
            condition=condition,
            description=f"Model {metric} exceeds {performance_threshold:.1%} threshold"
        )

    @staticmethod
    def volatility_spike(spike_threshold: float) -> AlertRule:
        """Create volatility spike alert rule."""
        def condition(data):
            if isinstance(data, dict) and 'volatility' in data:
                return data['volatility'] > spike_threshold
            return False

        return AlertRule(
            rule_id="volatility_spike",
            name="Volatility Spike Detected",
            alert_type=AlertType.MARKET_ANOMALY,
            severity=AlertSeverity.HIGH,
            condition=condition,
            description=f"Market volatility exceeds {spike_threshold:.1%} threshold"
        )

    @staticmethod
    def portfolio_rebalance_trigger(
        rebalance_threshold: float,
        check_frequency: timedelta = timedelta(days=1)
    ) -> AlertRule:
        """Create portfolio rebalance trigger alert rule."""
        def condition(data):
            if isinstance(data, dict) and 'deviation' in data:
                return data['deviation'] > rebalance_threshold
            return False

        return AlertRule(
            rule_id="rebalance_trigger",
            name="Portfolio Rebalance Required",
            alert_type=AlertType.PORTFOLIO_REBALANCE,
            severity=AlertSeverity.MEDIUM,
            condition=condition,
            cooldown_period=check_frequency,
            description=f"Portfolio deviation exceeds {rebalance_threshold:.1%} rebalance threshold"
        )


# Global alert manager instance
alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    return alert_manager
