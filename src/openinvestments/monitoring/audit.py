"""
Comprehensive audit trail and logging system for compliance and monitoring.

Provides detailed tracking of:
- User activities and authentication events
- Model operations and parameter changes
- Risk calculations and threshold breaches
- Data access and modifications
- System events and performance metrics
- Compliance reporting and regulatory requirements
"""

import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sqlite3
import pandas as pd
from contextlib import contextmanager

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    MODEL_OPERATION = "model_operation"
    RISK_CALCULATION = "risk_calculation"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"
    PERFORMANCE_METRIC = "performance_metric"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    source: str
    action: str
    resource: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    checksum: Optional[str] = None

    def __post_init__(self):
        """Generate checksum for event integrity."""
        if self.checksum is None:
            event_data = {
                "event_id": self.event_id,
                "event_type": self.event_type.value,
                "severity": self.severity.value,
                "timestamp": self.timestamp.isoformat(),
                "user_id": self.user_id,
                "session_id": self.session_id,
                "source": self.source,
                "action": self.action,
                "resource": self.resource,
                "details": json.dumps(self.details, sort_keys=True),
                "ip_address": self.ip_address,
                "user_agent": self.user_agent,
                "success": self.success,
                "error_message": self.error_message
            }

            # Create checksum from event data
            data_str = json.dumps(event_data, sort_keys=True)
            self.checksum = hashlib.sha256(data_str.encode()).hexdigest()


class AuditTrail:
    """
    Comprehensive audit trail system with persistent storage.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize audit trail system.

        Args:
            db_path: Path to SQLite database for audit storage
        """
        if db_path is None:
            db_path = config.base_dir / "data" / "audit.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()
        self.logger = logger

    def _initialize_database(self):
        """Initialize audit database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    source TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    checksum TEXT NOT NULL,
                    created_at REAL
                )
            """)

            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON audit_events(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON audit_events(source)")

            # Create retention policy table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_config (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Set default retention period (90 days)
            conn.execute("""
                INSERT OR IGNORE INTO audit_config (key, value)
                VALUES ('retention_days', '90')
            """)

    def log_event(self, event: AuditEvent) -> bool:
        """
        Log an audit event to the database.

        Args:
            event: Audit event to log

        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_events (
                        event_id, event_type, severity, timestamp, user_id,
                        session_id, source, action, resource, details,
                        ip_address, user_agent, success, error_message,
                        checksum, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event.severity.value,
                    event.timestamp.isoformat(),
                    event.user_id,
                    event.session_id,
                    event.source,
                    event.action,
                    event.resource,
                    json.dumps(event.details),
                    event.ip_address,
                    event.user_agent,
                    event.success,
                    event.error_message,
                    event.checksum,
                    datetime.now().timestamp()
                ))

            self.logger.info(f"Audit event logged: {event.event_id} - {event.action}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            return False

    def query_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """
        Query audit events with filtering options.

        Args:
            start_date: Start date for query
            end_date: End date for query
            event_type: Filter by event type
            user_id: Filter by user ID
            session_id: Filter by session ID
            source: Filter by source
            severity: Filter by severity
            limit: Maximum number of events to return

        Returns:
            List of matching audit events
        """
        conditions = []
        params = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type.value)

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        if source:
            conditions.append("source = ?")
            params.append(source)

        if severity:
            conditions.append("severity = ?")
            params.append(severity.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT event_id, event_type, severity, timestamp, user_id,
                   session_id, source, action, resource, details,
                   ip_address, user_agent, success, error_message, checksum
            FROM audit_events
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

            events = []
            for row in rows:
                event = AuditEvent(
                    event_id=row[0],
                    event_type=AuditEventType(row[1]),
                    severity=AuditSeverity(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    user_id=row[4],
                    session_id=row[5],
                    source=row[6],
                    action=row[7],
                    resource=row[8],
                    details=json.loads(row[9]) if row[9] else {},
                    ip_address=row[10],
                    user_agent=row[11],
                    success=bool(row[12]),
                    error_message=row[13],
                    checksum=row[14]
                )
                events.append(event)

            return events

        except Exception as e:
            self.logger.error(f"Failed to query audit events: {e}")
            return []

    def get_user_activity_report(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate user activity report.

        Args:
            user_id: User ID to report on
            start_date: Start date for report
            end_date: End date for report

        Returns:
            User activity summary
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        events = self.query_events(
            start_date=start_date,
            end_date=end_date,
            user_id=user_id
        )

        # Analyze user activity
        activity_summary = {
            "user_id": user_id,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_events": len(events),
            "event_types": {},
            "daily_activity": {},
            "session_summary": {},
            "risk_events": []
        }

        # Count event types
        for event in events:
            event_type = event.event_type.value
            activity_summary["event_types"][event_type] = \
                activity_summary["event_types"].get(event_type, 0) + 1

            # Daily activity
            day = event.timestamp.date().isoformat()
            activity_summary["daily_activity"][day] = \
                activity_summary["daily_activity"].get(day, 0) + 1

            # Session summary
            if event.session_id:
                session = event.session_id
                if session not in activity_summary["session_summary"]:
                    activity_summary["session_summary"][session] = {
                        "start_time": event.timestamp,
                        "end_time": event.timestamp,
                        "event_count": 0
                    }
                activity_summary["session_summary"][session]["event_count"] += 1
                activity_summary["session_summary"][session]["end_time"] = max(
                    activity_summary["session_summary"][session]["end_time"],
                    event.timestamp
                )

            # Risk-related events
            if event.event_type in [AuditEventType.RISK_CALCULATION, AuditEventType.MODEL_OPERATION]:
                activity_summary["risk_events"].append({
                    "timestamp": event.timestamp.isoformat(),
                    "action": event.action,
                    "resource": event.resource,
                    "details": event.details
                })

        return activity_summary

    def get_compliance_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report for regulatory requirements.

        Args:
            start_date: Start date for report
            end_date: End date for report

        Returns:
            Compliance report summary
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()

        # Get all events in period
        all_events = self.query_events(start_date=start_date, end_date=end_date, limit=50000)

        # Get security events
        security_events = [e for e in all_events if e.event_type == AuditEventType.SECURITY_EVENT]

        # Get compliance events
        compliance_events = [e for e in all_events if e.event_type == AuditEventType.COMPLIANCE_EVENT]

        # Get data access events
        data_access_events = [e for e in all_events if e.event_type == AuditEventType.DATA_ACCESS]

        compliance_report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": len(all_events),
                "security_events": len(security_events),
                "compliance_events": len(compliance_events),
                "data_access_events": len(data_access_events)
            },
            "security_incidents": [],
            "compliance_violations": [],
            "data_access_summary": {},
            "user_activity_summary": {}
        }

        # Analyze security events
        for event in security_events:
            if not event.success:
                compliance_report["security_incidents"].append({
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "action": event.action,
                    "error": event.error_message,
                    "details": event.details
                })

        # Analyze compliance events
        for event in compliance_events:
            if not event.success:
                compliance_report["compliance_violations"].append({
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "violation_type": event.action,
                    "details": event.details
                })

        # Analyze data access patterns
        access_by_user = {}
        access_by_resource = {}

        for event in data_access_events:
            user = event.user_id or "unknown"
            resource = event.resource or "unknown"

            if user not in access_by_user:
                access_by_user[user] = 0
            access_by_user[user] += 1

            if resource not in access_by_resource:
                access_by_resource[resource] = 0
            access_by_resource[resource] += 1

        compliance_report["data_access_summary"] = {
            "access_by_user": access_by_user,
            "access_by_resource": access_by_resource,
            "total_access_events": len(data_access_events)
        }

        return compliance_report

    def cleanup_old_events(self, retention_days: int = 90) -> int:
        """
        Clean up old audit events based on retention policy.

        Args:
            retention_days: Number of days to retain events

        Returns:
            Number of events deleted
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM audit_events
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))

                deleted_count = cursor.rowcount

            self.logger.info(f"Cleaned up {deleted_count} old audit events")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup audit events: {e}")
            return 0

    def export_events(
        self,
        output_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json"
    ) -> bool:
        """
        Export audit events to file.

        Args:
            output_path: Path to output file
            start_date: Start date for export
            end_date: End date for export
            format: Export format ('json', 'csv')

        Returns:
            Success status
        """
        events = self.query_events(
            start_date=start_date,
            end_date=end_date,
            limit=100000  # Large limit for export
        )

        try:
            if format == "json":
                export_data = [self._event_to_dict(event) for event in events]
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)

            elif format == "csv":
                export_data = [self._event_to_dict(event) for event in events]
                df = pd.DataFrame(export_data)
                df.to_csv(output_path, index=False)

            self.logger.info(f"Exported {len(events)} audit events to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export audit events: {e}")
            return False

    def _event_to_dict(self, event: AuditEvent) -> Dict[str, Any]:
        """Convert audit event to dictionary for export."""
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "session_id": event.session_id,
            "source": event.source,
            "action": event.action,
            "resource": event.resource,
            "details": event.details,
            "ip_address": event.ip_address,
            "user_agent": event.user_agent,
            "success": event.success,
            "error_message": event.error_message,
            "checksum": event.checksum
        }


class AuditLogger:
    """
    Convenient logger for audit events with automatic context capture.
    """

    def __init__(self, audit_trail: AuditTrail):
        self.audit_trail = audit_trail
        self.context = {
            "user_id": None,
            "session_id": None,
            "ip_address": None,
            "user_agent": None
        }

    def set_context(self, **kwargs):
        """Set audit context for subsequent events."""
        self.context.update(kwargs)

    def log_user_action(
        self,
        action: str,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log user action event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.USER_ACTION,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            timestamp=datetime.now(),
            user_id=self.context.get("user_id"),
            session_id=self.context.get("session_id"),
            source="user_interface",
            action=action,
            resource=resource,
            details=details or {},
            ip_address=self.context.get("ip_address"),
            user_agent=self.context.get("user_agent"),
            success=success,
            error_message=error_message
        )

        self.audit_trail.log_event(event)

    def log_model_operation(
        self,
        operation: str,
        model_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log model operation event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.MODEL_OPERATION,
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            timestamp=datetime.now(),
            user_id=self.context.get("user_id"),
            session_id=self.context.get("session_id"),
            source="model_engine",
            action=operation,
            resource=model_name,
            details={"parameters": parameters} if parameters else {},
            ip_address=self.context.get("ip_address"),
            user_agent=self.context.get("user_agent"),
            success=success,
            error_message=error_message
        )

        self.audit_trail.log_event(event)

    def log_risk_calculation(
        self,
        calculation_type: str,
        portfolio_id: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log risk calculation event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.RISK_CALCULATION,
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            timestamp=datetime.now(),
            user_id=self.context.get("user_id"),
            session_id=self.context.get("session_id"),
            source="risk_engine",
            action=calculation_type,
            resource=portfolio_id,
            details={"results": results} if results else {},
            ip_address=self.context.get("ip_address"),
            user_agent=self.context.get("user_agent"),
            success=success,
            error_message=error_message
        )

        self.audit_trail.log_event(event)

    def log_security_event(
        self,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.WARNING
    ):
        """Log security-related event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.SECURITY_EVENT,
            severity=severity,
            timestamp=datetime.now(),
            user_id=self.context.get("user_id"),
            session_id=self.context.get("session_id"),
            source="security_system",
            action=event_type,
            resource=None,
            details=details or {},
            ip_address=self.context.get("ip_address"),
            user_agent=self.context.get("user_agent"),
            success=True
        )

        self.audit_trail.log_event(event)


# Global audit system instances
audit_trail = AuditTrail()
audit_logger = AuditLogger(audit_trail)


def get_audit_trail() -> AuditTrail:
    """Get the global audit trail instance."""
    return audit_trail


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    return audit_logger


@contextmanager
def audit_context(**kwargs):
    """
    Context manager for setting audit context.

    Usage:
        with audit_context(user_id="user123", session_id="session456"):
            # Audit events will include this context
            audit_logger.log_user_action("viewed_portfolio", "PORT001")
    """
    previous_context = audit_logger.context.copy()
    audit_logger.set_context(**kwargs)

    try:
        yield
    finally:
        audit_logger.context = previous_context
