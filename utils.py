"""
Utility functions for the QA Dashboard application.

This module provides logging, metrics tracking, and error handling utilities.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Metrics file
metrics_file = log_dir / "usage_metrics.json"


def load_metrics() -> Dict[str, Any]:
    """Load usage metrics from file.
    
    Returns:
        Dictionary containing session count, errors, feature usage, and last update time.
    """
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return {"sessions": 0, "errors": {}, "features_used": {}, "last_updated": None}
    return {"sessions": 0, "errors": {}, "features_used": {}, "last_updated": None}


def save_metrics(metrics: Dict[str, Any]) -> None:
    """Save usage metrics to file.
    
    Args:
        metrics: Dictionary containing metrics data to save.
    """
    metrics["last_updated"] = datetime.now().isoformat()
    try:
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")


def track_feature_usage(feature_name: str) -> None:
    """Track feature usage for analytics.
    
    Args:
        feature_name: Name of the feature being used.
    """
    metrics = load_metrics()
    if "features_used" not in metrics:
        metrics["features_used"] = {}
    metrics["features_used"][feature_name] = metrics["features_used"].get(feature_name, 0) + 1
    save_metrics(metrics)


def track_error(error_type: str, error_message: str) -> None:
    """Track errors for alerting on repeated failures.
    
    Args:
        error_type: Type/category of the error.
        error_message: Error message or description.
    """
    metrics = load_metrics()
    if "errors" not in metrics:
        metrics["errors"] = {}
    
    error_key = f"{error_type}:{error_message[:50]}"
    if error_key not in metrics["errors"]:
        metrics["errors"][error_key] = {
            "count": 0,
            "first_seen": datetime.now().isoformat(),
            "last_seen": None
        }
    
    metrics["errors"][error_key]["count"] += 1
    metrics["errors"][error_key]["last_seen"] = datetime.now().isoformat()
    save_metrics(metrics)
    
    # Log error
    logger.error(f"{error_type}: {error_message}")
    
    # Alert on repeated failures (5+ occurrences)
    if metrics["errors"][error_key]["count"] >= 5:
        logger.warning(
            f"ALERT: Repeated failure detected - {error_key} "
            f"(count: {metrics['errors'][error_key]['count']})"
        )


def log_audit_event(username: str, action: str, details: Optional[str] = None) -> None:
    """Log audit events for admin users (Shannon and Chloe only).
    
    Args:
        username: Username performing the action.
        details: Additional details about the action.
    """
    admin_users = ["chloe", "shannon"]
    if username.lower() not in admin_users:
        return  # Only log for admins
    
    audit_file = log_dir / "audit_log.json"
    audit_log = []
    
    if audit_file.exists():
        try:
            with open(audit_file, 'r') as f:
                audit_log = json.load(f)
        except:
            audit_log = []
    
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "action": action,
        "details": details
    }
    
    audit_log.append(audit_entry)
    
    # Keep only last 1000 entries
    if len(audit_log) > 1000:
        audit_log = audit_log[-1000:]
    
    try:
        with open(audit_file, 'w') as f:
            json.dump(audit_log, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save audit log: {e}")


def check_session_timeout(last_activity: float, timeout_minutes: int = 30) -> bool:
    """Check if session has timed out due to inactivity.
    
    Args:
        last_activity: Timestamp of last activity.
        timeout_minutes: Minutes of inactivity before timeout.
        
    Returns:
        True if session has timed out, False otherwise.
    """
    if last_activity == 0:
        return False
    
    elapsed_minutes = (time.time() - last_activity) / 60
    return elapsed_minutes > timeout_minutes


def normalize_agent_id(agent_str: str) -> str:
    """Normalize agent ID to bpagent## format.
    
    Args:
        agent_str: Agent ID string (may contain extra characters).
        
    Returns:
        Normalized agent ID in bpagent## format, or original string if no match.
    """
    import re
    import pandas as pd
    
    if pd.isna(agent_str) or not agent_str:
        return agent_str
    
    agent_str = str(agent_str).lower().strip()
    match = re.search(r'bpagent(\d+)', agent_str)
    
    if match:
        agent_num = match.group(1)
        if len(agent_num) >= 2:
            agent_num = agent_num[:2]
        else:
            agent_num = agent_num.zfill(2)
        return f"bpagent{agent_num}"
    
    return agent_str

