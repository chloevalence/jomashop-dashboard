import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta, date
from pandas import ExcelWriter
import streamlit_authenticator as stauth
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import botocore.config
import json
import time
import re
import os
import sys
import logging
import shutil
from pathlib import Path
from contextlib import contextmanager
from utils import (
    log_audit_event,
    check_session_timeout,
)
import warnings

# File locking imports (platform-specific)
try:
    if sys.platform == "win32":
        import msvcrt

        HAS_FILE_LOCKING = True
    else:
        import fcntl

        HAS_FILE_LOCKING = True
except ImportError:
    HAS_FILE_LOCKING = False

# --- Project root directory (portable across systems) ---
PROJECT_ROOT = Path(__file__).parent.resolve()
DEBUG_LOG_FILE = PROJECT_ROOT / ".cursor" / "debug.log"

# --- Create .cursor directory for debug logging ---
try:
    DEBUG_LOG_FILE.parent.mkdir(exist_ok=True)
except Exception:
    # Silently fail - debug logging is optional
    pass

# --- Configuration: Limit for faster testing (set to None to load all files) ---
MAX_FILES_FOR_TESTING = None  # Set to None to disable limit

# --- Logging Setup ---
log_dir = Path("logs")
try:
    log_dir.mkdir(exist_ok=True)
except Exception as e:
    # Fallback: try to create in current directory if logs/ fails
    print(f"Warning: Could not create logs directory: {e}")
    log_dir = Path(".")
log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.debug(f"Logging initialized. Log file: {log_file}")

# Suppress noisy matplotlib categorical-unit warnings about string data that look like numbers/dates
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.category")

# Suppress matplotlib.category INFO level logging messages (categorical units warnings)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

# Configure matplotlib to automatically close figures and suppress memory warnings
matplotlib.rcParams["figure.max_open_warning"] = (
    0  # Suppress the "More than 20 figures" warning
)
# Set backend to non-interactive to reduce memory usage
try:
    matplotlib.use("Agg")  # Use non-interactive backend
except Exception:
    pass  # Backend may already be set, ignore

# Suppress inotify errors (file watcher limit reached) - these are non-critical
warnings.filterwarnings("ignore", message=".*inotify.*")
warnings.filterwarnings("ignore", message=".*fileWatcherType.*")


# --- File Locking (must be defined before load_metrics/save_metrics) ---
class LockTimeoutError(Exception):
    """Raised when file lock acquisition times out."""

    pass


@contextmanager
def cache_file_lock(filepath, timeout=30):
    """Acquire file lock for cache operations. Prevents concurrent reads/writes.

    Args:
        filepath: Path to the file to lock
        timeout: Maximum time to wait for lock (seconds). None = wait indefinitely (with max iteration limit).

    Yields:
        Locked file handle (or None if locking unavailable)

    Raises:
        LockTimeoutError: If lock cannot be acquired within timeout period (only if timeout is not None)
    """
    lock_path = filepath.with_suffix(filepath.suffix + ".lock")
    lock_file = None
    lock_acquired = False

    if not HAS_FILE_LOCKING:
        # Fallback: no locking available, just yield
        yield None
        return

    try:
        # Create lock file
        lock_file = open(lock_path, "w")
        start_time = time.time()

        # BUG FIX: Handle timeout=None with maximum iteration limit to prevent infinite hang
        # Maximum iterations = 5 minutes worth of retries (3000 iterations at 0.1s each)
        MAX_ITERATIONS = 3000 if timeout is None else None

        # Try to acquire lock
        iteration_count = 0
        while timeout is None or (time.time() - start_time < timeout):
            # BUG FIX: Add iteration limit for timeout=None to prevent infinite hang
            if timeout is None and MAX_ITERATIONS is not None:
                iteration_count += 1
                if iteration_count >= MAX_ITERATIONS:
                    # Reached maximum iterations - break to raise timeout error
                    break

            try:
                if sys.platform == "win32":
                    # Windows file locking
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Unix file locking
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
                break
            except (IOError, OSError):
                # Lock is held by another process, wait and retry
                time.sleep(0.1)

        if not lock_acquired:
            # Close the file before raising exception
            # This can only happen if timeout is not None and timeout expired
            try:
                lock_file.close()
            except Exception:
                pass
            lock_file = None
            if timeout is None:
                raise LockTimeoutError(
                    f"Could not acquire lock for {filepath} after {MAX_ITERATIONS * 0.1:.0f}s (max wait time). Another process may be holding the lock indefinitely."
                )
            else:
                raise LockTimeoutError(
                    f"Could not acquire lock for {filepath} within {timeout}s. Another process may be accessing the cache file."
                )

        yield lock_file

    except LockTimeoutError:
        # Re-raise LockTimeoutError - don't catch it, let it propagate to callers
        raise
    except Exception as e:
        # Only catch other unexpected errors, not LockTimeoutError
        logger.warning(f" Error acquiring file lock: {e}, proceeding without lock")
        yield None
    finally:
        # Release lock and close file
        if lock_file:
            # Only unlock if lock was actually acquired
            if lock_acquired:
                try:
                    if sys.platform == "win32":
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            # Always close the file to prevent resource leak
            try:
                lock_file.close()
            except Exception:
                pass

        # Remove lock file
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass


def atomic_write_json(filepath, data, max_retries=3, retry_delay=0.1):
    """Write JSON atomically using temp file + rename pattern.

    This prevents partial writes from corrupting the cache file.

    Args:
        filepath: Path to the JSON file to write
        data: Data to serialize to JSON
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries (seconds)

    Raises:
        Exception: If all retry attempts fail
    """
    # Ensure parent directory exists before writing
    filepath.parent.mkdir(parents=True, exist_ok=True)

    temp_path = filepath.with_suffix(filepath.suffix + ".tmp")

    # Clean up any existing temp file from previous failed attempts
    if temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors

    for attempt in range(max_retries):
        try:
            # Write to temporary file first
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Atomic rename (replaces existing file atomically on most systems)
            os.replace(temp_path, filepath)
            return

        except (IOError, OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f" Atomic write attempt {attempt + 1} failed: {e}, retrying..."
                )
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                # Clean up temp file on final failure
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                raise


# --- Usage Metrics Tracking ---
metrics_file = log_dir / "usage_metrics.json"


def load_metrics():
    """Load usage metrics from file."""
    if metrics_file.exists():
        try:
            with cache_file_lock(metrics_file, timeout=1):
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)
                    if not isinstance(metrics_data, dict):
                        logger.warning(
                            "Metrics file contains invalid data structure: "
                            f"{type(metrics_data).__name__}, expected dict"
                        )
                        return {
                            "sessions": 0,
                            "errors": {},
                            "features_used": {},
                            "last_updated": None,
                        }
                    return metrics_data
        except (LockTimeoutError, FileNotFoundError, json.JSONDecodeError):
            return {
                "sessions": 0,
                "errors": {},
                "features_used": {},
                "last_updated": None,
            }
        except Exception:
            return {
                "sessions": 0,
                "errors": {},
                "features_used": {},
                "last_updated": None,
            }
    return {
        "sessions": 0,
        "errors": {},
        "features_used": {},
        "last_updated": None,
    }


def save_metrics(metrics):
    """Save usage metrics to file."""
    metrics["last_updated"] = datetime.now().isoformat()
    try:
        with cache_file_lock(metrics_file, timeout=1):
            atomic_write_json(metrics_file, metrics)
    except LockTimeoutError:
        logger.warning("Could not acquire lock for metrics file, skipping save")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")


def track_feature_usage(feature_name):
    """Track feature usage."""
    metrics = load_metrics()
    if "features_used" not in metrics:
        metrics["features_used"] = {}
    metrics["features_used"][feature_name] = (
        metrics["features_used"].get(feature_name, 0) + 1
    )
    save_metrics(metrics)


def track_error(error_type, error_message):
    """Track errors for alerting on repeated failures."""
    metrics = load_metrics()
    if "errors" not in metrics:
        metrics["errors"] = {}

    error_key = f"{error_type}:{error_message[:50]}"
    if error_key not in metrics["errors"]:
        metrics["errors"][error_key] = {
            "count": 0,
            "first_seen": datetime.now().isoformat(),
            "last_seen": None,
        }

    metrics["errors"][error_key]["count"] += 1
    metrics["errors"][error_key]["last_seen"] = datetime.now().isoformat()
    save_metrics(metrics)

    # Log error
    logger.error(f"{error_type}: {error_message}")

    # Alert on repeated failures (5+ occurrences)
    if metrics["errors"][error_key]["count"] >= 5:
        logger.warning(
            f"ALERT: Repeated failure detected - {error_key} (count: {metrics['errors'][error_key]['count']})"
        )


# Initialize session metrics
if "session_started" not in st.session_state:
    metrics = load_metrics()
    metrics["sessions"] = metrics.get("sessions", 0) + 1
    save_metrics(metrics)
    st.session_state.session_started = True
    logger.info(f"New session started. Total sessions: {metrics['sessions']}")

st.set_page_config(page_title="Emotion Dashboard", layout="wide")
logger.debug("Page config set, starting app initialization...")


def st_pyplot_safe(fig, **kwargs):
    """Display matplotlib figure in Streamlit and automatically close it to prevent memory leaks.

    Args:
        fig: matplotlib figure object
        **kwargs: Additional arguments passed to st.pyplot()
    """
    try:
        st.pyplot(fig, **kwargs)
    except Exception as e:
        # Handle MediaFileStorageError when Streamlit tries to access a stale figure reference
        # This can happen when the page reruns and old figure references are no longer valid
        error_str = str(type(e).__name__) + ": " + str(e)
        if (
            "MediaFileStorageError" in error_str
            or "media file" in error_str.lower()
            or "No media file with id" in error_str
        ):
            # Silently ignore stale figure references - this is expected behavior
            # when filters change and the page reruns
            logger.debug(f"Ignoring stale matplotlib figure reference: {error_str}")
        else:
            # Re-raise other exceptions
            logger.warning(f"Error displaying matplotlib figure: {e}")
            raise
    finally:
        plt.close(fig)
        # Close all remaining open figures to prevent accumulation
        plt.close("all")
        # Force garbage collection of figure to free memory immediately
        import gc

        gc.collect()


# Show immediate feedback - app is loading
initial_status = st.empty()
initial_status.info(" **Initializing dashboard...** Please wait.")

# Initialize S3 client from secrets (but don't test connection yet - do that after login)
logger.debug("Initializing S3 client from secrets...")
try:
    # Check if secrets are available
    if "s3" not in st.secrets:
        raise KeyError("s3 section not found in secrets")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
        region_name=st.secrets["s3"].get("region_name", "us-east-1"),
    )
    s3_bucket_name = st.secrets["s3"]["bucket_name"]
    s3_prefix = st.secrets["s3"].get("prefix", "")  # Optional prefix/folder path
    logger.debug(
        f"S3 client initialized. Bucket: {s3_bucket_name}, Prefix: {s3_prefix}"
    )
    initial_status.empty()  # Clear initial status once S3 client is ready
except KeyError as e:
    logger.error(f"Missing S3 configuration in secrets: {e}")
    initial_status.empty()
    st.error(f" Missing S3 configuration in secrets: {e}")
    st.error(
        "Please check your `.streamlit/secrets.toml` file and ensure all S3 fields are set."
    )
    st.error(f"**Current working directory:** `{os.getcwd()}`")
    st.error(
        "**Expected secrets path:** `.streamlit/secrets.toml` in the project directory"
    )
    st.error(
        f"**Make sure you're running Streamlit from the project root directory:** `{PROJECT_ROOT}`"
    )
    st.stop()
except Exception as e:
    logger.exception(f"Error initializing S3 client: {e}")
    initial_status.empty()
    st.error(f" Error initializing S3 client: {e}")
    st.error("Please check your AWS credentials and try again.")
    st.stop()


# --- Agent ID Mapping System ---
AGENT_MAPPING_FILE = log_dir / "agent_id_mapping.json"

# Known agent mappings (from user specification)
# Include both versions (with and without spaces) to handle data inconsistencies
KNOWN_AGENT_MAPPINGS = {
    # Agent 1: Jesus
    "unknown": "Agent 1",
    "bp016803073": "Agent 1",
    "bp016803074": "Agent 1",
    "bp agent 016803073": "Agent 1",  # Space version
    "bp agent 016803074": "Agent 1",  # Space version
    # Agent 2: Gerardo
    "bpagent024577540": "Agent 2",
    "bp agent 024577540": "Agent 2",  # Space version
    # Agent 3: Edgar
    "bpagent030844482": "Agent 3",
    "bp agent 030844482": "Agent 3",  # Space version
    # Agent 4: Osiris
    "bpagent047779576": "Agent 4",
    "bp agent 047779576": "Agent 4",  # Space version
    # Agent 6: Daniela
    "bpagent065185612": "Agent 6",
    "bp agent 065185612": "Agent 6",  # Space version
    # Agent 7: Yasmin
    "bpagent072229421": "Agent 7",
    "bp agent 072229421": "Agent 7",  # Space version
    # Agent 8: Moises
    "bpagent089724913": "Agent 8",
    "bp agent 089724913": "Agent 8",  # Space version
    # Agent 9: Marcos
    "bpagent093540654": "Agent 9",
    "bp agent 093540654": "Agent 9",  # Space version
    # Agent 10: Angel
    "bpagent102256681": "Agent 10",
    "bp agent 102256681": "Agent 10",  # Space version
    # Agent 5: (Left, but calls need to be accessible to admins)
    "bpagent051705087": "Agent 5",
    "bp agent 051705087": "Agent 5",  # Space version
    # Agent 11: (No agent account, but viewable to admins)
    "bpagent113827380": "Agent 11",
    "bp agent 113827380": "Agent 11",  # Space version
}


def load_agent_mapping():
    """Load agent ID mapping from file, or return known mappings if file doesn't exist."""
    if AGENT_MAPPING_FILE.exists():
        try:
            with open(AGENT_MAPPING_FILE, "r", encoding="utf-8") as f:
                file_content = f.read()
                if not file_content.strip():
                    logger.warning(
                        "Agent mapping file is empty, using known mappings only"
                    )
                    return KNOWN_AGENT_MAPPINGS.copy()

                mapping = json.loads(file_content)

                # Validate that mapping is a dictionary
                if not isinstance(mapping, dict):
                    logger.warning(
                        f"Agent mapping file contains invalid data type: {type(mapping).__name__}, "
                        "expected dict. Using known mappings only."
                    )
                    # Try to backup corrupted file
                    try:
                        backup_path = AGENT_MAPPING_FILE.with_suffix(".json.backup")
                        AGENT_MAPPING_FILE.rename(backup_path)
                        logger.info(
                            f"Backed up corrupted mapping file to {backup_path}"
                        )
                    except Exception as backup_error:
                        logger.warning(
                            f"Could not backup corrupted mapping file: {backup_error}"
                        )
                    return KNOWN_AGENT_MAPPINGS.copy()

                # Validate mapping values are strings in "Agent X" format
                valid_mapping = {}
                for key, value in mapping.items():
                    if isinstance(value, str) and value.startswith("Agent "):
                        valid_mapping[key] = value
                    else:
                        logger.warning(
                            f"Invalid mapping value for '{key}': {value} (expected 'Agent X' format)"
                        )

                # Merge with known mappings (known mappings take precedence)
                # This ensures KNOWN_AGENT_MAPPINGS always override file mappings
                # Also, remove any incorrect mappings for known agents from the file
                merged_mapping = {}
                # First, add all file mappings
                for key, value in valid_mapping.items():
                    key_normalized = key.replace(" ", "").replace("_", "").lower()
                    # Check if this key matches a known agent ID
                    is_known_agent = False
                    for known_id in KNOWN_AGENT_MAPPINGS.keys():
                        known_id_normalized = (
                            known_id.replace(" ", "").replace("_", "").lower()
                        )
                        if (
                            key_normalized == known_id_normalized
                            or key.lower() == known_id.lower()
                        ):
                            is_known_agent = True
                            break
                    # Only add if it's not a known agent (known mappings will override)
                    if not is_known_agent:
                        merged_mapping[key] = value
                # Then, add known mappings (these take precedence)
                merged_mapping.update(KNOWN_AGENT_MAPPINGS)
                return merged_mapping
        except json.JSONDecodeError as e:
            logger.warning(
                f"Agent mapping file contains invalid JSON: {e}, using known mappings only"
            )
            # Try to backup corrupted file
            try:
                backup_path = AGENT_MAPPING_FILE.with_suffix(".json.backup")
                AGENT_MAPPING_FILE.rename(backup_path)
                logger.info(f"Backed up corrupted mapping file to {backup_path}")
            except Exception as backup_error:
                logger.warning(
                    f"Could not backup corrupted mapping file: {backup_error}"
                )
            return KNOWN_AGENT_MAPPINGS.copy()
        except Exception as e:
            logger.warning(
                f"Failed to load agent mapping file: {e}, using known mappings only"
            )
            return KNOWN_AGENT_MAPPINGS.copy()
    return KNOWN_AGENT_MAPPINGS.copy()


def save_agent_mapping(mapping):
    """Save agent ID mapping to file, removing any incorrect mappings for known agents."""
    try:
        # Clean up: remove any mappings for known agents (they should use KNOWN_AGENT_MAPPINGS)
        cleaned_mapping = {}
        for key, value in mapping.items():
            key_normalized = key.replace(" ", "").replace("_", "").lower()
            # Check if this key matches a known agent ID
            is_known_agent = False
            for known_id in KNOWN_AGENT_MAPPINGS.keys():
                known_id_normalized = known_id.replace(" ", "").replace("_", "").lower()
                if (
                    key_normalized == known_id_normalized
                    or key.lower() == known_id.lower()
                ):
                    is_known_agent = True
                    break
            # Only save if it's not a known agent
            if not is_known_agent:
                cleaned_mapping[key] = value

        AGENT_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AGENT_MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned_mapping, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Saved agent mapping to {AGENT_MAPPING_FILE} (removed {len(mapping) - len(cleaned_mapping)} incorrect known agent mappings)"
        )
    except Exception as e:
        logger.warning(f"Failed to save agent mapping file: {e}")


def get_or_create_agent_mapping(agent_id_lower):
    """Get agent number for an agent ID, or create a new mapping deterministically."""
    mapping = load_agent_mapping()

    # Normalize by removing spaces to handle "bp agent 102256681" vs "bpagent102256681"
    agent_id_normalized = agent_id_lower.replace(" ", "").replace("_", "")

    # Check if already in normalized format (e.g., "agent 1", "agent 2")
    # This handles cases where cache already has normalized values
    if agent_id_lower.startswith("agent "):
        # Already normalized, return as-is (with proper capitalization)
        agent_num_str = agent_id_lower.replace("agent ", "").strip()
        try:
            agent_num = int(agent_num_str)
            return f"Agent {agent_num}"
        except ValueError:
            pass  # Not a valid number, continue with mapping logic

    # CRITICAL: Check KNOWN_AGENT_MAPPINGS FIRST (before checking file mappings)
    # This ensures known agents always get correct numbers, even if file has incorrect entries
    # Normalize known mapping keys for comparison
    for known_id, known_name in KNOWN_AGENT_MAPPINGS.items():
        known_id_normalized = known_id.replace(" ", "").replace("_", "")
        if (
            agent_id_normalized == known_id_normalized
            or agent_id_lower == known_id.lower()
        ):
            logger.debug(
                f"Matched known agent mapping: {agent_id_lower} -> {known_name}"
            )
            return known_name

    # Check special cases (normalize these too)
    if agent_id_normalized == "unknown" or agent_id_lower == "unknown":
        return "Agent 1"
    if agent_id_normalized in ["bp016803073", "bp016803074"] or agent_id_lower in [
        "bp016803073",
        "bp016803074",
    ]:
        return "Agent 1"
    if agent_id_normalized.startswith("bp01") or agent_id_lower.startswith("bp01"):
        return "Agent 1"

    # NOW check file mappings (only for unknown agents)
    # Double-check: if this is a known agent ID that got incorrectly mapped, use known mapping
    if agent_id_lower in mapping:
        mapped_value = mapping[agent_id_lower]
        # Verify this isn't a known agent that was incorrectly mapped
        for known_id, known_name in KNOWN_AGENT_MAPPINGS.items():
            known_id_normalized = known_id.replace(" ", "").replace("_", "")
            if (
                agent_id_normalized == known_id_normalized
                or agent_id_lower == known_id.lower()
            ):
                logger.warning(
                    f"Found incorrect mapping in file for known agent {agent_id_lower} -> {mapped_value}, using correct mapping {known_name}"
                )
                return known_name
        return mapped_value
    if agent_id_normalized in mapping:
        mapped_value = mapping[agent_id_normalized]
        # Verify this isn't a known agent that was incorrectly mapped
        for known_id, known_name in KNOWN_AGENT_MAPPINGS.items():
            known_id_normalized = known_id.replace(" ", "").replace("_", "")
            if (
                agent_id_normalized == known_id_normalized
                or agent_id_lower == known_id.lower()
            ):
                logger.warning(
                    f"Found incorrect mapping in file for known agent {agent_id_normalized} -> {mapped_value}, using correct mapping {known_name}"
                )
                return known_name
        return mapped_value

    # For new agent IDs, assign deterministically based on hash
    # This ensures the same agent ID always gets the same number
    # Use normalized version for hash to ensure consistency
    import hashlib

    hash_value = int(hashlib.md5(agent_id_normalized.encode()).hexdigest(), 16)
    # Use modulo to get a number between 1 and 99
    # Note: Agent 5 and Agent 11 are now reserved for known mappings, skip them in hash assignment
    agent_number = (hash_value % 99) + 1
    # Skip 5 and 11 in hash assignment (they're reserved for known mappings)
    if agent_number == 5:
        agent_number = 12  # Skip 5, use 12 instead
    if agent_number == 11:
        agent_number = 12  # Skip 11, use 12 instead (if it was already 12, it stays 12)

    agent_name = f"Agent {agent_number}"

    # Save the new mapping (save both versions to prevent future lookups)
    mapping[agent_id_lower] = agent_name
    mapping[agent_id_normalized] = agent_name
    save_agent_mapping(mapping)

    logger.info(f"Created new agent mapping: {agent_id_lower} -> {agent_name}")
    return agent_name


def normalize_agent_id(agent_str):
    """Normalize agent ID by extracting first two digits after 'bpagent'.

    Format: 'bpagent###########' → extract first two digits → 'Agent ##'
    Exceptions: Special cases for Jesus (unknown, bp016803073, bp016803074, bp01*)
    This same normalized value is used everywhere (display and filtering).
    """
    if pd.isna(agent_str) or not agent_str:
        return agent_str

    agent_str_lower = str(agent_str).lower().strip()

    # Check if already in "Agent X" or "BPO Agent X" format - return as-is
    if agent_str_lower.startswith("agent ") or agent_str_lower.startswith("bpo agent "):
        # Extract number and return in consistent format
        agent_str_clean = (
            agent_str_lower.replace("bpo agent ", "").replace("agent ", "").strip()
        )
        try:
            agent_num = int(agent_str_clean)
            return f"Agent {agent_num}"
        except ValueError:
            pass

    # Special cases for Jesus (Agent 1)
    agent_id_normalized = agent_str_lower.replace(" ", "").replace("_", "")
    if agent_id_normalized == "unknown" or agent_str_lower == "unknown":
        return "Agent 1"
    if agent_id_normalized in ["bp016803073", "bp016803074"]:
        return "Agent 1"
    if agent_id_normalized.startswith("bp01"):
        return "Agent 1"

    # Extract first two digits after "bpagent"
    # Pattern: bpagent########### → extract first two digits (##)
    match = re.search(r"bpagent(\d{2})", agent_id_normalized)
    if match:
        agent_num = int(match.group(1))
        return f"Agent {agent_num}"

    # If no match, try the mapping system as fallback
    return get_or_create_agent_mapping(agent_str_lower)


def normalize_category(value):
    """
    Normalize call reason/outcome categories to merge duplicates and handle case sensitivity.

    Rules:
    1. Case-insensitive: "order status inquiry" and "Order status inquiry" → "Order status inquiry"
    2. Merge shipping-related: "customer informed about shipping delay", "shipping status",
       "shipping timeline" → "shipping status"
    3. Rename: "refund and return process initiated" → "return process initiated"

    Args:
        value: Category string to normalize

    Returns:
        Normalized category string
    """
    if pd.isna(value) or not value or not str(value).strip():
        return value

    value_str = str(value).strip()

    # Normalize to lowercase for comparison
    value_lower = value_str.lower()

    # Merge shipping-related categories
    shipping_variants = [
        "customer informed about shipping delay",
        "shipping status",
        "shipping timeline",
    ]
    if value_lower in shipping_variants:
        return "shipping status"

    # Rename "refund and return process initiated" to "return process initiated"
    if value_lower == "refund and return process initiated":
        return "return process initiated"

    # For case-insensitive duplicates, use the first occurrence's capitalization
    # We'll handle this by creating a mapping of lowercase -> preferred capitalization
    # For now, just return the value as-is (we'll handle case normalization separately)
    return value_str


def parse_csv_row(row, filename):
    """
    Parse a CSV row and convert it to the existing call data structure.

    Args:
        row: pandas Series representing a CSV row
        filename: CSV filename (for metadata)

    Returns:
        Dictionary with parsed call data matching expected call data format
    """
    data = {}

    # Basic field mappings
    # CRITICAL: Normalize empty strings to None to avoid issues in extract_cache_key
    call_id_raw = row.get("call_id", "")
    if pd.isna(call_id_raw) or call_id_raw == "" or str(call_id_raw).strip() == "":
        data["call_id"] = None
    else:
        data["call_id"] = str(call_id_raw).strip()
    data["qa_score"] = (
        float(row.get("qa_score", 0)) if pd.notna(row.get("qa_score")) else None
    )
    data["label"] = str(row.get("label", "")) if pd.notna(row.get("label")) else None
    data["strengths"] = (
        str(row.get("strengths", "")) if pd.notna(row.get("strengths")) else None
    )
    data["challenges"] = (
        str(row.get("challenges", "")) if pd.notna(row.get("challenges")) else None
    )
    # Normalize reason and outcome during parsing so they're saved normalized to cache
    reason_raw = (
        str(row.get("call_reason", "")) if pd.notna(row.get("call_reason")) else None
    )
    data["reason"] = normalize_category(reason_raw) if reason_raw else None

    outcome_raw = (
        str(row.get("call_outcome", "")) if pd.notna(row.get("call_outcome")) else None
    )
    data["outcome"] = normalize_category(outcome_raw) if outcome_raw else None
    data["summary"] = (
        str(row.get("call_summary", "")) if pd.notna(row.get("call_summary")) else None
    )
    data["call_direction"] = (
        str(row.get("call_direction", ""))
        if pd.notna(row.get("call_direction"))
        else None
    )
    data["revenue_retention_amount"] = (
        float(row.get("revenue_retention_amount", 0))
        if pd.notna(row.get("revenue_retention_amount"))
        else 0.0
    )
    data["revenue_retention_summary"] = (
        str(row.get("revenue_retention_summary", ""))
        if pd.notna(row.get("revenue_retention_summary"))
        else None
    )

    # Handle handle_time_minutes conversion
    handle_time = row.get("handle_time_minutes")
    if pd.notna(handle_time):
        try:
            minutes = float(handle_time)
            data["speaking_time_per_speaker"] = {
                "total": f"{int(minutes)}:{int((minutes % 1) * 60):02d}"
            }
        except (ValueError, TypeError):
            data["speaking_time_per_speaker"] = None
    else:
        data["speaking_time_per_speaker"] = None

    # Parse call_date (YYYYMMDD format)
    call_date_str = row.get("call_date")
    if pd.notna(call_date_str):
        try:
            call_date_str = str(call_date_str).strip()
            if len(call_date_str) == 8:  # YYYYMMDD
                call_date = datetime.strptime(call_date_str, "%Y%m%d")
                data["call_date"] = call_date
                # Create date_raw in MMDDYYYY format for backward compatibility
                data["date_raw"] = (
                    f"{call_date_str[4:6]}{call_date_str[6:8]}{call_date_str[0:4]}"
                )
            else:
                # Try parsing as datetime string
                data["call_date"] = pd.to_datetime(call_date_str, errors="coerce")
                if pd.notna(data["call_date"]):
                    data["date_raw"] = data["call_date"].strftime("%m%d%Y")
                else:
                    data["call_date"] = None
                    data["date_raw"] = None
        except (ValueError, TypeError):
            data["call_date"] = None
            data["date_raw"] = None
    else:
        data["call_date"] = None
        data["date_raw"] = None

    # Extract time from call_id if available (format: YYYYMMDD_HHMMSS_...)
    if data.get("call_id"):
        try:
            call_id_parts = str(data["call_id"]).split("_")
            if len(call_id_parts) >= 2:
                time_str = call_id_parts[1]
                if len(time_str) == 6:  # HHMMSS
                    data["time"] = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
                else:
                    data["time"] = None
            else:
                data["time"] = None
        except Exception:
            data["time"] = None
    else:
        data["time"] = None

    # Normalize agent name
    # Try multiple possible column names (agent_name, agent, Agent, agent_id)
    agent_name = None
    for col_name in ["agent_name", "agent", "Agent", "agent_id"]:
        if col_name in row.index:
            agent_value = row.get(col_name)
            if pd.notna(agent_value) and str(agent_value).strip():
                agent_name = str(agent_value).strip()
                break

    if agent_name:
        data["agent"] = normalize_agent_id(agent_name)
    else:
        data["agent"] = normalize_agent_id("Unknown")

    # Build rubric_details dict
    rubric_details = {}
    rubric_code_pattern = re.compile(r"^\d+\.\d+\.\d+$")

    for col in row.index:
        if rubric_code_pattern.match(col):
            code = col
            status_value = row.get(code)
            reason_col = f"{code}__reason"
            reason_value = row.get(reason_col) if reason_col in row.index else None

            # Convert status: True -> "Pass", False -> "Fail", N/A or NaN -> "N/A"
            if (
                pd.isna(status_value)
                or status_value == "N/A"
                or str(status_value).upper() == "N/A"
            ):
                status = "N/A"
            elif status_value is True or str(status_value).lower() == "true":
                status = "Pass"
            elif status_value is False or str(status_value).lower() == "false":
                status = "Fail"
            else:
                # Try to parse as boolean
                status_str = str(status_value).lower()
                if status_str in ["true", "pass"]:
                    status = "Pass"
                elif status_str in ["false", "fail"]:
                    status = "Fail"
                else:
                    status = "N/A"

            # Get note (reason)
            note = None
            if pd.notna(reason_value) and str(reason_value).strip():
                note = str(reason_value).strip()

            rubric_details[code] = {"status": status, "note": note}

    data["rubric_details"] = rubric_details

    # Calculate rubric statistics
    total_rubric_items = len(rubric_details)
    pass_count = sum(1 for r in rubric_details.values() if r["status"] == "Pass")
    fail_count = sum(1 for r in rubric_details.values() if r["status"] == "Fail")
    na_count = sum(1 for r in rubric_details.values() if r["status"] == "N/A")

    data["rubric_pass_count"] = pass_count
    data["rubric_fail_count"] = fail_count
    data["rubric_na_count"] = na_count
    data["rubric_total_count"] = total_rubric_items

    # Build coaching_suggestions list
    coaching_suggestions = []
    for i in range(1, 4):
        coaching_col = f"coaching_{i}"
        if coaching_col in row.index:
            coaching_value = row.get(coaching_col)
            if pd.notna(coaching_value) and str(coaching_value).strip():
                coaching_suggestions.append(str(coaching_value).strip())
    data["coaching_suggestions"] = coaching_suggestions

    # Set metadata fields
    # Use call_id as the unique identifier for each row, not the filename
    # This ensures each row in a CSV is treated as a separate call
    call_id = data.get("call_id", "")
    normalized_filename = filename.strip("/")
    if call_id and str(call_id).strip():
        # Create unique _id from call_id and filename to ensure uniqueness
        # _id should be "filename:call_id" for CSV rows
        data["_id"] = f"{normalized_filename}:{call_id}"
        # _s3_key should be just the filename (for CSV files, this is the file path)
        # This allows extract_cache_key to detect CSV format by checking if _id contains colon
        data["_s3_key"] = normalized_filename
    else:
        # CRITICAL FIX: If no call_id, use row index to ensure uniqueness
        # Get row index from the row if available, otherwise use a timestamp
        row_index = getattr(row, "name", None)
        if row_index is None:
            import time

            row_index = (
                int(time.time() * 1000000) % 1000000
            )  # Use microsecond timestamp as fallback
        # Fallback: use filename + row index to ensure uniqueness
        data["_id"] = f"{normalized_filename}:row_{row_index}"
        data["_s3_key"] = normalized_filename
    data["filename"] = filename
    data["company"] = "Jomashop"

    # Backward compatibility fields
    if data.get("qa_score") is not None:
        data["average_happiness_value"] = data["qa_score"]
    else:
        data["average_happiness_value"] = 0.0

    # Legacy emotion fields
    data["happy"] = 0
    data["angry"] = 0
    data["sad"] = 0
    data["neutral"] = 1 if data.get("label", "").lower() == "neutral" else 0

    # Low confidences
    data["low_confidences"] = 0.0

    return data


def load_calls_from_csv(s3_client, s3_bucket, s3_prefix):
    """
    Load call data from CSV files in S3.

    Args:
        s3_client: Boto3 S3 client
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix/folder path

    Returns:
        Tuple: (list of call dictionaries, list of error messages)
    """
    all_calls = []
    errors = []

    try:
        # List all CSV files in S3 bucket
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix, MaxKeys=1000)

        csv_keys = []
        for page in pages:
            if isinstance(page, dict) and "Contents" in page:
                for obj in page["Contents"]:
                    key = obj.get("Key")
                    if key and key.lower().endswith(".csv"):
                        csv_keys.append(key)

        if not csv_keys:
            logger.info("No CSV files found in S3 bucket")
            return [], ["No CSV files found in S3 bucket"]

        logger.info(f"Found {len(csv_keys)} CSV file(s) in S3 bucket")

        # Process each CSV file
        for csv_key in csv_keys:
            try:
                logger.debug(f"Processing CSV file: {csv_key}")

                # Download CSV from S3
                response = s3_client.get_object(Bucket=s3_bucket, Key=csv_key)
                csv_content = response["Body"].read().decode("utf-8")

                # Read CSV into DataFrame
                csv_file = io.StringIO(csv_content)
                df = pd.read_csv(csv_file)

                # Extract filename from key
                filename = csv_key.split("/")[-1]

                # Process each row
                for idx, row in df.iterrows():
                    try:
                        parsed_data = parse_csv_row(row, filename)
                        if parsed_data:
                            all_calls.append(parsed_data)
                    except Exception as e:
                        error_msg = (
                            f"Error parsing row {idx + 1} in {filename}: {str(e)}"
                        )
                        errors.append(error_msg)
                        logger.warning(error_msg)
                        continue

                logger.info(f"Processed {len(df)} rows from {filename}")

            except Exception as e:
                error_msg = f"Error processing CSV file {csv_key}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                continue

        logger.info(
            f"Successfully loaded {len(all_calls)} calls from {len(csv_keys)} CSV file(s)"
        )

    except Exception as e:
        error_msg = f"Error listing CSV files in S3: {str(e)}"
        errors.append(error_msg)
        logger.error(error_msg)

    return all_calls, errors


def load_all_calls_internal(max_files=None):
    """
    Internal function to load CSV files from S3 bucket (PDFs are ignored).
    Returns tuple: (call_data_list, error_message)

    Args:
        max_files: Maximum number of CSV files to load (None = load all)
    """
    try:
        all_calls = []

        # Configure S3 client with timeout
        import botocore.config

        config = botocore.config.Config(
            connect_timeout=10, read_timeout=30, retries={"max_attempts": 2}
        )
        s3_client_with_timeout = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
            region_name=st.secrets["s3"].get("region_name", "us-east-1"),
            config=config,
        )

        # Load calls from CSV files (PDFs are ignored)
        logger.info("Loading from CSV files (PDFs ignored)...")
        csv_calls, csv_errors = load_calls_from_csv(
            s3_client_with_timeout, s3_bucket_name, s3_prefix
        )

        if csv_errors:
            logger.warning(
                f"Encountered {len(csv_errors)} error(s) while loading CSV files"
            )
            for error in csv_errors[:10]:  # Log first 10 errors
                logger.warning(f"  {error}")

        if not csv_calls:
            return [], "No CSV files found in S3 bucket or no valid data loaded"

        all_calls = csv_calls

        # Sort calls by call_date (most recent first) if available
        try:
            all_calls.sort(
                key=lambda x: x.get("call_date", datetime.min)
                if isinstance(x.get("call_date"), datetime)
                else datetime.min,
                reverse=True,
            )
        except Exception:
            # If sorting fails, keep original order
            pass

        # Track processed S3 keys in session state (for smart refresh)
        if "processed_s3_keys" not in st.session_state:
            st.session_state["processed_s3_keys"] = set()
        processed_keys = {
            call.get("_s3_key") for call in all_calls if call.get("_s3_key")
        }
        st.session_state["processed_s3_keys"].update(processed_keys)

        # Sort calls by call_date (most recent first) if available
        try:
            all_calls.sort(
                key=lambda x: x.get("call_date", datetime.min)
                if isinstance(x.get("call_date"), datetime)
                else datetime.min,
                reverse=True,
            )
        except Exception:
            # If sorting fails, keep original order
            pass

        # Track processed S3 keys in session state (for smart refresh)
        if "processed_s3_keys" not in st.session_state:
            st.session_state["processed_s3_keys"] = set()
        processed_keys = {
            call.get("_s3_key") for call in all_calls if call.get("_s3_key")
        }
        st.session_state["processed_s3_keys"].update(processed_keys)

        # Return with error message if any
        error_message = None if not csv_errors else "; ".join(csv_errors[:5])
        return all_calls, error_message

    except NoCredentialsError as e:
        error_msg = (
            "AWS credentials not found. Please configure S3 credentials in secrets."
        )
        track_error("S3_NoCredentials", str(e))
        return [], error_msg
    except ClientError as e:
        error_code = "Unknown"
        if hasattr(e, "response") and isinstance(e.response, dict):
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchBucket":
            error_msg = f"S3 bucket '{s3_bucket_name}' not found."
            track_error(f"S3_{error_code}", error_msg)
            return [], error_msg
        elif error_code == "AccessDenied":
            error_msg = f"Access denied to S3 bucket '{s3_bucket_name}'. Check your credentials."
            track_error(f"S3_{error_code}", error_msg)
            return [], error_msg
        else:
            error_msg = f"S3 error: {e}"
            track_error(f"S3_{error_code}", error_msg)
            return [], error_msg
    except Exception as e:
        error_msg = f"Unexpected error loading from S3: {e}"
        track_error("S3_Unexpected", error_msg)
        logger.exception("Unexpected error in load_all_calls_internal")
        return [], error_msg


# Persistent cache file that survives app restarts (prevents timeout issues)
CACHE_FILE = log_dir / "cached_calls_data.json"
LOCK_FILE = log_dir / "cached_calls_data.json.lock"

# S3 cache key for storing cache in S3 (survives deployments and app restarts)
S3_CACHE_KEY = "cache/cached_calls_data.json"


def get_s3_client_and_bucket():
    """Get S3 client and bucket name. Returns (client, bucket_name) or (None, None) if unavailable."""
    try:
        if "s3" not in st.secrets:
            return None, None
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
            region_name=st.secrets["s3"].get("region_name", "us-east-1"),
        )
        s3_bucket_name = st.secrets["s3"]["bucket_name"]
        return s3_client, s3_bucket_name
    except Exception as e:
        logger.debug(f"Could not get S3 client: {e}")
        return None, None


def recover_partial_json(filepath):
    """Attempt to recover partial data from corrupted JSON file.

    Tries to extract valid call_data entries even if metadata is corrupted.

    Args:
        filepath: Path to the corrupted JSON file

    Returns:
        tuple: (recovered_call_data, recovered_errors) or (None, None) if recovery fails
    """
    try:
        # Check file size before reading (optional optimization)
        file_size = filepath.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            logger.warning(
                f" Corrupted cache file is very large ({file_size / 1024 / 1024:.1f}MB), recovery may be slow"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Try to find call_data array in the content
        # Look for patterns like "call_data": [ ... ]

        # Try to extract call_data array
        call_data_match = re.search(r'"call_data"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        if call_data_match:
            # Try to parse just the array portion
            array_content = "[" + call_data_match.group(1) + "]"
            try:
                # Try to parse as JSON array
                recovered_calls = json.loads(array_content)
                if isinstance(recovered_calls, list):
                    # Validate that list contains dicts (call objects)
                    if len(recovered_calls) > 0 and not isinstance(
                        recovered_calls[0], dict
                    ):
                        logger.warning(" Recovered data is not in expected format")
                        return None, None
                    logger.info(
                        f" Recovered {len(recovered_calls)} calls from corrupted cache (recovery complete)"
                    )
                    return recovered_calls, []
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract individual call objects with balanced braces
        # Look for patterns like {"_s3_key": "...", ...} and handle nested structures
        recovered_calls = []
        # Find all positions where {"_s3_key": appears
        pattern = r'\{"_s3_key"\s*:\s*"[^"]+"'
        for match in re.finditer(pattern, content):
            start_pos = match.start()
            # Find the matching closing brace by tracking brace depth
            brace_depth = 0
            in_string = False
            escape_next = False
            end_pos = start_pos

            for i in range(start_pos, len(content)):
                char = content[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == "{":
                        brace_depth += 1
                    elif char == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            end_pos = i + 1
                            break

            if brace_depth == 0 and end_pos > start_pos:
                # Extract the complete object
                try:
                    obj_str = content[start_pos:end_pos]
                    call_obj = json.loads(obj_str)
                    if isinstance(call_obj, dict) and call_obj.get("_s3_key"):
                        recovered_calls.append(call_obj)
                except json.JSONDecodeError:
                    continue

        if recovered_calls:
            logger.info(
                f" Recovered {len(recovered_calls)} calls from corrupted cache (partial recovery complete)"
            )
            return recovered_calls, []

        return None, None

    except Exception as e:
        logger.warning(f" Failed to recover partial data: {e}")
        return None, None


def extract_cache_key(call):
    """
    Extract a consistent cache key from a call object.

    For CSV rows: Returns call_id (detected by colon in _id or _s3_key format)
    For PDF rows: Returns _s3_key or _id
    Handles both old format (_id = filename) and new format (_id = "filename:call_id")

    Args:
        call: Dictionary representing a call

    Returns:
        String key for cache lookup, or None if no valid key found
    """
    if not isinstance(call, dict):
        return None

    call_id = call.get("call_id")
    _id = call.get("_id", "")
    _s3_key = call.get("_s3_key", "")

    # If _id contains a colon, it's from a CSV (format: filename:call_id or filename:row_X)
    # Use the part after the colon as the unique key for CSV rows
    if ":" in str(_id):
        # Extract the unique part after the colon (call_id or row_X)
        parts = str(_id).split(":", 1)
        if len(parts) == 2:
            return parts[1]  # Return call_id or row_X
        # Fallback: return _id if split fails
        return str(_id)
    elif ":" in str(_s3_key):
        # Same logic for _s3_key (shouldn't happen with new format, but handle legacy)
        parts = str(_s3_key).split(":", 1)
        if len(parts) == 2:
            return parts[1]
        return str(_s3_key)
    else:
        # For PDFs or legacy format, use _s3_key or _id
        return _s3_key or _id or (str(call_id) if call_id else None)


def migrate_old_cache_format(call_data):
    """
    Migrate old cache format calls to new format and normalize agent IDs.

    Old format: _id = filename (causes all rows from same CSV to be treated as duplicates)
    New format: _id = "filename:call_id" (each row is unique)

    Args:
        call_data: List of call dictionaries

    Returns:
        List of migrated call dictionaries with normalized agent IDs
    """
    if not call_data:
        return call_data

    migrated_count = 0
    agent_normalized_count = 0
    migrated_calls = []

    for call in call_data:
        if not isinstance(call, dict):
            migrated_calls.append(call)
            continue

        _id = call.get("_id", "")
        _s3_key = call.get("_s3_key", "")
        call_id = call.get("call_id")
        filename = call.get("filename", "")

        # Check if this is an old format call (CSV row with _id = filename, not "filename:call_id")
        # Old format indicators:
        # 1. _id doesn't contain a colon but filename ends with .csv
        # 2. _id equals filename (or filename without path)
        # 3. call_id exists but _id doesn't contain call_id

        is_old_format = False
        if call_id and filename:
            filename_base = filename.split("/")[-1]  # Get just filename, not path
            normalized_filename = filename.strip("/")
            # Check if _id is just the filename (old format)
            # Compare against both full filename and base filename
            if _id == filename or _id == filename_base or _id == normalized_filename:
                if filename.lower().endswith(".csv") and ":" not in str(_id):
                    is_old_format = True

        if is_old_format and call_id:
            # Migrate to new format: _id = "filename:call_id", _s3_key = filename
            call["_id"] = f"{filename}:{call_id}"
            if not _s3_key or _s3_key == filename or _s3_key == filename_base:
                # _s3_key should be just the filename (not filename:call_id)
                call["_s3_key"] = filename
            migrated_count += 1

        # CRITICAL: Normalize agent ID in cached data to ensure consistency
        # This fixes the issue where cached data has old agent IDs like "bpagent030844482"
        if "agent" in call:
            original_agent = call.get("agent")
            normalized_agent = normalize_agent_id(original_agent)
            if original_agent != normalized_agent:
                call["agent"] = normalized_agent
                agent_normalized_count += 1

        migrated_calls.append(call)

    if migrated_count > 0:
        logger.info(
            f" Migrated {migrated_count} calls from old cache format to new format"
        )
    if agent_normalized_count > 0:
        logger.info(f" Normalized {agent_normalized_count} agent IDs in cached data")

    return migrated_calls


def deduplicate_calls(call_data):
    """Remove duplicate calls based on _s3_key or _id. Keeps the first occurrence.

    For CSV files, uses call_id as the primary identifier since multiple rows
    can come from the same CSV file.
    """
    if not call_data:
        return []

    seen_keys = set()
    deduplicated = []
    duplicates_count = 0

    for call in call_data:
        # Skip non-dict items (defensive programming)
        if not isinstance(call, dict):
            logger.warning(
                f" Skipping non-dict item in call_data: {type(call).__name__}"
            )
            continue

        # Use consistent key extraction
        key = extract_cache_key(call)
        if key and key not in seen_keys:
            seen_keys.add(key)
            deduplicated.append(call)
        elif key:
            duplicates_count += 1

    if duplicates_count > 0:
        logger.warning(
            f" Removed {duplicates_count} duplicate calls (kept {len(deduplicated)} unique calls)"
        )
    else:
        logger.debug(
            f" Deduplication check: No duplicates found, all {len(deduplicated)} calls are unique"
        )

    return deduplicated


def cleanup_pdf_sourced_calls():
    """
    One-time cache cleanup to remove PDF-sourced calls from cache.
    This function checks if cleanup has already been performed and only runs once.
    """
    try:
        # Check if cleanup has already been performed
        cleanup_flag_key = "cache_cleaned_pdf_calls"

        # Check both disk and S3 cache for cleanup flag
        cleanup_performed = False

        # Check disk cache first
        if CACHE_FILE.exists():
            try:
                with cache_file_lock(CACHE_FILE, timeout=5):
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    if cached_data.get(cleanup_flag_key, False):
                        cleanup_performed = True
                        logger.info(
                            "Cache cleanup already performed (found flag in disk cache)"
                        )
            except (
                LockTimeoutError,
                json.JSONDecodeError,
                FileNotFoundError,
                Exception,
            ):
                # If we can't read the cache, assume cleanup not performed
                pass

        # Check S3 cache if disk check didn't find flag
        if not cleanup_performed:
            s3_client, s3_bucket = get_s3_client_and_bucket()
            if s3_client and s3_bucket:
                try:
                    response = s3_client.get_object(Bucket=s3_bucket, Key=S3_CACHE_KEY)
                    body = response["Body"]
                    chunks = []
                    for chunk in body.iter_chunks(chunk_size=8192):
                        chunks.append(chunk)
                    cached_data = json.loads(b"".join(chunks).decode("utf-8"))
                    if cached_data.get(cleanup_flag_key, False):
                        cleanup_performed = True
                        logger.info(
                            "Cache cleanup already performed (found flag in S3 cache)"
                        )
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code == "NoSuchKey":
                        # No cache in S3, proceed with cleanup
                        pass
                    else:
                        logger.warning(
                            f"Could not check S3 cache for cleanup flag: {e}"
                        )
                except Exception:
                    # If we can't read S3 cache, proceed with cleanup
                    pass

        if cleanup_performed:
            return  # Cleanup already done, skip

        logger.info("Starting one-time cache cleanup to remove PDF-sourced calls...")

        # Load existing cache from disk and S3
        call_data = []
        errors = []
        cache_timestamp = None

        # Try disk cache first
        if CACHE_FILE.exists():
            try:
                with cache_file_lock(CACHE_FILE, timeout=5):
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    call_data = cached_data.get("call_data", [])
                    errors = cached_data.get("errors", [])
                    cache_timestamp = cached_data.get("timestamp")
            except Exception as e:
                logger.warning(f"Could not load disk cache for cleanup: {e}")

        # If no disk cache or empty, try S3
        if not call_data:
            s3_client, s3_bucket = get_s3_client_and_bucket()
            if s3_client and s3_bucket:
                try:
                    response = s3_client.get_object(Bucket=s3_bucket, Key=S3_CACHE_KEY)
                    body = response["Body"]
                    chunks = []
                    for chunk in body.iter_chunks(chunk_size=8192):
                        chunks.append(chunk)
                    cached_data = json.loads(b"".join(chunks).decode("utf-8"))
                    call_data = cached_data.get("call_data", [])
                    errors = cached_data.get("errors", [])
                    cache_timestamp = cached_data.get("timestamp")
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code != "NoSuchKey":
                        logger.warning(f"Could not load S3 cache for cleanup: {e}")
                except Exception as e:
                    logger.warning(f"Error loading S3 cache for cleanup: {e}")

        if not call_data:
            logger.info("No cache found to clean - marking cleanup as complete")
            # Mark cleanup as complete even if no cache exists
            cleanup_metadata = {
                "call_data": [],
                "errors": [],
                "timestamp": datetime.now().isoformat(),
                cleanup_flag_key: True,
            }
            try:
                with cache_file_lock(CACHE_FILE, timeout=5):
                    atomic_write_json(CACHE_FILE, cleanup_metadata)
                # Also save to S3
                s3_client, s3_bucket = get_s3_client_and_bucket()
                if s3_client and s3_bucket:
                    try:
                        json_str = json.dumps(cleanup_metadata, default=str)
                        s3_client.put_object(
                            Bucket=s3_bucket,
                            Key=S3_CACHE_KEY,
                            Body=json_str.encode("utf-8"),
                            ContentType="application/json",
                        )
                    except Exception:
                        pass  # S3 save is optional
            except Exception:
                pass
            return

        # Identify PDF-sourced calls
        original_count = len(call_data)
        pdf_sourced_count = 0

        cleaned_calls = []
        for call in call_data:
            s3_key = call.get("_s3_key", "")
            filename = call.get("filename", "")

            # Check if call is PDF-sourced
            is_pdf_sourced = False
            if s3_key and str(s3_key).lower().endswith(".pdf"):
                is_pdf_sourced = True
            elif filename and str(filename).lower().endswith(".pdf"):
                is_pdf_sourced = True

            if is_pdf_sourced:
                pdf_sourced_count += 1
            else:
                cleaned_calls.append(call)

        # Migrate old cache format calls to new format
        if cleaned_calls:
            cleaned_calls = migrate_old_cache_format(cleaned_calls)

        if pdf_sourced_count == 0:
            logger.info(
                "No PDF-sourced calls found in cache - marking cleanup as complete"
            )
            # Mark cleanup as complete
            if CACHE_FILE.exists():
                try:
                    with cache_file_lock(CACHE_FILE, timeout=5):
                        with open(CACHE_FILE, "r", encoding="utf-8") as f:
                            cached_data = json.load(f)
                        cached_data[cleanup_flag_key] = True
                        atomic_write_json(CACHE_FILE, cached_data)
                        # Also save to S3
                        s3_client, s3_bucket = get_s3_client_and_bucket()
                        if s3_client and s3_bucket:
                            try:
                                json_str = json.dumps(cached_data, default=str)
                                s3_client.put_object(
                                    Bucket=s3_bucket,
                                    Key=S3_CACHE_KEY,
                                    Body=json_str.encode("utf-8"),
                                    ContentType="application/json",
                                )
                            except Exception:
                                pass
                except Exception:
                    pass
            return

        logger.info(
            f"Removed {pdf_sourced_count} PDF-sourced calls from cache "
            f"({original_count} → {len(cleaned_calls)} calls, one-time cleanup)"
        )

        # Update cache with cleaned data and cleanup flag
        cleaned_cache = {
            "call_data": cleaned_calls,
            "errors": errors,
            "timestamp": cache_timestamp or datetime.now().isoformat(),
            cleanup_flag_key: True,  # Mark cleanup as complete
        }

        # Preserve other metadata if it exists
        if CACHE_FILE.exists():
            try:
                with cache_file_lock(CACHE_FILE, timeout=5):
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        old_cache = json.load(f)
                    # Preserve other metadata fields
                    for key in ["partial", "processed", "total", "last_save_time"]:
                        if key in old_cache:
                            cleaned_cache[key] = old_cache[key]
            except Exception:
                pass

        # Save cleaned cache to disk
        try:
            with cache_file_lock(CACHE_FILE, timeout=10):
                atomic_write_json(CACHE_FILE, cleaned_cache)
            logger.info(f"Saved cleaned cache to disk: {len(cleaned_calls)} calls")
        except Exception as e:
            logger.error(f"Failed to save cleaned cache to disk: {e}")

        # Save cleaned cache to S3
        s3_client, s3_bucket = get_s3_client_and_bucket()
        if s3_client and s3_bucket:
            try:
                json_str = json.dumps(cleaned_cache, default=str)
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=S3_CACHE_KEY,
                    Body=json_str.encode("utf-8"),
                    ContentType="application/json",
                )
                logger.info(f"Saved cleaned cache to S3: {len(cleaned_calls)} calls")
            except Exception as e:
                logger.warning(f"Failed to save cleaned cache to S3: {e}")

    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
        logger.exception("Cache cleanup failed")


def load_cached_data_from_disk(max_retries=3, retry_delay=0.1):
    """Load cached data from S3 first, then fall back to local disk if S3 unavailable.

    Features:
    - Tries S3 first (survives deployments)
    - Falls back to local disk if S3 unavailable
    - File locking to prevent concurrent access
    - Retry logic for transient errors
    - Corruption recovery to extract partial data
    - Graceful degradation if locking unavailable
    - Session state memoization during refresh (5 second cache)

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries (seconds)

    Returns:
        tuple: (call_data, errors) or (None, None) if load fails
    """
    import time

    # Check session state cache first during refresh (optimization to reduce S3 calls)
    refresh_in_progress = st.session_state.get("refresh_in_progress", False)
    if refresh_in_progress:
        cache_key = "_disk_cache_during_refresh"
        if cache_key in st.session_state:
            cached_result, cached_timestamp = st.session_state[cache_key]
            # Use cached result if less than 5 seconds old (refresh updates frequently)
            if time.time() - cached_timestamp < 5:
                logger.debug(" Using session-cached disk cache result during refresh")
                return cached_result

    # CRITICAL: Check session state for S3 cache first (to avoid duplicate S3 loads)
    # If load_all_calls_cached() already loaded from S3, reuse that result
    s3_cache_key = "_s3_cache_result"
    if s3_cache_key in st.session_state:
        cached_result = st.session_state[s3_cache_key]
        logger.debug(
            " Using session-cached S3 result in load_cached_data_from_disk() - avoiding duplicate S3 load"
        )

        # Try to sync to local disk if file doesn't exist or is older (non-blocking, don't reload from S3)
        # Just check if disk cache exists and is recent, if not, we'll sync on next S3 load
        # This avoids another S3 load just for syncing

        # Cache result in session state during refresh
        result = cached_result
        if refresh_in_progress:
            cache_key = "_disk_cache_during_refresh"
            st.session_state[cache_key] = (result, time.time())

        return result

    # Try loading from S3 first (only if not already in session state)
    s3_client, s3_bucket = get_s3_client_and_bucket()
    if s3_client and s3_bucket:
        # Create dedicated S3 client with longer timeout for cache loading
        try:
            config = botocore.config.Config(
                connect_timeout=30,
                read_timeout=120,  # 2 minutes for large cache files
                retries={"max_attempts": 3, "mode": "adaptive"},
            )
            s3_cache_client = boto3.client(
                "s3",
                aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
                aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
                region_name=st.secrets["s3"].get("region_name", "us-east-1"),
                config=config,
            )
        except Exception as config_error:
            logger.warning(
                f" Could not create S3 cache client with extended timeout, using default: {config_error}"
            )
            s3_cache_client = s3_client

        # Retry logic for S3 cache download
        max_retries = 3
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f" Attempting to load cache from S3 (attempt {attempt + 1}/{max_retries})..."
                )
                response = s3_cache_client.get_object(
                    Bucket=s3_bucket, Key=S3_CACHE_KEY
                )

                # Stream download instead of loading all at once to prevent memory issues
                body = response["Body"]
                chunks = []
                chunk_size = 8192  # 8KB chunks
                total_bytes = 0

                try:
                    for chunk in body.iter_chunks(chunk_size=chunk_size):
                        chunks.append(chunk)
                        total_bytes += len(chunk)
                        # Log progress for large files (every 1MB)
                        if total_bytes % (1024 * 1024) < chunk_size:
                            logger.debug(
                                f" Loading cache from S3: {total_bytes / (1024 * 1024):.1f} MB downloaded..."
                            )
                except Exception as stream_error:
                    logger.warning(
                        f" Error streaming cache from S3: {stream_error}, will retry..."
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise

                # Parse JSON from streamed chunks
                try:
                    cached_data = json.loads(b"".join(chunks).decode("utf-8"))
                except json.JSONDecodeError as json_error:
                    logger.error(
                        f" Failed to parse JSON from S3 cache: {json_error}. "
                        f"Cache file may be corrupted. Will try local disk."
                    )
                    break  # Break retry loop, fall back to disk

                if not isinstance(cached_data, dict):
                    logger.warning(
                        f" S3 cache contains invalid data structure: {type(cached_data).__name__}, expected dict"
                    )
                    break  # Break retry loop, fall back to disk
                else:
                    call_data = cached_data.get("call_data", [])
                    errors = cached_data.get("errors", [])
                    cache_timestamp = cached_data.get("timestamp", None)
                    cache_count = len(call_data)
                    is_partial = cached_data.get("partial", False)

                    if is_partial:
                        processed = cached_data.get("processed", 0)
                        total = cached_data.get("total", 0)
                        logger.info(
                            f" Found PARTIAL cache in S3 with {cache_count} calls "
                            f"({processed}/{total} = {processed * 100 // total if total > 0 else 0}% complete, "
                            f"saved at {cache_timestamp})"
                        )
                    else:
                        logger.info(
                            f" Found COMPLETE cache in S3 with {cache_count} calls (saved at {cache_timestamp})"
                        )

                    # Also sync to local disk for faster subsequent loads
                    try:
                        with cache_file_lock(CACHE_FILE, timeout=5):
                            atomic_write_json(CACHE_FILE, cached_data)
                            logger.debug(
                                f" Synced S3 cache to local disk: {CACHE_FILE}"
                            )
                    except Exception as sync_error:
                        logger.debug(
                            f" Could not sync S3 cache to local disk: {sync_error}"
                        )

                    # Cache result in session state during refresh to reduce repeated S3 loads
                    result = (call_data, errors)
                    if refresh_in_progress:
                        cache_key = "_disk_cache_during_refresh"
                        st.session_state[cache_key] = (result, time.time())

                    return result

            except ClientError as s3_error:
                error_code = s3_error.response.get("Error", {}).get("Code", "")
                if error_code == "NoSuchKey":
                    logger.debug(
                        f" No cache found in S3: s3://{s3_bucket}/{S3_CACHE_KEY}"
                    )
                    break  # No retry needed for missing key
                else:
                    error_msg = str(s3_error)
                    if (
                        "timeout" in error_msg.lower()
                        or "timed out" in error_msg.lower()
                    ):
                        logger.warning(
                            f" S3 cache download timed out (attempt {attempt + 1}/{max_retries}). "
                            f"Will retry with exponential backoff..."
                        )
                    elif (
                        "connection reset" in error_msg.lower()
                        or "ConnectionResetError" in error_msg
                    ):
                        logger.warning(
                            f" S3 cache download connection reset (attempt {attempt + 1}/{max_retries}). "
                            f"Will retry with exponential backoff..."
                        )
                    else:
                        logger.warning(
                            f" Failed to load cache from S3 (attempt {attempt + 1}/{max_retries}): {s3_error}"
                        )

                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.info(f" Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f" Failed to load cache from S3 after {max_retries} attempts. "
                            f"Will try local disk cache."
                        )

            except Exception as s3_error:
                error_msg = str(s3_error)
                error_type = type(s3_error).__name__

                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    logger.warning(
                        f" S3 cache download timeout (attempt {attempt + 1}/{max_retries}): {error_type} - {error_msg}"
                    )
                elif (
                    "connection reset" in error_msg.lower()
                    or "ConnectionResetError" in error_msg
                ):
                    logger.warning(
                        f" S3 cache download connection reset (attempt {attempt + 1}/{max_retries}): {error_type} - {error_msg}"
                    )
                elif "MemoryError" in error_type or "memory" in error_msg.lower():
                    logger.error(
                        f" Memory error loading S3 cache: {error_type} - {error_msg}. "
                        f"Cache file may be too large. Will try local disk."
                    )
                    break  # Don't retry memory errors
                else:
                    logger.warning(
                        f" Failed to load cache from S3 (attempt {attempt + 1}/{max_retries}): {error_type} - {error_msg}"
                    )

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f" Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f" Failed to load cache from S3 after {max_retries} attempts: {error_type} - {error_msg}. "
                        f"Will try local disk cache."
                    )

    # Fall back to local disk if S3 unavailable or not found
    if not CACHE_FILE.exists():
        return None, None

    # Retry logic for transient errors
    for attempt in range(max_retries):
        try:
            # Use file locking to prevent concurrent reads during writes
            # BUG FIX: Use timeout for read locks to prevent deadlock with concurrent threads
            # ThreadPoolExecutor uses 10 workers, and if a writer crashes/hangs, readers should timeout and retry
            # rather than wait forever. Use 5-second timeout (same order of magnitude as write timeout of 10 seconds)
            try:
                with cache_file_lock(CACHE_FILE, timeout=5):
                    logger.debug(f" Checking persistent cache file: {CACHE_FILE}")
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                        if not isinstance(cached_data, dict):
                            logger.error(
                                f" Cache file contains invalid data structure: "
                                f"{type(cached_data).__name__}, expected dict"
                            )
                            return None, None
                        call_data = cached_data.get("call_data", [])
                        errors = cached_data.get("errors", [])
                        cache_timestamp = cached_data.get("timestamp", None)
                        cache_count = len(call_data)

                        # OPTIMIZATION: Cache should already be deduplicated (done in save_cached_data_to_disk)
                        # No need to deduplicate here - reduces redundant processing
                        is_partial = cached_data.get("partial", False)

                        if is_partial:
                            processed = cached_data.get("processed", 0)
                            total = cached_data.get("total", 0)
                            logger.info(
                                f" Found PARTIAL cache with {cache_count} calls "
                                f"({processed}/{total} = {processed * 100 // total if total > 0 else 0}% complete, "
                                f"saved at {cache_timestamp})"
                            )
                            logger.debug(
                                " This partial cache will be used if it's more complete than a fresh load"
                            )
                        else:
                            logger.info(
                                f" Found COMPLETE persistent cache with {cache_count} calls (saved at {cache_timestamp})"
                            )

                        # Cache result in session state during refresh to reduce repeated disk loads
                        result = (call_data, errors)
                        if refresh_in_progress:
                            cache_key = "_disk_cache_during_refresh"
                            st.session_state[cache_key] = (result, time.time())

                        return result
            except LockTimeoutError as e:
                # Lock timeout - retry with exponential backoff
                if attempt < max_retries - 1:
                    logger.warning(
                        f" Lock timeout on cache read attempt {attempt + 1}: {e}, retrying..."
                    )
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(
                        f" Failed to acquire lock for cache read after {max_retries} attempts: {e}"
                    )
                    return None, None

        except json.JSONDecodeError as e:
            logger.warning(
                f" Failed to load persistent cache: Corrupted JSON file - {e}"
            )

            # Try to recover partial data before deleting
            # Note: Recovery operations need lock protection to prevent concurrent access
            # Read corrupted file with lock protection
            recovered_calls, recovered_errors = None, None
            try:
                with cache_file_lock(CACHE_FILE, timeout=10):
                    recovered_calls, recovered_errors = recover_partial_json(CACHE_FILE)
            except LockTimeoutError as e:
                logger.error(
                    f" Lock timeout while reading corrupted cache for recovery: {e}"
                )
            except Exception as recovery_read_error:
                logger.warning(
                    f" Failed to read corrupted cache for recovery: {recovery_read_error}"
                )

            if recovered_calls:
                # Save recovered data to a new cache file (with lock protection)
                try:
                    try:
                        with cache_file_lock(CACHE_FILE, timeout=10):
                            backup_path = CACHE_FILE.with_suffix(".json.corrupted")
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            backup_path = (
                                CACHE_FILE.parent
                                / f"cached_calls_data.json.corrupted.{timestamp}"
                            )
                            shutil.copy2(CACHE_FILE, backup_path)
                            logger.info(f" Backed up corrupted cache to {backup_path}")

                            # Save recovered data
                            recovered_data = {
                                "call_data": recovered_calls,
                                "errors": recovered_errors,
                                "timestamp": datetime.now().isoformat(),
                                "count": len(recovered_calls),
                                "partial": True,  # Mark as partial since we lost metadata
                                "recovered_from_corruption": True,
                            }
                            atomic_write_json(CACHE_FILE, recovered_data)
                            logger.info(
                                f" Recovered and saved {len(recovered_calls)} calls from corrupted cache "
                                f"(recovery and save complete)"
                            )
                            return recovered_calls, recovered_errors
                    except LockTimeoutError as e:
                        logger.error(f" Lock timeout while saving recovered data: {e}")
                        raise  # Re-raise to trigger outer exception handler

                except Exception as recovery_error:
                    logger.error(f" Failed to save recovered data: {recovery_error}")

            # If recovery failed, backup and delete corrupted cache (with lock protection)
            try:
                try:
                    with cache_file_lock(CACHE_FILE, timeout=10):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = (
                            CACHE_FILE.parent
                            / f"cached_calls_data.json.corrupted.{timestamp}"
                        )
                        shutil.copy2(CACHE_FILE, backup_path)
                        CACHE_FILE.unlink()
                        logger.info(
                            f" Backed up corrupted cache to {backup_path} and deleted original"
                        )
                except LockTimeoutError as e:
                    logger.error(f" Lock timeout while backing up corrupted cache: {e}")
            except Exception as backup_error:
                logger.error(f" Failed to backup corrupted cache: {backup_error}")

            # Don't retry on JSON decode errors - file is corrupted
            return None, None

        except (IOError, OSError, PermissionError) as e:
            # Transient errors - retry
            if attempt < max_retries - 1:
                logger.warning(
                    f" Cache read attempt {attempt + 1} failed: {e}, retrying..."
                )
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(
                    f" Failed to load persistent cache after {max_retries} attempts: {e}"
                )
                return None, None

        except Exception as e:
            # Other errors - log and return None
            logger.warning(f" Failed to load persistent cache: {e}")

    # Cache result in session state during refresh to reduce repeated S3/disk loads
    result = None, None
    if refresh_in_progress:
        cache_key = "_disk_cache_during_refresh"
        st.session_state[cache_key] = (result, time.time())

    return result


def save_cached_data_to_disk(call_data, errors, partial=False, processed=0, total=0):
    """Save cached data to disk for persistence across app restarts. Automatically deduplicates before saving.

    Uses atomic writes and file locking to prevent corruption.

    Args:
        call_data: List of call data to save
        errors: List of errors encountered
        partial: Whether this is a partial save (in-progress)
        processed: Number of files processed (for partial saves)
        total: Total number of files to process (for partial saves)
    """
    try:
        # Validate inputs to prevent crashes and cache corruption
        if call_data is None:
            call_data = []
        elif not isinstance(call_data, list):
            raise TypeError(f"call_data must be a list, got {type(call_data).__name__}")

        if errors is None:
            errors = []
        elif not isinstance(errors, list):
            raise TypeError(f"errors must be a list, got {type(errors).__name__}")

        # Deduplicate and normalize agent IDs before saving to prevent cache bloat and ensure consistency
        if call_data:
            original_count = len(call_data)
            # Normalize agent IDs in all calls before saving
            agent_normalized_count = 0
            for call in call_data:
                if isinstance(call, dict) and "agent" in call:
                    original_agent = call.get("agent")
                    normalized_agent = normalize_agent_id(original_agent)
                    if original_agent != normalized_agent:
                        call["agent"] = normalized_agent
                        agent_normalized_count += 1
            if agent_normalized_count > 0:
                logger.info(
                    f" Normalized {agent_normalized_count} agent IDs before saving to cache"
                )

            call_data = deduplicate_calls(call_data)
            if len(call_data) < original_count:
                logger.info(
                    f" Deduplicated before save: {original_count} → {len(call_data)} unique calls"
                )

        cache_data = {
            "call_data": call_data,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "last_save_time": time.time(),  # Track last save time
            "count": len(call_data),
            "partial": partial,  # Mark as partial or complete
        }

        # Add progress info for partial saves
        if partial:
            cache_data["processed"] = processed
            cache_data["total"] = total

        # Use file locking and atomic writes
        # Retry on lock timeout with exponential backoff
        max_lock_retries = 3
        for lock_attempt in range(max_lock_retries):
            try:
                with cache_file_lock(CACHE_FILE, timeout=10):
                    atomic_write_json(CACHE_FILE, cache_data)
                break  # Success, exit retry loop
            except LockTimeoutError as e:
                if lock_attempt < max_lock_retries - 1:
                    wait_time = 0.5 * (lock_attempt + 1)
                    logger.warning(
                        f" Lock timeout on save attempt {lock_attempt + 1}/{max_lock_retries}: {e}, retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f" Failed to acquire lock for cache save after {max_lock_retries} attempts: {e}"
                    )
                    raise  # Re-raise on final failure

        status = "PARTIAL" if partial else "COMPLETE"
        logger.info(
            f" Successfully saved {len(call_data)} calls to persistent cache ({status}): {CACHE_FILE}"
        )

        # Also save to S3 for persistence across deployments
        s3_client, s3_bucket = get_s3_client_and_bucket()
        if s3_client and s3_bucket:
            # PROTECTION: Check existing cache before overwriting with PARTIAL cache
            if partial:  # Only check if saving PARTIAL cache
                try:
                    # Create S3 client with timeout for checking existing cache
                    config = botocore.config.Config(
                        connect_timeout=10, read_timeout=30, retries={"max_attempts": 2}
                    )
                    s3_check_client = boto3.client(
                        "s3",
                        aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
                        aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
                        region_name=st.secrets["s3"].get("region_name", "us-east-1"),
                        config=config,
                    )

                    existing_response = s3_check_client.get_object(
                        Bucket=s3_bucket, Key=S3_CACHE_KEY
                    )
                    # Stream download for existing cache check
                    body = existing_response["Body"]
                    chunks = []
                    for chunk in body.iter_chunks(chunk_size=8192):
                        chunks.append(chunk)
                    existing_data = json.loads(b"".join(chunks).decode("utf-8"))
                    existing_is_partial = existing_data.get("partial", False)
                    existing_call_data = existing_data.get("call_data", [])
                    existing_count = len(existing_call_data)

                    if not existing_is_partial:  # Existing cache is COMPLETE
                        # CRITICAL FIX: Allow overwriting COMPLETE cache if it has 0 calls (invalid/empty)
                        # A COMPLETE cache with 0 calls is essentially invalid and should be overwritten
                        if existing_count == 0:
                            logger.info(
                                f" COMPLETE cache has 0 calls (invalid/empty) - allowing overwrite with PARTIAL cache ({len(call_data)} calls)"
                            )
                            # Continue to save below - don't return early
                        else:
                            logger.warning(
                                f" PROTECTED: Not overwriting COMPLETE cache ({existing_count} calls) "
                                f"with PARTIAL cache ({len(call_data)} calls). "
                                f"Complete cache preserved. Local cache saved successfully."
                            )
                            return  # Don't overwrite COMPLETE cache with PARTIAL
                    else:
                        # Both are PARTIAL - check progress
                        existing_processed = existing_data.get("processed", 0)
                        existing_total = existing_data.get("total", 0)
                        new_processed = processed
                        new_total = total

                        # Check if new cache is more complete
                        new_is_better = False
                        if new_processed > existing_processed:
                            new_is_better = True
                        elif (
                            new_processed == existing_processed
                            and len(call_data) > existing_count
                        ):
                            new_is_better = True

                        if not new_is_better:
                            logger.warning(
                                f" PROTECTED: Not overwriting PARTIAL cache ({existing_processed}/{existing_total}, {existing_count} calls) "
                                f"with less complete PARTIAL cache ({new_processed}/{new_total}, {len(call_data)} calls). "
                                f"More complete cache preserved. Local cache saved successfully."
                            )
                            return  # Don't overwrite with less complete cache

                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code != "NoSuchKey":
                        logger.warning(
                            f" Could not check existing cache status: {e}. Proceeding with save."
                        )
                    # If no existing cache, proceed with save
                except Exception as check_error:
                    logger.warning(
                        f" Error checking existing cache: {check_error}. Proceeding with save."
                    )
                    # Proceed with save if check fails

            # Proceed with save (protected by checks above)
            try:
                cache_json = json.dumps(cache_data, default=str, ensure_ascii=False)
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=S3_CACHE_KEY,
                    Body=cache_json.encode("utf-8"),
                    ContentType="application/json",
                )
                logger.info(
                    f" Successfully saved {len(call_data)} calls to S3 cache ({status}): s3://{s3_bucket}/{S3_CACHE_KEY}"
                )
            except Exception as s3_error:
                # Don't fail the entire save if S3 upload fails - local cache is still saved
                logger.warning(
                    f" Failed to save cache to S3 (local cache saved successfully): {s3_error}"
                )
    except LockTimeoutError:
        # Re-raise LockTimeoutError - callers need to know about this
        raise
    except Exception as e:
        # CRITICAL FIX: Re-raise all exceptions after logging so callers can handle failures
        # This prevents silent failures that cause data loss
        logger.error(f" Failed to save persistent cache: {e}")
        import traceback

        logger.error(f" Traceback: {traceback.format_exc()}")
        raise  # Re-raise so callers know save failed


# Cached wrapper - uses both Streamlit cache (fast) and disk cache (persistent, survives restarts)
# First load will take time, subsequent loads will be instant
# Use "Refresh New Data" button when new CSV files are added to S3 - it only loads new files (PDFs are ignored)
# Note: Using max_entries=1 to prevent cache from growing, and no TTL so it never auto-expires
@st.cache_data(ttl=None, max_entries=1, show_spinner=True)
def load_all_calls_cached(cache_version=0):
    """Cached wrapper - loads ALL data once, then serves from cache indefinitely until manually refreshed.
    Performs one-time cache cleanup before loading to remove PDF-sourced calls.

    cache_version parameter forces cache refresh when incremented (used after refresh completes).

    Strategy:
        1. Always check S3 cache first (source of truth, shared across all users)
        2. Invalidate Streamlit cache if S3 cache is newer
        3. Use Streamlit cache only if it matches S3 cache timestamp
        4. Fall back to disk cache only if S3 unavailable (backup only)

    For incremental updates, use the "Refresh New Data" button which calls load_new_calls_only().
    """
    # Perform one-time cache cleanup to remove PDF-sourced calls
    cleanup_pdf_sourced_calls()
    import time

    start_time = time.time()

    # Prevent concurrent loads - check if a load is already in progress
    load_in_progress_key = "_data_load_in_progress"
    load_start_time_key = "_data_load_start_time"

    if st.session_state.get(load_in_progress_key, False):
        # Check if load has been in progress for too long (timeout after 30 minutes)
        load_start_time = st.session_state.get(load_start_time_key, time.time())
        load_duration = time.time() - load_start_time
        timeout_seconds = 30 * 60  # 30 minutes

        if load_duration > timeout_seconds:
            logger.warning(
                f"Data load has been in progress for {load_duration / 60:.1f} minutes "
                f"(exceeded {timeout_seconds / 60:.0f} minute timeout), clearing flag and retrying"
            )
            # Clear the flag and start a new load
            if load_in_progress_key in st.session_state:
                del st.session_state[load_in_progress_key]
            if load_start_time_key in st.session_state:
                del st.session_state[load_start_time_key]
        else:
            logger.warning(
                f"Data load already in progress (started {load_duration:.1f}s ago), "
                f"waiting for completion or using cache"
            )
            # Wait for the first load to complete (max 60 seconds, check every 1 second)
            # This allows time for the full load to complete (typically 30-40 seconds)
            max_wait = 60.0
            wait_interval = 1.0
            waited = 0.0
            while waited < max_wait:
                time.sleep(wait_interval)
                waited += wait_interval

                # Check if load completed
                if not st.session_state.get(load_in_progress_key, False):
                    # Load completed, try to get cached result
                    logger.info("First load completed, retrieving cached data")
                    break

                # Check for cache availability periodically (every 2 seconds)
                if int(waited) % 2 == 0:
                    # Try disk cache
                    try:
                        disk_result = load_cached_data_from_disk()
                        if disk_result and disk_result[0] and len(disk_result[0]) > 0:
                            migrated = migrate_old_cache_format(disk_result[0])
                            logger.info(
                                f"Found disk cache during wait: {len(migrated)} calls (waited {waited:.1f}s)"
                            )
                            return migrated, disk_result[1] if disk_result[1] else []
                    except Exception as e:
                        logger.debug(f"Disk cache not available yet: {e}")

                    # Try session state cache
                    if "_s3_cache_result" in st.session_state:
                        cached_result = st.session_state["_s3_cache_result"]
                        if cached_result and len(cached_result) >= 100:
                            logger.info(
                                f"Found session cache during wait: {len(cached_result)} calls (waited {waited:.1f}s)"
                            )
                            return cached_result, st.session_state.get(
                                "_last_load_errors", []
                            )

            # Try multiple sources for data after wait
            # 1. Try session state cache first (fastest)
            if "_s3_cache_result" in st.session_state:
                cached_result = st.session_state["_s3_cache_result"]
                if cached_result and len(cached_result) >= 100:
                    logger.info(
                        f"Returning session cache after wait: {len(cached_result)} calls"
                    )
                    return cached_result, st.session_state.get("_last_load_errors", [])

            # 2. Try Streamlit cache (if load completed)
            try:
                # Force cache refresh by using a dummy cache key
                # The actual cache will be used if available
                from streamlit.runtime.caching import cache_data_api

                # Try to get from cache using the function's cache key
                cache_key = f"load_all_calls_cached_{cache_version}"
                if hasattr(st.session_state, "_cached_call_data"):
                    cached_data = st.session_state.get("_cached_call_data")
                    if cached_data and len(cached_data) > 0:
                        logger.info(
                            f"Returning cached data from session state: {len(cached_data)} calls"
                        )
                        return cached_data, []
            except Exception as e:
                logger.debug(f"Could not get from Streamlit cache: {e}")

            # 3. Try disk cache
            try:
                disk_result = load_cached_data_from_disk()
                if disk_result and disk_result[0] and len(disk_result[0]) > 0:
                    migrated = migrate_old_cache_format(disk_result[0])
                    logger.info(
                        f"Returning disk cache after wait: {len(migrated)} calls"
                    )
                    return migrated, disk_result[1] if disk_result[1] else []
            except Exception as e:
                logger.warning(f"Could not load disk cache after wait: {e}")

            # 3. Try S3 cache from session state (most up-to-date, shared across users)
            try:
                s3_cache_key = "_s3_cache_result"
                s3_timestamp_key = "_s3_cache_timestamp"
                if (
                    s3_cache_key in st.session_state
                    and s3_timestamp_key in st.session_state
                ):
                    s3_cached_data = st.session_state[s3_cache_key]
                    if (
                        s3_cached_data
                        and isinstance(s3_cached_data, tuple)
                        and len(s3_cached_data) >= 1
                    ):
                        if s3_cached_data[0] and len(s3_cached_data[0]) > 0:
                            migrated = migrate_old_cache_format(s3_cached_data[0])
                            logger.info(
                                f"Returning S3 cache from session state during concurrent load: {len(migrated)} calls"
                            )
                            return migrated, s3_cached_data[1] if len(
                                s3_cached_data
                            ) > 1 and s3_cached_data[1] else []
            except Exception as e:
                logger.debug(
                    f"Could not get S3 cache from session state during concurrent load: {e}"
                )

            # If all else fails, return empty (will retry on next run)
            if st.session_state.get(load_in_progress_key, False):
                logger.warning(
                    "Load still in progress and no cache available, returning empty (will retry)"
                )
            else:
                logger.warning(
                    "Load completed but no cached data found, returning empty"
                )
            return [], []

    # Mark load as in progress with timestamp
    st.session_state[load_in_progress_key] = True
    st.session_state[load_start_time_key] = time.time()
    try:
        # Check if user explicitly requested full reload - MUST check BEFORE loading cache
        reload_all_triggered = st.session_state.get("reload_all_triggered", False)

        # CRITICAL: If reload_all_triggered, delete ALL caches BEFORE trying to load
        if reload_all_triggered:
            logger.info(
                "Reload ALL Data triggered - clearing all caches before loading"
            )

            # Delete disk cache file with retry and validation
            disk_cache_deleted = False
            try:
                if CACHE_FILE.exists():
                    with cache_file_lock(CACHE_FILE, timeout=5):
                        CACHE_FILE.unlink()
                        # Validate deletion succeeded
                        if not CACHE_FILE.exists():
                            disk_cache_deleted = True
                            logger.info(f" Deleted disk cache file: {CACHE_FILE}")
                        else:
                            logger.warning(
                                f" Disk cache file still exists after deletion attempt"
                            )
            except LockTimeoutError:
                logger.warning(
                    f" Timeout deleting disk cache (file may be locked by another process)"
                )
            except Exception as e:
                logger.warning(f" Could not delete disk cache: {e}")

            # Delete S3 cache with retry logic
            s3_cache_deleted = False
            max_s3_retries = 3
            for retry in range(max_s3_retries):
                try:
                    s3_client, s3_bucket = get_s3_client_and_bucket()
                    if s3_client and s3_bucket:
                        try:
                            s3_client.delete_object(Bucket=s3_bucket, Key=S3_CACHE_KEY)
                            s3_cache_deleted = True
                            logger.info(
                                f" Deleted S3 cache: s3://{s3_bucket}/{S3_CACHE_KEY}"
                            )
                            break
                        except ClientError as e:
                            if e.response.get("Error", {}).get("Code") != "NoSuchKey":
                                if retry < max_s3_retries - 1:
                                    logger.warning(
                                        f" S3 cache deletion failed (attempt {retry + 1}/{max_s3_retries}), retrying..."
                                    )
                                    time.sleep(0.5)
                                    continue
                                else:
                                    logger.warning(
                                        f" Could not delete S3 cache after {max_s3_retries} attempts: {e}"
                                    )
                            else:
                                s3_cache_deleted = True
                                logger.info(
                                    " S3 cache does not exist (already deleted)"
                                )
                                break
                except Exception as e:
                    if retry < max_s3_retries - 1:
                        logger.warning(
                            f" S3 cache deletion error (attempt {retry + 1}/{max_s3_retries}), retrying..."
                        )
                        time.sleep(0.5)
                        continue
                    else:
                        logger.warning(
                            f" Could not delete S3 cache after {max_s3_retries} attempts: {e}"
                        )

            # Clear Streamlit cache
            streamlit_cache_cleared = False
            try:
                load_all_calls_cached.clear()
                streamlit_cache_cleared = True
                logger.info(" Cleared Streamlit cache")
            except Exception as e:
                logger.warning(f" Could not clear Streamlit cache: {e}")

            # Clear session state cache FIRST to prevent stale data issues
            session_cache_cleared = True
            session_keys_to_clear = [
                "_s3_cache_result",
                "_s3_cache_timestamp",
                "_merged_cache_data",
                "_merged_cache_errors",
                "_merged_cache_data_timestamp",
            ]
            for key in session_keys_to_clear:
                try:
                    if key in st.session_state:
                        del st.session_state[key]
                except Exception as clear_error:
                    logger.warning(
                        f" Error clearing session state key '{key}': {clear_error}"
                    )
                    session_cache_cleared = False

            if session_cache_cleared:
                logger.info(" Cleared session state cache")

            # Log summary of cache clearing results
            logger.info(
                f" Cache clearing summary: disk={disk_cache_deleted}, s3={s3_cache_deleted}, "
                f"streamlit={streamlit_cache_cleared}, session={session_cache_cleared}"
            )

            # DON'T clear the flag here - keep it set so the load knows to load ALL files
            # The flag will be cleared after the load completes successfully
            # Skip S3 cache check - we just deleted it, need fresh load
            s3_cache_result = None
            s3_cache_timestamp = None
            logger.info(
                " Skipping S3 cache check - caches deleted, will load fresh from S3"
            )
        else:
            # CRITICAL: Always check S3 cache first (source of truth, shared across all users)
            # This ensures all users get the same up-to-date data
            # Use session state caching to prevent duplicate S3 loads
            s3_cache_result = None
            s3_cache_timestamp = None

            # Check session state first to avoid duplicate S3 loads
            s3_cache_key = "_s3_cache_result"
            s3_timestamp_key = "_s3_cache_timestamp"
            if (
                s3_cache_key in st.session_state
                and s3_timestamp_key in st.session_state
            ):
                cached_timestamp = st.session_state[s3_timestamp_key]
                # Use cached result if timestamp matches (cache is still valid)
                # CRITICAL: Validate cached result before accessing to prevent crashes from corrupted session state
                cached_result = st.session_state[s3_cache_key]
                if (
                    cached_result is not None
                    and isinstance(cached_result, tuple)
                    and len(cached_result) >= 1
                    and cached_result[0] is not None
                ):
                    s3_cache_result = cached_result
                    s3_cache_timestamp = cached_timestamp
                    logger.debug(
                        f" Using session-cached S3 result: {len(s3_cache_result[0])} calls (timestamp: {s3_cache_timestamp})"
                    )
                else:
                    # Session state contains invalid data - clear it and reload from S3
                    logger.warning(
                        "Session state contains invalid S3 cache data, clearing and reloading from S3"
                    )
                    if s3_cache_key in st.session_state:
                        del st.session_state[s3_cache_key]
                    if s3_timestamp_key in st.session_state:
                        del st.session_state[s3_timestamp_key]
                    s3_cache_result = None
                    s3_cache_timestamp = None
            else:
                # Load from S3 (only if not in session state)
                s3_client, s3_bucket = get_s3_client_and_bucket()
                if s3_client and s3_bucket:
                    try:
                        response = s3_client.get_object(
                            Bucket=s3_bucket, Key=S3_CACHE_KEY
                        )
                        s3_cached_data = json.loads(
                            response["Body"].read().decode("utf-8")
                        )
                        if isinstance(s3_cached_data, dict):
                            s3_cache_result = (
                                s3_cached_data.get("call_data", []),
                                s3_cached_data.get("errors", []),
                            )
                            s3_cache_timestamp = s3_cached_data.get("timestamp", None)
                            logger.info(
                                f" Loaded from S3 cache (source of truth): {len(s3_cache_result[0])} calls (timestamp: {s3_cache_timestamp})"
                            )

                            # Cache in session state to avoid duplicate loads
                            st.session_state[s3_cache_key] = s3_cache_result
                            if s3_cache_timestamp:
                                st.session_state[s3_timestamp_key] = s3_cache_timestamp
                    except ClientError as s3_error:
                        error_code = s3_error.response.get("Error", {}).get("Code", "")
                        if error_code != "NoSuchKey":
                            logger.warning(f" Failed to load from S3 cache: {s3_error}")
                    except Exception as s3_error:
                        logger.warning(f" Failed to load from S3 cache: {s3_error}")

        # CRITICAL: If S3 cache exists, check if Streamlit cache is stale
        # (Skip this check if we just deleted caches due to reload_all_triggered)
        # Invalidate Streamlit cache if S3 cache is newer
        if s3_cache_timestamp:
            streamlit_cache_timestamp = st.session_state.get(
                "_s3_cache_timestamp", None
            )
        if (
            streamlit_cache_timestamp
            and streamlit_cache_timestamp != s3_cache_timestamp
        ):
            # S3 cache has been updated - clear Streamlit cache AND session state cache
            logger.info(
                f" S3 cache updated ({s3_cache_timestamp} != {streamlit_cache_timestamp}) - invalidating caches"
            )
            try:
                # CRITICAL FIX: Only clear cache if we're not already in a cache-clearing operation
                # This prevents recursive cache clearing that can cause crashes
                if not st.session_state.get("_cache_clearing_in_progress", False):
                    st.session_state["_cache_clearing_in_progress"] = True
                    try:
                        load_all_calls_cached.clear()
                        # Clear session state cache so we reload from S3
                        if s3_cache_key in st.session_state:
                            del st.session_state[s3_cache_key]
                        logger.info(
                            "Cleared stale Streamlit cache and session state cache - will reload from S3"
                        )
                    finally:
                        # Always clear the flag, even if clearing failed
                        if "_cache_clearing_in_progress" in st.session_state:
                            del st.session_state["_cache_clearing_in_progress"]
                else:
                    logger.debug(
                        "Cache clearing already in progress, skipping to prevent recursion"
                    )
            except Exception as clear_error:
                logger.warning(f"Could not clear Streamlit cache: {clear_error}")
                # Clear the flag even on error
                if "_cache_clearing_in_progress" in st.session_state:
                    del st.session_state["_cache_clearing_in_progress"]
            # Store S3 cache timestamp for future comparison
            st.session_state["_s3_cache_timestamp"] = s3_cache_timestamp

        # CRITICAL: If refresh is in progress, use S3 cache directly (most up-to-date)
        refresh_in_progress = st.session_state.get("refresh_in_progress", False)
        if refresh_in_progress:
            if s3_cache_result and s3_cache_result[0]:
                # Migrate old cache format to new format
                migrated_calls = migrate_old_cache_format(s3_cache_result[0])
                logger.info(
                    f" Refresh in progress - using S3 cache directly: {len(migrated_calls)} calls"
                )
                return (
                    migrated_calls,
                    s3_cache_result[1] if len(s3_cache_result) > 1 else [],
                )
            else:
                logger.info(
                    " Refresh in progress but no S3 cache found - continuing with normal load"
                )

        # Check if there's merged cache data from refresh operation
        # BUG FIX: Only use _merged_cache_data if S3 cache is not newer
        # If S3 cache is newer, clear stale _merged_cache_data and reload from S3
        if "_merged_cache_data" in st.session_state:
            # Check if S3 cache is newer than when _merged_cache_data was set
            if (
                s3_cache_timestamp
                and "_merged_cache_data_timestamp" in st.session_state
            ):
                merged_data_timestamp = st.session_state["_merged_cache_data_timestamp"]
                if s3_cache_timestamp > merged_data_timestamp:
                    # S3 cache is newer - clear stale merged data and reload from S3
                    logger.info(
                        f" S3 cache is newer ({s3_cache_timestamp} > {merged_data_timestamp}) - clearing stale _merged_cache_data"
                    )
                    del st.session_state["_merged_cache_data"]
                    if "_merged_cache_errors" in st.session_state:
                        del st.session_state["_merged_cache_errors"]
                    if "_merged_cache_data_timestamp" in st.session_state:
                        del st.session_state["_merged_cache_data_timestamp"]
                # Fall through to use S3 cache below
            else:
                # Use merged cache data (it's still current)
                merged_data = st.session_state["_merged_cache_data"]
                merged_errors = st.session_state.get("_merged_cache_errors", [])
                # Migrate old cache format to new format
                merged_data = migrate_old_cache_format(merged_data)
                logger.info(
                    f" Using merged cache data from refresh: {len(merged_data)} calls"
                )
                # Store S3 timestamp if available
                if s3_cache_timestamp:
                    st.session_state["_s3_cache_timestamp"] = s3_cache_timestamp
                return merged_data, merged_errors
        # If _merged_cache_data doesn't exist, fall through to use S3 cache or disk cache

        # CRITICAL: Use S3 cache if available (source of truth)
        if s3_cache_result and s3_cache_result[0]:
            # Migrate old cache format to new format
            migrated_calls = migrate_old_cache_format(s3_cache_result[0])
            logger.info(
                f" Using S3 cache (source of truth): {len(migrated_calls)} calls"
            )
            # Store timestamp for future comparison
            if s3_cache_timestamp:
                st.session_state["_s3_cache_timestamp"] = s3_cache_timestamp
            return (
                migrated_calls,
                s3_cache_result[1] if len(s3_cache_result) > 1 else [],
            )

        # Fall back to disk cache only if S3 unavailable (backup only)
        # NOTE: We don't call load_cached_data_from_disk() here because it also loads from S3
        # We'll call it later only if needed, and it will check session state first
        # Disk cache is just a local backup, not the source of truth

        # Strategy: Always use the most up-to-date cache
        # 1. Check disk cache first (if not reloading)
        # 2. Try to load (will use Streamlit cache if available, or load from S3)
        # 3. Compare and use the best/most recent data
        # 4. Update disk cache with the best data

        # CRITICAL: Always check disk cache FIRST to prevent restart loops
        # The app keeps restarting during loads, losing progress. We MUST use partial caches.
        disk_call_data = None
        disk_errors = None
        is_partial = False
        partial_processed = 0
        partial_total = 0

        # NOTE: Removed duplicate check for _merged_cache_data - it's already checked earlier at line 1271
        # and returns immediately, making this check unreachable dead code

        # Load disk cache regardless of reload_all_triggered - we'll check that flag later
        disk_result = load_cached_data_from_disk()
        # CRITICAL FIX: Check if disk_result is None before accessing its elements
        if disk_result and disk_result[0] is not None and len(disk_result[0]) > 0:
            disk_call_data, disk_errors = disk_result
            # Migrate old cache format to new format
            disk_call_data = migrate_old_cache_format(disk_call_data)
            cache_count = len(disk_call_data)

            # Get cache metadata
            if CACHE_FILE.exists():
                try:
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                        is_partial = cached_data.get("partial", False)
                        partial_processed = cached_data.get("processed", 0)
                        partial_total = cached_data.get("total", 0)
                except Exception as e:
                    logger.warning(f" Failed to read cache metadata: {e}")

            # Log cache comparison immediately after loading disk cache
            logger.info(
                f" Cache Comparison: Disk cache = {cache_count} files (partial={is_partial}, {partial_processed}/{partial_total if partial_total > 0 else '?'} processed)"
            )

            # CRITICAL: Only return early if cache is COMPLETE (not partial)
            # If partial, continue loading to complete it with incremental saves
            if not reload_all_triggered:
                if cache_count >= 100:  # Use cache if we have 100+ calls
                    if is_partial:
                        progress_pct = (
                            (partial_processed * 100 // partial_total)
                            if partial_total > 0
                            else 0
                        )
                        logger.info(
                            f" Found PARTIAL cache: {cache_count} calls ({progress_pct}% complete)"
                        )
                        logger.info(
                            "Will continue loading remaining files from S3 with incremental saves"
                        )
                        # Don't return early - continue to load remaining files
                    else:
                        logger.info(
                            f" USING COMPLETE DISK CACHE: {cache_count} calls - prevents restart loss"
                        )

                        # CRITICAL FIX: Clean up _merged_cache_data if Streamlit cache has correct count
                        # This prevents keeping stale _merged_cache_data when Streamlit cache is already updated
                        if "_merged_cache_data" in st.session_state:
                            expected_count = len(st.session_state["_merged_cache_data"])
                            if cache_count >= expected_count:
                                # Streamlit cache has correct data (or more), safe to delete _merged_cache_data
                                logger.info(
                                    f" Streamlit cache confirmed updated ({cache_count} >= {expected_count} calls), cleaning up _merged_cache_data"
                                )
                                del st.session_state["_merged_cache_data"]
                                if "_merged_cache_errors" in st.session_state:
                                    del st.session_state["_merged_cache_errors"]

                        # Return disk cache (will update Streamlit cache with this value)
                        logger.info(
                            f" USING COMPLETE DISK CACHE: {cache_count} calls - prevents restart loss"
                        )
                        return disk_call_data, disk_errors if disk_errors else []
                else:
                    logger.info(
                        f" Cache has only {cache_count} calls (< 100), will load from S3"
                    )
            else:
                logger.info(
                    f" Reload ALL Data triggered - ignoring cache with {cache_count} calls, will load fresh from S3"
                )

        # Only reach here if:
        # 1. No cache found, OR
        # 2. Cache has < 100 calls, OR
        # 3. User explicitly requested reload (reload_all_triggered = True)
        # 4. Cache is partial (will continue loading remaining files)

        # Determine what to load
        if reload_all_triggered:
            # User explicitly requested full dataset - load ALL files (may take 10-20 min)
            # NOTE: Caches were already cleared at the beginning of this function (lines 2056-2104)
            # No need to clear again here to avoid duplicate operations
            logger.info(
                "Reload ALL Data triggered - loading ALL files from S3 (this will take 10-20 minutes)"
            )
            max_files = None  # Load all files
            # Clear flag after use (but keep it until load completes to prevent re-triggering)
            # We'll clear it at the end of the try block
        elif (
            disk_call_data
            and is_partial
            and partial_total > 0
            and partial_processed < partial_total
        ):
            # Partial cache exists - but limit auto-continuation to prevent crashes
            remaining_files = partial_total - partial_processed

            # Only auto-continue if remaining files is small (<= 1000) to prevent long blocking operations
            # This prevents crashes from trying to load thousands of files synchronously during page load
            if remaining_files <= 1000:
                logger.info(
                    f" Continuing partial cache: {partial_processed}/{partial_total} files loaded, {remaining_files} remaining"
                )
                logger.info(
                    f" Loading remaining {remaining_files} files from S3 with incremental saves (skipping already-processed files)"
                )

                try:
                    load_start = time.time()
                    new_calls, new_errors, actual_new_count = load_new_calls_only()
                    load_duration = time.time() - load_start
                    elapsed = time.time() - start_time

                    if isinstance(new_errors, str):
                        logger.error(f" Error loading new files: {new_errors}")
                        # Return existing cache if new load fails
                        return disk_call_data, [new_errors] if disk_errors else []

                    # Merge new calls with existing disk cache
                    all_calls_merged = disk_call_data + new_calls
                    all_calls_merged = deduplicate_calls(all_calls_merged)

                    # Save merged data to disk immediately
                    save_cached_data_to_disk(
                        all_calls_merged, new_errors if new_errors else []
                    )

                    logger.info(
                        f" Loaded {actual_new_count} new files. Total: {len(all_calls_merged)} calls (merged with {len(disk_call_data)} from cache)"
                    )

                    # Return merged data
                    return all_calls_merged, new_errors if new_errors else []
                except Exception as e:
                    logger.error(f" Error in load_new_calls_only: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
                    # Fallback to disk cache if new load fails
                    return disk_call_data, disk_errors if disk_errors else []
            else:
                # Too many files remaining - return partial cache and let user manually refresh
                # This prevents crashes from long-running operations during page load
                logger.info(
                    f" Found PARTIAL cache: {partial_processed}/{partial_total} files ({remaining_files} remaining)"
                )
                logger.info(
                    f" Returning partial cache immediately - use 'Refresh New Data' button to load remaining {remaining_files} files"
                )
                logger.info(
                    f" Auto-continuation skipped to prevent crashes (>{1000} files remaining)"
                )
                return disk_call_data, disk_errors if disk_errors else []
        else:
            # No substantial cache - load ALL files from S3
            logger.info("No substantial cache found - loading ALL files from S3")
            max_files = None  # Load all files

        try:
            load_start = time.time()
            result = load_all_calls_internal(max_files=max_files)
            load_duration = time.time() - load_start
            elapsed = time.time() - start_time

            # Ensure we always return a tuple
            if not isinstance(result, tuple) or len(result) != 2:
                result = (result if isinstance(result, list) else [], [])

            streamlit_call_data, streamlit_errors = result
            # Migrate old cache format to new format
            if streamlit_call_data:
                streamlit_call_data = migrate_old_cache_format(streamlit_call_data)

            # Log cache counts for debugging (this happens after S3 load, so Streamlit cache may have data now)
            streamlit_cache_count = (
                len(streamlit_call_data) if streamlit_call_data else 0
            )
            disk_cache_count = len(disk_call_data) if disk_call_data else 0
            logger.info(
                f" Cache Comparison (after load): Streamlit cache = {streamlit_cache_count} files, Disk cache = {disk_cache_count} files"
            )

            # Determine which cache is better (more recent or more complete)
            use_streamlit_cache = False
            use_disk_cache = False

            if streamlit_call_data and len(streamlit_call_data) > 0:
                # Streamlit cache has data - check if it's better than disk cache
                if load_duration < 2.0:
                    # Fast load = likely from Streamlit's in-memory cache
                    logger.info(
                        f"⚡ Detected Streamlit in-memory cache ({len(streamlit_call_data)} calls loaded in {load_duration:.2f}s)"
                    )

                if disk_call_data and len(disk_call_data) > 0:
                    # CRITICAL: Always prefer disk cache if it has more data, regardless of load speed
                    # This ensures we never show stale Streamlit cache when disk cache is more complete
                    if len(disk_call_data) > len(streamlit_call_data):
                        logger.info(
                            f" Disk cache is more complete ({len(disk_call_data)} vs {len(streamlit_call_data)} calls) - ALWAYS using disk cache"
                        )
                        use_disk_cache = True
                    elif len(streamlit_call_data) > len(disk_call_data):
                        logger.info(
                            f" Streamlit cache is more complete ({len(streamlit_call_data)} vs {len(disk_call_data)} calls) - using it"
                        )
                        use_streamlit_cache = True
                    elif len(streamlit_call_data) == len(disk_call_data):
                        # Same size - prefer disk cache as source of truth (it persists across restarts)
                        logger.info(
                            "Caches match - using disk cache as source of truth (persists across restarts)"
                        )
                        use_disk_cache = True
                else:
                    # No disk cache - use Streamlit cache
                    use_streamlit_cache = True
            else:
                # Slow load = loaded from S3, not from cache
                # This is new data, use it and ALWAYS save to disk (critical for persistence)
                logger.info(
                    f" Loaded {len(streamlit_call_data)} calls from S3 (took {load_duration:.1f}s)"
                )
                logger.info("CRITICAL: Saving to disk cache to prevent loss on restart")
                use_streamlit_cache = True
                # Force save to disk immediately for slow loads (they're fresh data from S3)
                if streamlit_call_data:
                    # CRITICAL FIX: Wrap save in try/except so save failures don't cause function to return empty data
                    try:
                        save_cached_data_to_disk(streamlit_call_data, streamlit_errors)
                    except Exception as save_error:
                        logger.error(
                            f" Failed to save S3-loaded data to disk: {save_error}"
                        )
                        logger.warning(
                            "Data was successfully loaded but not saved - will be lost on restart"
                        )
                        # Continue anyway - return the successfully loaded data
                elif disk_call_data and len(disk_call_data) > 0:
                    # Only disk cache has data
                    logger.info(f" Using disk cache ({len(disk_call_data)} calls)")
                    use_disk_cache = True

            # Use the best cache
            if use_streamlit_cache:
                final_call_data, final_errors = streamlit_call_data, streamlit_errors
                # Deduplicate Streamlit cache data
                if final_call_data:
                    original_count = len(final_call_data)
                    final_call_data = deduplicate_calls(final_call_data)
                    if len(final_call_data) < original_count:
                        logger.info(
                            f" Deduplicated Streamlit cache: {original_count} → {len(final_call_data)} unique calls"
                        )
                # ALWAYS update disk cache with Streamlit cache data (it's more recent)
                # This ensures Streamlit cache is preserved to disk for future restarts
                if final_call_data:
                    # Only skip save if we already saved it above (for slow S3 loads)
                    if load_duration >= 2.0:
                        # Already saved above for slow loads, just log
                        logger.info(
                            "Streamlit cache data already saved to disk (from S3 load)"
                        )
                    else:
                        # Fast load = from Streamlit cache, save it to disk now
                        # CRITICAL FIX: Wrap save in try/except so save failures don't cause function to return empty data
                        try:
                            save_cached_data_to_disk(final_call_data, final_errors)
                            logger.info(
                                f" Saved Streamlit cache to disk cache ({len(final_call_data)} calls) - preserved for future restarts"
                            )
                        except Exception as save_error:
                            logger.error(
                                f" Failed to save Streamlit cache to disk: {save_error}"
                            )
                            logger.warning(
                                "Data is available but not saved - will be lost on restart"
                            )
                            # Continue anyway - return the successfully loaded data
            elif use_disk_cache:
                final_call_data, final_errors = disk_call_data, disk_errors
                # Deduplicate disk cache data (should already be deduplicated, but double-check)
                if final_call_data:
                    original_count = len(final_call_data)
                    final_call_data = deduplicate_calls(final_call_data)
                    if len(final_call_data) < original_count:
                        logger.info(
                            f" Deduplicated disk cache: {original_count} → {len(final_call_data)} unique calls"
                        )
                    logger.info(
                        "Using disk cache - populating Streamlit's in-memory cache for faster access"
                    )
            else:
                # No cache available - use what we loaded
                final_call_data, final_errors = streamlit_call_data, streamlit_errors
                # Deduplicate before saving
                if final_call_data:
                    original_count = len(final_call_data)
                    final_call_data = deduplicate_calls(final_call_data)
                    if len(final_call_data) < original_count:
                        logger.info(
                            f" Deduplicated fresh load: {original_count} → {len(final_call_data)} unique calls"
                        )
                    # CRITICAL FIX: Wrap save in try/except so save failures don't cause function to return empty data
                    try:
                        save_cached_data_to_disk(final_call_data, final_errors)
                    except Exception as save_error:
                        logger.error(f"Failed to save fresh load to disk: {save_error}")
                        logger.warning(
                            "Data was successfully loaded but not saved - will be lost on restart"
                        )
                        # Continue anyway - return the successfully loaded data

            logger.info(
                f" Total time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)"
            )

            # CRITICAL: Ensure cache is saved after full load completes
            # This is a safety net to ensure data persists even if earlier saves failed
            if final_call_data and len(final_call_data) > 0:
                # Verify cache was saved - if not, save it now
                try:
                    # Check if we already saved (to avoid duplicate saves)
                    # If this is a full load (not partial), ensure it's saved as COMPLETE
                    if not reload_all_triggered or len(final_call_data) >= 100:
                        # This is a full load - ensure it's saved as COMPLETE
                        # Note: save_cached_data_to_disk defaults to partial=False (COMPLETE)
                        # We'll let the existing saves handle it, but log for verification
                        logger.info(
                            f" Full load completed: {len(final_call_data)} calls loaded. "
                            f"Cache should be saved as COMPLETE (partial=False)"
                        )
                except Exception as verify_error:
                    logger.warning(f" Could not verify cache save: {verify_error}")

            # Clear reload_all_triggered flag after successful load
            if reload_all_triggered:
                st.session_state["reload_all_triggered"] = False
                logger.info(
                    f" Returning {len(final_call_data) if final_call_data else 0} calls from FULL dataset"
                )
            else:
                logger.info(
                    f" Returning {len(final_call_data) if final_call_data else 0} calls"
                )

            # Migrate old cache format to new format before returning
            # This also normalizes agent IDs in cached data
            if final_call_data:
                final_call_data = migrate_old_cache_format(final_call_data)

            # CRITICAL FIX: Store loaded data in session state BEFORE clearing load_in_progress flag
            # This allows concurrent requests to immediately access the data
            if final_call_data and len(final_call_data) > 0:
                st.session_state["_s3_cache_result"] = (final_call_data, final_errors)
                st.session_state["_s3_cache_timestamp"] = datetime.now().isoformat()
                logger.info(
                    f" Stored {len(final_call_data)} calls in session state for concurrent requests"
                )

            # Clear load in progress flag
            if load_in_progress_key in st.session_state:
                del st.session_state[load_in_progress_key]

            # Return the data - Streamlit's @st.cache_data automatically caches this return value
            # This ensures both caches are in sync with the most recent data
            return final_call_data, final_errors
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(
                f" Error in load_all_calls_cached after {elapsed:.1f} seconds: {e}"
            )

            # CRITICAL: Clear reload_all_triggered flag even on error to prevent infinite retry loop
            if reload_all_triggered:
                st.session_state["reload_all_triggered"] = False
                logger.error(
                    "Reload ALL Data failed - cleared flag to prevent retry loop"
                )

            # Clear load in progress flag and timestamp
            try:
                if load_in_progress_key in st.session_state:
                    del st.session_state[load_in_progress_key]
                if load_start_time_key in st.session_state:
                    del st.session_state[load_start_time_key]
            except Exception as clear_error:
                logger.warning(f"Failed to clear load progress flags: {clear_error}")

            # Try to return disk cache if available as fallback
            if disk_call_data and len(disk_call_data) > 0:
                logger.warning(
                    f" Reload failed, falling back to disk cache: {len(disk_call_data)} calls"
                )
                # Normalize agent IDs in fallback cache
                try:
                    disk_call_data = migrate_old_cache_format(disk_call_data)
                    return disk_call_data, [
                        f"Reload failed: {str(e)}. Using cached data."
                    ]
                except Exception as migrate_error:
                    logger.error(f"Failed to migrate fallback cache: {migrate_error}")
                    # Return unmigrated data rather than empty
                    return disk_call_data, [
                        f"Reload failed: {str(e)}. Using cached data (unmigrated)."
                    ]

            # Return empty data with error message only if no fallback available
            return [], [f"Failed to load data: {str(e)}"]
    except Exception as e:
        # Outer try block error handler
        elapsed = time.time() - start_time
        logger.exception(
            f" Critical error in load_all_calls_cached after {elapsed:.1f} seconds: {e}"
        )

        # Clear all flags
        if "reload_all_triggered" in st.session_state:
            st.session_state["reload_all_triggered"] = False
        if load_in_progress_key in st.session_state:
            del st.session_state[load_in_progress_key]

        # Return empty data
        return [], [f"Critical error: {str(e)}"]


# Chart caching helper - cache chart figures based on data hash
@st.cache_data(ttl=3600, show_spinner=False)  # Cache charts for 1 hour
def get_cached_chart_figure(chart_id: str, data_hash: str, chart_func, *args, **kwargs):
    """Cache matplotlib chart figures to avoid regeneration.

    Args:
        chart_id: Unique identifier for the chart type
        data_hash: Hash of the data to ensure cache invalidation on data change
        chart_func: Function that generates the chart
        *args, **kwargs: Arguments to pass to chart_func

    Returns:
        matplotlib figure object
    """
    return chart_func(*args, **kwargs)


def create_data_hash(df: pd.DataFrame, additional_params: dict = None) -> str:
    """Create a hash of the dataframe and parameters for cache key.

    Args:
        df: DataFrame to hash
        additional_params: Additional parameters to include in hash

    Returns:
        Hash string
    """
    import hashlib

    data_str = str(df.shape) + str(df.columns.tolist()) + str(df.index.tolist()[:10])
    if additional_params:
        data_str += str(additional_params)
    return hashlib.md5(data_str.encode()).hexdigest()


def create_filter_cache_key(
    date_range, agent_filter, score_filter, label_filter, search_text=None
):
    """Create a cache key based on filter parameters for chart caching.

    Args:
        date_range: Tuple of (start_date, end_date)
        agent_filter: List of selected agents
        score_filter: Tuple of (min_score, max_score)
        label_filter: List of selected labels
        search_text: Optional search text

    Returns:
        Cache key string
    """
    import hashlib

    key_parts = [
        str(date_range),
        str(sorted(agent_filter)) if agent_filter else "all",
        str(score_filter),
        str(sorted(label_filter)) if label_filter else "all",
        str(search_text) if search_text else "",
    ]
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


# Enhanced chart caching with filter-based cache keys
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_chart_with_filters(
    chart_id: str, cache_key: str, chart_func, *args, **kwargs
):
    """Cache matplotlib chart figures with filter-based cache keys.

    Args:
        chart_id: Unique identifier for the chart type
        cache_key: Cache key based on filter parameters
        chart_func: Function that generates the chart
        *args, **kwargs: Arguments to pass to chart_func

    Returns:
        matplotlib figure object
    """
    return chart_func(*args, **kwargs)


# --- Benchmarking Functions ---
def calculate_historical_baselines(df, current_start_date, current_end_date):
    """Calculate historical baselines for comparison.

    Args:
        df: DataFrame with call data
        current_start_date: Start date of current period (date or datetime)
        current_end_date: End date of current period (date or datetime)

    Returns:
        Dictionary with baseline metrics
    """
    from datetime import timedelta
    import pandas as pd

    # Convert to pandas Timestamp for consistent comparison
    if not isinstance(current_start_date, pd.Timestamp):
        current_start_date = pd.Timestamp(current_start_date)
    if not isinstance(current_end_date, pd.Timestamp):
        current_end_date = pd.Timestamp(current_end_date)

    baselines = {}

    # Get the earliest and latest dates in the dataset
    if df.empty or "Call Date" not in df.columns:
        return baselines

    df_min_date = df["Call Date"].min()
    df_max_date = df["Call Date"].max()

    # Last 30 days baseline
    # First try: data before current period
    last_30_start = current_end_date - timedelta(days=30)
    last_30_data = df[
        (df["Call Date"] >= last_30_start) & (df["Call Date"] < current_start_date)
    ]

    # Fallback: if no data before current period, use earlier 30 days from within dataset
    if last_30_data.empty:
        # Use 30 days ending just before current period, or if that's not possible,
        # use the earliest 30 days in the dataset
        if current_start_date > df_min_date:
            # Try to get 30 days before current_start_date
            fallback_end = current_start_date - timedelta(days=1)
            fallback_start = fallback_end - timedelta(days=30)
            if fallback_start < df_min_date:
                fallback_start = df_min_date
            last_30_data = df[
                (df["Call Date"] >= fallback_start) & (df["Call Date"] <= fallback_end)
            ]
        else:
            # Current period starts at beginning, use earliest 30 days
            last_30_data = df[
                (df["Call Date"] >= df_min_date)
                & (
                    df["Call Date"]
                    <= min(df_min_date + timedelta(days=30), df_max_date)
                )
            ]

    if not last_30_data.empty:
        baselines["last_30_days"] = {
            "avg_score": last_30_data["QA Score"].mean()
            if "QA Score" in last_30_data.columns
            else None,
            "pass_rate": calculate_pass_rate(last_30_data),
            "total_calls": len(last_30_data),
            "period": (
                last_30_data["Call Date"].min(),
                last_30_data["Call Date"].max(),
            ),
        }

    # Last 90 days baseline
    # First try: data before current period
    last_90_start = current_end_date - timedelta(days=90)
    last_90_data = df[
        (df["Call Date"] >= last_90_start) & (df["Call Date"] < current_start_date)
    ]

    # Fallback: if no data before current period, use earlier 90 days from within dataset
    if last_90_data.empty:
        if current_start_date > df_min_date:
            # Try to get 90 days before current_start_date
            fallback_end = current_start_date - timedelta(days=1)
            fallback_start = fallback_end - timedelta(days=90)
            if fallback_start < df_min_date:
                fallback_start = df_min_date
            last_90_data = df[
                (df["Call Date"] >= fallback_start) & (df["Call Date"] <= fallback_end)
            ]
        else:
            # Current period starts at beginning, use earliest 90 days
            last_90_data = df[
                (df["Call Date"] >= df_min_date)
                & (
                    df["Call Date"]
                    <= min(df_min_date + timedelta(days=90), df_max_date)
                )
            ]

    if not last_90_data.empty:
        baselines["last_90_days"] = {
            "avg_score": last_90_data["QA Score"].mean()
            if "QA Score" in last_90_data.columns
            else None,
            "pass_rate": calculate_pass_rate(last_90_data),
            "total_calls": len(last_90_data),
            "period": (
                last_90_data["Call Date"].min(),
                last_90_data["Call Date"].max(),
            ),
        }

    # Year-over-year (if data available)
    if current_start_date.year > df["Call Date"].min().year:
        yoy_start = current_start_date - timedelta(days=365)
        yoy_end = current_end_date - timedelta(days=365)
        yoy_data = df[(df["Call Date"] >= yoy_start) & (df["Call Date"] <= yoy_end)]
        if not yoy_data.empty:
            baselines["year_over_year"] = {
                "avg_score": yoy_data["QA Score"].mean()
                if "QA Score" in yoy_data.columns
                else None,
                "pass_rate": calculate_pass_rate(yoy_data),
                "total_calls": len(yoy_data),
                "period": (yoy_start, yoy_end),
            }

    return baselines


def calculate_pass_rate(df):
    """Calculate pass rate from dataframe."""
    if "Rubric Pass Count" in df.columns and "Rubric Fail Count" in df.columns:
        total_pass = df["Rubric Pass Count"].sum()
        total_fail = df["Rubric Fail Count"].sum()
        if (total_pass + total_fail) > 0:
            return (total_pass / (total_pass + total_fail)) * 100
    return None


def calculate_percentile_rankings(df, metric_col="QA Score"):
    """Calculate percentile rankings for agents.

    Args:
        df: DataFrame with agent performance data
        metric_col: Column name for the metric to rank

    Returns:
        DataFrame with percentile rankings
    """
    if metric_col not in df.columns:
        return pd.DataFrame()

    agent_perf = df.groupby("Agent")[metric_col].mean().reset_index()
    agent_perf["percentile"] = agent_perf[metric_col].rank(pct=True) * 100
    agent_perf = agent_perf.sort_values("percentile", ascending=False)

    return agent_perf


# --- Predictive Analytics Functions ---
def predict_future_scores(df, days_ahead=7):
    """Predict future QA scores using time series forecasting.

    Args:
        df: DataFrame with historical QA scores and dates
        days_ahead: Number of days to forecast

    Returns:
        Dictionary with forecast data and confidence intervals
    """
    try:
        from prophet import Prophet
    except ImportError:
        # Fallback to simple linear trend if Prophet not available
        return predict_future_scores_simple(df, days_ahead)

    if "Call Date" not in df.columns or "QA Score" not in df.columns:
        return None

    # Prepare data for Prophet
    daily_scores = df.groupby("Call Date")["QA Score"].mean().reset_index()
    daily_scores.columns = ["ds", "y"]
    daily_scores = daily_scores.sort_values("ds")

    if len(daily_scores) < 7:  # Need at least 7 days of data
        return predict_future_scores_simple(df, days_ahead)

    try:
        model = Prophet(
            interval_width=0.95, daily_seasonality=False, weekly_seasonality=True
        )
        model.fit(daily_scores)

        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)

        # Extract forecast for future dates only
        forecast_dates = forecast.tail(days_ahead)

        return {
            "dates": forecast_dates["ds"].dt.date.tolist(),
            "forecast": forecast_dates["yhat"].tolist(),
            "lower_bound": forecast_dates["yhat_lower"].tolist(),
            "upper_bound": forecast_dates["yhat_upper"].tolist(),
            "method": "prophet",
        }
    except Exception as e:
        logger.warning(f"Prophet forecasting failed: {e}, using simple method")
        return predict_future_scores_simple(df, days_ahead)


def predict_future_scores_simple(df, days_ahead=7):
    """Simple linear trend forecasting as fallback.

    Args:
        df: DataFrame with historical QA scores and dates
        days_ahead: Number of days to forecast

    Returns:
        Dictionary with forecast data
    """
    if "Call Date" not in df.columns or "QA Score" not in df.columns:
        return None

    daily_scores = df.groupby("Call Date")["QA Score"].mean().reset_index()
    daily_scores = daily_scores.sort_values("Call Date")

    if len(daily_scores) < 2:
        return None

    # Simple linear regression
    from datetime import timedelta
    import numpy as np
    from numpy.linalg import LinAlgError

    dates = daily_scores["Call Date"]
    scores = daily_scores["QA Score"]

    # Check if all dates are the same
    if dates.nunique() == 1:
        # All dates are the same - return flat forecast
        avg_score = scores.mean()
        return {
            "dates": [
                (dates.max() + timedelta(days=i + 1)).date() for i in range(days_ahead)
            ],
            "forecast": [avg_score] * days_ahead,
            "lower_bound": [avg_score - 5] * days_ahead,  # Simple confidence interval
            "upper_bound": [avg_score + 5] * days_ahead,
            "method": "flat",
        }

    # Check if all scores are the same
    if scores.nunique() == 1:
        # All scores are the same - return flat forecast
        avg_score = scores.iloc[0]
        return {
            "dates": [
                (dates.max() + timedelta(days=i + 1)).date() for i in range(days_ahead)
            ],
            "forecast": [avg_score] * days_ahead,
            "lower_bound": [avg_score - 5] * days_ahead,
            "upper_bound": [avg_score + 5] * days_ahead,
            "method": "flat",
        }

    # Convert dates to numeric for regression
    date_nums = [(d - dates.min()).days for d in dates]

    # Check if date_nums are all the same (defensive)
    if len(set(date_nums)) == 1:
        avg_score = scores.mean()
        return {
            "dates": [
                (dates.max() + timedelta(days=i + 1)).date() for i in range(days_ahead)
            ],
            "forecast": [avg_score] * days_ahead,
            "lower_bound": [avg_score - 5] * days_ahead,
            "upper_bound": [avg_score + 5] * days_ahead,
            "method": "flat",
        }

    try:
        # Linear regression
        coeffs = np.polyfit(date_nums, scores, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Predict future dates
        last_date = dates.max()
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(days_ahead)]
        forecast_nums = [(d - dates.min()).days for d in forecast_dates]
        forecast_scores = [slope * n + intercept for n in forecast_nums]

        # Calculate confidence interval (simple std-based)
        residuals = scores - (slope * np.array(date_nums) + intercept)
        std_error = np.std(residuals)

        return {
            "dates": [d.date() for d in forecast_dates],
            "forecast": forecast_scores,
            "lower_bound": [f - 1.96 * std_error for f in forecast_scores],
            "upper_bound": [f + 1.96 * std_error for f in forecast_scores],
            "method": "linear",
        }
    except (LinAlgError, ValueError, np.linalg.LinAlgError) as e:
        # Handle numerical errors gracefully (SVD convergence issues, etc.)
        logger.warning(f"Could not calculate forecast: {e}, returning flat forecast")
        avg_score = scores.mean()
        return {
            "dates": [
                (dates.max() + timedelta(days=i + 1)).date() for i in range(days_ahead)
            ],
            "forecast": [avg_score] * days_ahead,
            "lower_bound": [avg_score - 5] * days_ahead,
            "upper_bound": [avg_score + 5] * days_ahead,
            "method": "flat_fallback",
        }


def identify_at_risk_agents(df, threshold=70, lookback_days=14):
    """Identify agents at risk of dropping below threshold.

    Args:
        df: DataFrame with agent performance data
        threshold: QA score threshold
        lookback_days: Number of days to analyze for trend

    Returns:
        List of dictionaries with at-risk agent information
    """
    from datetime import timedelta

    if (
        "Call Date" not in df.columns
        or "Agent" not in df.columns
        or "QA Score" not in df.columns
    ):
        return []

    cutoff_date = df["Call Date"].max() - timedelta(days=lookback_days)
    recent_data = df[df["Call Date"] >= cutoff_date]

    if recent_data.empty:
        return []

    at_risk = []

    for agent in recent_data["Agent"].unique():
        agent_data = recent_data[recent_data["Agent"] == agent].sort_values("Call Date")

        if len(agent_data) < 3:  # Need at least 3 data points
            continue

        # Calculate metrics
        recent_avg = agent_data["QA Score"].mean()
        trend_slope = calculate_trend_slope(
            agent_data["Call Date"], agent_data["QA Score"]
        )
        volatility = agent_data["QA Score"].std()
        proximity_to_threshold = threshold - recent_avg

        # Calculate risk score (0-100)
        # Made less sensitive: requires stronger signals or multiple risk factors
        risk_score = 0

        # Trend component (0-40 points) - stricter thresholds
        if trend_slope < -2:  # Strong declining trend (>2 points/day)
            risk_score += 40
        elif trend_slope < -1.5:  # Moderate-strong decline
            risk_score += 25
        elif trend_slope < -1:  # Moderate decline
            risk_score += 15

        # Volatility component (0-30 points) - higher thresholds
        if volatility > 20:  # Very high volatility
            risk_score += 30
        elif volatility > 15:  # High volatility
            risk_score += 18

        # Proximity component (0-30 points) - stricter thresholds
        if proximity_to_threshold <= 3:  # Very close to threshold (<3 points away)
            risk_score += 30
        elif proximity_to_threshold <= 5:  # Close to threshold
            risk_score += 18
        elif proximity_to_threshold <= 8:  # Moderately close
            risk_score += 10

        # Require higher risk score AND at least one strong signal
        # This prevents flagging agents with only moderate risk factors
        has_strong_trend = trend_slope < -1.5
        has_high_volatility = volatility > 15
        is_very_close = proximity_to_threshold <= 5

        if risk_score >= 65 and (
            has_strong_trend or has_high_volatility or is_very_close
        ):  # Higher threshold + requires strong signal
            at_risk.append(
                {
                    "agent": agent,
                    "risk_score": risk_score,
                    "recent_avg": recent_avg,
                    "trend_slope": trend_slope,
                    "volatility": volatility,
                    "proximity_to_threshold": proximity_to_threshold,
                    "recent_calls": len(agent_data),
                }
            )

    # Sort by risk score
    at_risk.sort(key=lambda x: x["risk_score"], reverse=True)
    return at_risk


def calculate_trend_slope(dates, scores):
    """Calculate linear trend slope.

    Args:
        dates: Series of dates
        scores: Series of scores

    Returns:
        Slope value (positive = improving, negative = declining)
    """
    import numpy as np
    from numpy.linalg import LinAlgError

    if len(dates) < 2:
        return 0

    # Check if all dates are the same (would cause date_nums to all be 0)
    if dates.nunique() == 1:
        return 0  # No trend if all dates are the same

    # Check if all scores are the same (no variation)
    if scores.nunique() == 1:
        return 0  # No trend if all scores are the same

    try:
        date_nums = [(d - dates.min()).days for d in dates]

        # Check if date_nums are all the same (shouldn't happen after above check, but defensive)
        if len(set(date_nums)) == 1:
            return 0

        coeffs = np.polyfit(date_nums, scores, 1)
        return coeffs[0]
    except (LinAlgError, ValueError, np.linalg.LinAlgError) as e:
        # Handle numerical errors gracefully (SVD convergence issues, etc.)
        logger.debug(f"Could not calculate trend slope: {e}, returning 0")
        return 0


def classify_trajectory(df, agent=None):
    """Classify agent trajectory as improving, declining, stable, or volatile.

    Args:
        df: DataFrame with performance data
        agent: Optional agent ID to filter

    Returns:
        Dictionary with trajectory classification
    """
    if agent:
        df = df[df["Agent"] == agent]

    if "Call Date" not in df.columns or "QA Score" not in df.columns or len(df) < 3:
        return {"trajectory": "insufficient_data", "slope": 0, "volatility": 0}

    df_sorted = df.sort_values("Call Date")
    dates = df_sorted["Call Date"]
    scores = df_sorted["QA Score"]

    slope = calculate_trend_slope(dates, scores)
    volatility = scores.std()

    # Classify trajectory
    if volatility > 15:
        trajectory = "volatile"
    elif slope > 0.5:
        trajectory = "improving"
    elif slope < -0.5:
        trajectory = "declining"
    else:
        trajectory = "stable"

    # Projected score if trend continues
    last_score = scores.iloc[-1]
    projected_score = last_score + (slope * 7)  # Project 7 days ahead

    return {
        "trajectory": trajectory,
        "slope": slope,
        "volatility": volatility,
        "current_score": last_score,
        "projected_score": projected_score,
    }


def load_new_calls_only():
    """
    Smart refresh: Only loads CSV files that haven't been processed yet (PDFs are ignored).
    Returns tuple: (new_call_data_list, error_message, count_of_new_files)
    """
    try:
        # OPTIMIZATION: Load cache ONCE at start and reuse throughout refresh
        # CRITICAL FIX: Check if S3 cache is newer than local cache before using stale data
        logger.debug(" Loading cache once at start of refresh (will reuse throughout)")

        # Check S3 cache timestamp to see if it's newer than local/session cache
        s3_cache_newer = False
        try:
            s3_bucket_name = st.secrets["s3"]["bucket_name"]
            s3_cache_client = boto3.client(
                "s3",
                aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
                aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
                region_name=st.secrets["s3"].get("region_name", "us-east-1"),
            )
            # Use head_object for faster check (just metadata, no body download)
            response = s3_cache_client.head_object(
                Bucket=s3_bucket_name, Key=S3_CACHE_KEY
            )
            s3_last_modified = response.get("LastModified")

            # Compare with session state timestamp
            local_timestamp = st.session_state.get("_s3_cache_timestamp")
            if local_timestamp and s3_last_modified:
                # Convert local timestamp to datetime for comparison
                try:
                    if isinstance(local_timestamp, str):
                        # Parse ISO format string
                        local_dt = datetime.fromisoformat(
                            local_timestamp.replace("Z", "+00:00")
                        )
                    elif isinstance(local_timestamp, (int, float)):
                        # Unix timestamp
                        local_dt = datetime.fromtimestamp(local_timestamp)
                    else:
                        local_dt = local_timestamp

                    # Compare timestamps (S3 LastModified is timezone-aware, ensure local_dt is too)
                    s3_dt = s3_last_modified

                    # Make local_dt timezone-aware if S3 timestamp is timezone-aware
                    comparison_done = False
                    if local_dt.tzinfo is None and s3_dt.tzinfo is not None:
                        # Make local_dt timezone-aware for comparison (use S3's timezone)
                        try:
                            local_dt = local_dt.replace(tzinfo=s3_dt.tzinfo)
                            # Now both are timezone-aware, can compare directly below
                        except Exception:
                            # Fallback: convert both to timestamps for comparison
                            try:
                                s3_ts = s3_dt.timestamp()
                                local_ts = local_dt.timestamp()
                                if s3_ts > local_ts:
                                    s3_cache_newer = True
                                    logger.debug(
                                        f" S3 cache is newer (timestamp comparison: {s3_ts} > {local_ts}) - clearing stale cache"
                                    )
                                comparison_done = True
                            except Exception:
                                # If timestamp conversion fails, assume S3 is newer to be safe
                                s3_cache_newer = True
                                logger.debug(
                                    " Could not compare timestamps, assuming S3 is newer"
                                )
                                comparison_done = True

                    # Compare directly if we haven't already done the comparison
                    if not comparison_done and s3_dt > local_dt:
                        s3_cache_newer = True
                        logger.debug(
                            f" S3 cache is newer ({s3_dt}) than local ({local_dt}) - clearing stale cache"
                        )
                except Exception as compare_error:
                    logger.debug(
                        f" Could not compare timestamps: {compare_error}, assuming S3 is newer"
                    )
                    s3_cache_newer = True
            elif not local_timestamp:
                # No local timestamp, assume S3 might be newer (first load or cache cleared)
                s3_cache_newer = True
                logger.debug(" No local timestamp found - will check S3 cache")
        except ClientError as s3_error:
            error_code = s3_error.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                # S3 cache doesn't exist, use local cache
                logger.debug(" S3 cache does not exist, using local cache")
            else:
                logger.debug(
                    f" Could not check S3 cache timestamp: {s3_error}, will use existing cache"
                )
        except Exception as s3_check_error:
            logger.debug(
                f" Could not check S3 cache timestamp: {s3_check_error}, will use existing cache"
            )

        # Clear session state cache only if S3 is newer
        if s3_cache_newer:
            if "_s3_cache_result" in st.session_state:
                logger.debug(" Clearing stale session cache to force fresh S3 load")
                del st.session_state["_s3_cache_result"]
            if "_s3_cache_timestamp" in st.session_state:
                del st.session_state["_s3_cache_timestamp"]

        # Load from cache (will use S3 if session state was cleared, otherwise uses existing cache)
        disk_result = load_cached_data_from_disk()
        # CRITICAL FIX: Check if disk_result is None before accessing its elements
        existing_calls = (
            disk_result[0] if (disk_result and disk_result[0] is not None) else []
        )

        # Extract existing_cache_keys once for duplicate checking
        # Use consistent key extraction logic that matches cache check logic
        existing_cache_keys = set()
        if existing_calls:
            existing_cache_keys = {
                key
                for call in existing_calls
                if (key := extract_cache_key(call)) is not None
            }
            logger.debug(
                f" Built existing_cache_keys: {len(existing_cache_keys)} keys from {len(existing_calls)} calls"
            )

        # Get last_save_time from cache metadata (if available)
        # CRITICAL FIX: Read from S3 cache (shared source of truth) first, then fall back to local
        last_save_time = 0

        # First, try to get last_save_time from S3 cache (shared across all app instances)
        try:
            s3_bucket_name = st.secrets["s3"]["bucket_name"]
            s3_cache_client = boto3.client(
                "s3",
                aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
                aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
                region_name=st.secrets["s3"].get("region_name", "us-east-1"),
            )
            response = s3_cache_client.get_object(
                Bucket=s3_bucket_name, Key=S3_CACHE_KEY
            )
            cached_data = json.loads(response["Body"].read().decode("utf-8"))
            if isinstance(cached_data, dict):
                last_save_time = cached_data.get("last_save_time", 0)
                logger.debug(f" Read last_save_time from S3 cache: {last_save_time}")
        except Exception as s3_error:
            logger.debug(f" Could not read last_save_time from S3 cache: {s3_error}")
            # Fall back to local cache below
            pass

        # Fallback: Read from local disk cache if S3 read failed
        if not last_save_time and existing_calls and CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if isinstance(cached_data, dict):
                    last_save_time = cached_data.get("last_save_time", 0)
                    logger.debug(
                        f" Read last_save_time from local cache: {last_save_time}"
                    )
            except Exception:
                pass

        # Also check session state as fallback for last_save_time
        if not last_save_time:
            last_save_time = getattr(st.session_state, "_last_incremental_save_time", 0)

        # Get already processed keys from disk cache (survives restarts)
        # Also check session state as a fallback
        processed_keys = set()

        if existing_calls:
            cached_calls = existing_calls
            logger.debug(
                f" Loaded {len(cached_calls)} calls from disk cache - extracting S3 keys..."
            )

            # Extract processed file keys from cached calls
            # NOTE: processed_keys is used for FILE-level checking (which CSV files have been processed)
            # For call-level duplicate checking, we use existing_cache_keys (which uses extract_cache_key)
            keys_found = 0
            keys_missing = 0
            csv_files = 0
            pdf_files = 0
            sample_keys = []
            for call in cached_calls:
                # Extract file key (filename for CSV, _s3_key for PDF)
                filename = call.get("filename", "")
                _s3_key = call.get("_s3_key", "")
                _id = call.get("_id", "")

                # Determine if this is a CSV file
                is_csv = (
                    ":" in str(_id)
                    or ":" in str(_s3_key)
                    or (filename and filename.lower().endswith(".csv"))
                )

                if is_csv and filename:
                    # For CSV files, use filename as the file key
                    file_key = filename.strip("/")
                    processed_keys.add(file_key)
                    keys_found += 1
                    csv_files += 1
                    if len(sample_keys) < 5:
                        sample_keys.append(file_key)
                elif _s3_key:
                    # For PDF files, use _s3_key as the file key
                    file_key = _s3_key.strip("/")
                    processed_keys.add(file_key)
                    keys_found += 1
                    pdf_files += 1
                    if len(sample_keys) < 5:
                        sample_keys.append(file_key)
                else:
                    keys_missing += 1

            logger.debug(
                f" File Key Extraction: {keys_found} file keys found ({csv_files} CSV files, {pdf_files} PDF files), {keys_missing} missing"
            )
            if sample_keys:
                logger.debug(f" Sample cached keys (first 5): {sample_keys[:5]}")
            if keys_missing > 0:
                logger.warning(
                    f" {keys_missing} cached calls are missing _s3_key - they may be reprocessed"
                )
            else:
                logger.info(
                    " No disk cache found in count_new_csvs - all files will be treated as new"
                )

        # Also check session state (for files processed in current session)
        session_keys = st.session_state.get("processed_s3_keys", set())
        if session_keys:
            processed_keys.update(session_keys)
            logger.info(f" Found {len(session_keys)} additional files in session state")

        logger.info(
            f" Total {len(processed_keys)} files already processed - will skip these"
        )

        # Configure S3 client
        import botocore.config

        config = botocore.config.Config(
            connect_timeout=10, read_timeout=30, retries={"max_attempts": 2}
        )
        s3_client_with_timeout = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
            region_name=st.secrets["s3"].get("region_name", "us-east-1"),
            config=config,
        )

        # List all CSV files in S3 (PDFs are ignored)
        paginator = s3_client_with_timeout.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=s3_bucket_name, Prefix=s3_prefix, MaxKeys=1000
        )

        # Find new CSV files (not in processed_keys)
        new_csv_keys = []
        processed_keys_normalized = {key.strip("/") for key in processed_keys}
        total_s3_files = 0
        sample_s3_keys = []
        all_s3_keys = []  # Store all S3 keys for comparison

        # Track pagination progress
        page_count = 0
        files_per_page = []
        is_truncated = False

        for page in pages:
            page_count += 1
            page_file_count = 0

            if isinstance(page, dict) and "Contents" in page:
                for obj in page["Contents"]:
                    key_raw = obj.get("Key")
                    if not key_raw:
                        continue
                    key = key_raw.strip(
                        "/"
                    )  # Normalize S3 key (consistent with cache normalization)
                    if key.lower().endswith(".csv"):
                        total_s3_files += 1
                        page_file_count += 1
                        all_s3_keys.append(key)
                        if len(sample_s3_keys) < 10:
                            sample_s3_keys.append(key)
                        if key not in processed_keys_normalized:
                            last_modified = obj.get("LastModified", datetime.min)
                            # Ensure LastModified is a datetime object for safe sorting
                            if not isinstance(last_modified, datetime):
                                last_modified = datetime.min
                            new_csv_keys.append(
                                {
                                    "key": key,  # Store normalized key
                                    "last_modified": last_modified,
                                }
                            )

            files_per_page.append(page_file_count)
            # Check if pagination is truncated (though paginator should handle this automatically)
            if "IsTruncated" in page:
                is_truncated = page["IsTruncated"]
                # Log IsTruncated status every 10 pages or if truncated
                if page_count % 10 == 0 or is_truncated:
                    logger.info(
                        f" Page {page_count}: IsTruncated={is_truncated}, files_in_page={page_file_count}"
                    )

            # Log pagination progress every 10 pages
            if page_count % 10 == 0:
                logger.info(
                    f" Processing page {page_count}, found {total_s3_files} CSV files so far..."
                )

        # Log pagination completion with final IsTruncated status (combined)
        logger.info(
            f" Pagination complete: {page_count} pages, {total_s3_files} files, IsTruncated={is_truncated} (False=complete, True=may be incomplete)"
        )
        if files_per_page:
            logger.debug(
                f" Files per page: min={min(files_per_page)}, max={max(files_per_page)}, last_page={files_per_page[-1]}"
            )

        # Verify pagination completed (warn if suspicious)
        if total_s3_files > 0 and total_s3_files % 1000 == 0:
            logger.warning(
                f" Total files ({total_s3_files}) is exactly divisible by 1000 - pagination might be incomplete!"
            )
        if is_truncated:
            logger.error(
                "CRITICAL: Last page had IsTruncated=True! Pagination may be incomplete - expected more pages!"
            )

        # Additional check: If cache has files and S3 listing found same count, but user expects more files,
        # this might indicate files are in a different location or being filtered
        # Check if last page was full (1000 files) - if so, there might be more pages even if IsTruncated=False
        if files_per_page and len(files_per_page) > 0:
            last_page_count = files_per_page[-1]
            if last_page_count == 1000 and not is_truncated:
                logger.warning(
                    "WARNING: Last page had exactly 1000 files but IsTruncated=False!"
                )
                logger.warning(
                    "This is unusual - typically a full page (1000 files) means more pages exist."
                )
                logger.warning(
                    "If you know there are more files in S3, they may be in a different prefix/folder."
                )

        # Exhaustive key comparison - compare ALL cache keys against ALL S3 keys
        # Count actual matches (exhaustive comparison)
        actual_matches = len([k for k in all_s3_keys if k in processed_keys_normalized])

        # Count S3 keys NOT in cache (should equal new files)
        s3_keys_not_in_cache = len(
            [k for k in all_s3_keys if k not in processed_keys_normalized]
        )

        # Count cache keys NOT in S3 (orphaned cache entries)
        cache_keys_not_in_s3 = len(
            [k for k in processed_keys_normalized if k not in all_s3_keys]
        )

        match_rate = (
            (actual_matches / total_s3_files * 100) if total_s3_files > 0 else 0
        )
        cache_hit_rate = (
            (actual_matches / len(processed_keys_normalized) * 100)
            if len(processed_keys_normalized) > 0
            else 0
        )

        # Condensed key comparison results
        logger.info(
            f" Key Comparison: S3={total_s3_files}, Cache={len(processed_keys_normalized)}, Matches={actual_matches}, New={s3_keys_not_in_cache}, Orphaned={cache_keys_not_in_s3}, MatchRate={match_rate:.1f}%, HitRate={cache_hit_rate:.1f}%"
        )
        logger.info(f"   - New files to process: {len(new_csv_keys)}")

        # Validate that new file count matches expected
        if s3_keys_not_in_cache != len(new_csv_keys):
            logger.warning(
                f" WARNING: Mismatch! S3 keys not in cache ({s3_keys_not_in_cache}) != new_csv_keys count ({len(new_csv_keys)})"
            )

        if sample_s3_keys:
            logger.info(f" Sample S3 keys (first 10): {sample_s3_keys[:10]}")

        # Diagnostic warnings for key matching issues
        if len(processed_keys_normalized) > 0 and cache_hit_rate < 90:
            logger.warning(
                f" Cache hit rate is only {cache_hit_rate:.1f}% - some cached keys not found in S3"
            )
            logger.warning(
                "This could indicate key normalization issues or orphaned cache entries"
            )
            if cache_keys_not_in_s3 > 0:
                logger.warning(
                    f" Found {cache_keys_not_in_s3} orphaned cache keys (not in S3)"
                )

        # Warn if all files are being treated as new when cache exists
        if (
            len(new_csv_keys) == total_s3_files
            and len(processed_keys_normalized) > 0
            and total_s3_files > 0
        ):
            logger.error(
                f" ERROR: All {total_s3_files} files are being treated as new, but {len(processed_keys_normalized)} are cached!"
            )
            logger.error(
                f" Cache hit rate: {cache_hit_rate:.1f}% - Key extraction or normalization may have failed!"
            )
            logger.error(f" Sample cached keys: {list(processed_keys_normalized)[:5]}")
            logger.error(f" Sample S3 keys: {all_s3_keys[:5]}")

        # Validate S3 listing completeness before early exit
        # IMPORTANT: If cache count matches S3 count exactly, but user knows there are more files in S3,
        # this indicates S3 listing is incomplete (files might be in different prefix/folder or filtered out)
        if len(processed_keys_normalized) > total_s3_files:
            logger.error(
                f" CRITICAL: Cache has MORE keys ({len(processed_keys_normalized)}) than S3 ({total_s3_files})!"
            )
            logger.error(
                "This suggests S3 listing is incomplete or cache has orphaned entries."
            )
            logger.error(
                f" Difference: {len(processed_keys_normalized) - total_s3_files} keys"
            )
            logger.error(f" S3 prefix used: '{s3_prefix}' (empty means root of bucket)")
            logger.error(
                "If you know there are more files, they may be in a different prefix/folder."
            )
            # Don't exit early - S3 listing appears incomplete, continue to process
        elif len(processed_keys_normalized) == total_s3_files and total_s3_files > 0:
            # Cache count matches S3 count - verify all are actually matched
            # WARNING: This doesn't mean S3 listing is complete - files might be in different locations
            if actual_matches == total_s3_files and not new_csv_keys:
                logger.info(
                    f" All {total_s3_files} S3 files (prefix '{s3_prefix}') are in cache and verified. Note: More files may exist in other prefixes."
                )
                return (
                    [],
                    None,
                    0,
                )  # No new files - verified complete match for current prefix
            elif actual_matches == total_s3_files and len(new_csv_keys) > 0:
                logger.warning(
                    f"WARNING: All keys match, but {len(new_csv_keys)} files marked as new!"
                )
                logger.warning(
                    "This indicates a logic error - proceeding to process new files"
                )
            else:
                logger.warning(
                    f"Cache count ({len(processed_keys_normalized)}) matches S3 count ({total_s3_files}), but only {actual_matches} keys match!"
                )
                logger.warning(
                    "This suggests key normalization issues - proceeding to process new files"
                )
        elif not new_csv_keys and len(processed_keys_normalized) < total_s3_files:
            # S3 has more files than cache, but no new files found - this shouldn't happen
            logger.warning(
                f" WARNING: S3 has {total_s3_files} files, cache has {len(processed_keys_normalized)}, but no new files found!"
            )
            logger.warning("This suggests a comparison issue - proceeding anyway")

        # Final early exit check (only if validation passed above)
        if (
            not new_csv_keys
            and len(processed_keys_normalized) <= total_s3_files
            and actual_matches == total_s3_files
        ):
            logger.info(
                " No new files found - all files are already processed and verified"
            )
            return [], None, 0  # No new files

        # Additional check: if processed count matches total, all should be cached
        if len(processed_keys_normalized) >= total_s3_files and total_s3_files > 0:
            logger.warning(
                f"WARNING: Cached keys ({len(processed_keys_normalized)}) >= Total S3 files ({total_s3_files})"
            )
            logger.warning(
                f" But {len(new_csv_keys)} files are marked as new. This indicates a key format mismatch."
            )
            logger.warning(
                f" Match rate: {match_rate:.1f}% - Keys are not matching correctly."
            )

        # Sort by modification date (most recent first)
        new_csv_keys.sort(key=lambda x: x["last_modified"], reverse=True)

        # Process new CSV files
        new_calls = []
        errors = []

        def process_csv(key_item):
            """Process a single CSV file: download, parse rows, and return results.

            Args:
                key_item: Either a string (normalized key) or dict with 'key' and 'last_modified'
            """
            try:
                # Handle both string and dict formats
                if isinstance(key_item, dict):
                    normalized_key = key_item["key"]
                else:
                    normalized_key = key_item

                # Use normalized key for S3
                s3_key_to_use = normalized_key
                if s3_prefix and not normalized_key.startswith(s3_prefix):
                    s3_key_to_use = (
                        f"{s3_prefix.rstrip('/')}/{normalized_key}"
                        if not normalized_key.startswith(s3_prefix)
                        else normalized_key
                    )

                # Download CSV from S3
                response = s3_client_with_timeout.get_object(
                    Bucket=s3_bucket_name, Key=s3_key_to_use
                )
                csv_content = response["Body"].read().decode("utf-8")

                # Read CSV into DataFrame
                csv_file = io.StringIO(csv_content)
                df = pd.read_csv(csv_file)

                # Extract filename from key
                filename = normalized_key.split("/")[-1]

                # Process each row
                csv_calls = []
                for idx, row in df.iterrows():
                    try:
                        parsed_data = parse_csv_row(row, filename)
                        if parsed_data:
                            # _id and _s3_key are already set in parse_csv_row based on call_id
                            # No need to override them here - each row should have unique _id
                            csv_calls.append(parsed_data)
                    except Exception as e:
                        error_msg = (
                            f"Error parsing row {idx + 1} in {filename}: {str(e)}"
                        )
                        errors.append(error_msg)
                        logger.warning(error_msg)
                        continue

                return csv_calls, None
            except Exception as e:
                logger.warning(f" Failed to process CSV '{normalized_key}': {e}")
                return [], f"{normalized_key}: {str(e)}"

        # Process in smaller batches to reduce lock contention with large cache files
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        BATCH_SIZE = 25  # Reduced from 50 to lower memory per batch
        DASHBOARD_UPDATE_INTERVAL = 500
        MAX_FILES_PER_REFRESH = 500  # Reduced from 1000 to process in smaller chunks and reduce memory pressure

        # Limit the number of files processed per refresh
        total_new_unlimited = len(new_csv_keys)
        new_csv_keys = new_csv_keys[:MAX_FILES_PER_REFRESH]
        total_new = len(new_csv_keys)

        if total_new_unlimited > MAX_FILES_PER_REFRESH:
            remaining = total_new_unlimited - MAX_FILES_PER_REFRESH
            logger.info(
                f" Refresh New Data: Found {total_new_unlimited} new CSV files total, processing {total_new} this refresh (limit: {MAX_FILES_PER_REFRESH}), {remaining} remaining"
            )
        else:
            logger.info(
                f" Refresh New Data: Found {total_new} new CSV files to process"
            )

        logger.info(
            f" Starting to process {total_new} new CSV files in batches of {BATCH_SIZE} (out of {total_s3_files} total in S3)"
        )

        # Memory monitoring - check initial memory usage
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_percent = process.memory_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_percent > 80:
                logger.warning(
                    f" High initial memory usage: {memory_percent:.1f}% ({memory_mb:.0f} MB) - processing may be slower"
                )
            else:
                logger.info(
                    f" Initial memory usage: {memory_percent:.1f}% ({memory_mb:.0f} MB)"
                )
        except ImportError:
            logger.debug("psutil not available for memory monitoring (optional)")
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")

        processing_start_time = time.time()
        processed_count = 0
        last_dashboard_update = 0
        last_incremental_save_time = time.time()
        batches_since_save = 0
        SAVE_INTERVAL_BATCHES = (
            2  # Save every 2 batches (more frequent to reduce memory accumulation)
        )
        SAVE_INTERVAL_SECONDS = 90  # Or every 90 seconds, whichever comes first

        # Note: last_save_time already loaded at start of function (reused here)

        # Process in batches
        for batch_start in range(0, total_new, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_new)
            batch_keys = new_csv_keys[batch_start:batch_end]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (total_new + BATCH_SIZE - 1) // BATCH_SIZE

            if batch_num % 5 == 0 or batch_num == 1 or batch_num == total_batches:
                logger.info(
                    f" Processing batch {batch_num}/{total_batches}: files {batch_start + 1}-{batch_end} of {total_new}"
                )

            # Track calls from this batch only (for incremental save)
            batch_calls = []

            # OPTIMIZATION: Reuse existing_cache_keys loaded at start (no need to reload from disk)
            # existing_cache_keys already loaded at function start

            with ThreadPoolExecutor(
                max_workers=5
            ) as executor:  # Reduced from 10 to lower memory pressure
                future_to_key = {
                    executor.submit(process_csv, item): item["key"]
                    for item in batch_keys
                }

                for future in as_completed(future_to_key):
                    try:
                        csv_calls_list, error = future.result(
                            timeout=120
                        )  # 120 second timeout per CSV file (may contain many rows)
                    except Exception as e:
                        # Handle both TimeoutError and concurrent.futures.TimeoutError
                        from concurrent.futures import (
                            TimeoutError as FuturesTimeoutError,
                        )

                        if isinstance(e, (TimeoutError, FuturesTimeoutError)):
                            key = future_to_key.get(future, "Unknown")
                            logger.error(f" Timeout processing CSV {key}: {e}")
                            errors.append(f"{key}: Processing timeout (120s)")
                            processed_count += 1
                            continue
                        else:
                            # Unexpected error in future execution
                            logger.error(f" Unexpected error in future: {e}")
                            errors.append(f"Unknown: {str(e)}")
                            processed_count += 1
                            continue

                    processed_count += 1

                    if csv_calls_list:
                        # Extend with list of calls from CSV
                        duplicate_count = 0
                        for parsed_data in csv_calls_list:
                            # Check if this call is already in cache BEFORE adding
                            # Use consistent key extraction logic
                            call_key = extract_cache_key(parsed_data)

                            if call_key and call_key in existing_cache_keys:
                                # Already in cache - skip this call
                                duplicate_count += 1
                                continue

                            # Not in cache - add to new calls
                        new_calls.append(parsed_data)
                        batch_calls.append(parsed_data)  # Track for this batch

                        if duplicate_count > 0:
                            logger.debug(
                                f" Skipped {duplicate_count} duplicate calls from {len(csv_calls_list)} total in batch"
                            )
                    elif error:
                        errors.append(error)

                    # Log progress every 100 files (unconditionally for each processed file)
                    if processed_count % 100 == 0:
                        elapsed = time.time() - processing_start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        remaining = (
                            (total_new - processed_count) / rate if rate > 0 else 0
                        )
                        logger.info(
                            f" Refresh Progress: {processed_count}/{total_new} files processed ({processed_count * 100 // total_new if total_new > 0 else 0}%), {len(new_calls)} successful, {len(errors)} errors. Rate: {rate:.1f} files/sec, ETA: {remaining / 60:.1f} min"
                        )

                        # Periodic memory check every 100 files
                        try:
                            import psutil
                            import os

                            process = psutil.Process(os.getpid())
                            memory_percent = process.memory_percent()
                            memory_mb = process.memory_info().rss / 1024 / 1024
                            if memory_percent > 85:
                                logger.warning(
                                    f" High memory usage: {memory_percent:.1f}% ({memory_mb:.0f} MB) - consider reducing batch size if issues occur"
                                )
                        except Exception:
                            pass  # Ignore memory check errors

            # Update existing_calls with batch data (for next batch)
            # OPTIMIZATION: Use extend() instead of concatenation for better performance
            # This modifies the list in-place (O(k)) instead of creating a new list (O(n))
            existing_calls.extend(batch_calls)
            # Update existing_cache_keys to include new batch keys
            # Use consistent key extraction logic
            batch_keys_set = {
                key
                for call in batch_calls
                if (key := extract_cache_key(call)) is not None
            }
            existing_cache_keys.update(batch_keys_set)

            # Periodic memory cleanup every 5 batches
            if batch_num % 5 == 0:
                import gc

                gc.collect()

            batches_since_save += 1
            time_since_save = time.time() - last_incremental_save_time
            should_save = (
                (batches_since_save >= SAVE_INTERVAL_BATCHES)
                or (time_since_save >= SAVE_INTERVAL_SECONDS)
                or (batch_num == total_batches)
            )

            # OPTIMIZATION: Save incrementally every 3 batches or 2 minutes (instead of every batch)
            if should_save:
                try:
                    # existing_calls already contains all calls including current batch (updated above)
                    # Deduplication will happen in save_cached_data_to_disk() - no need to do it here
                    calls_to_save = existing_calls

                    # CRITICAL FIX: Add retry logic for save failures to prevent silent data loss
                    # Retry up to 3 times with exponential backoff
                    max_save_retries = 3
                    save_success = False
                    save_error = None

                    for save_attempt in range(max_save_retries):
                        try:
                            # Use atomic write with locking via save_cached_data_to_disk
                            save_cached_data_to_disk(
                                calls_to_save,
                                errors.copy(),
                                partial=True,
                                processed=processed_count,
                                total=total_new,
                            )
                            save_success = True
                            break  # Success, exit retry loop
                        except Exception as save_e:
                            save_error = save_e
                            if save_attempt < max_save_retries - 1:
                                wait_time = 0.5 * (
                                    save_attempt + 1
                                )  # Exponential backoff: 0.5s, 1s, 1.5s
                                logger.warning(
                                    f" Incremental save attempt {save_attempt + 1}/{max_save_retries} failed: {save_e}, retrying in {wait_time}s..."
                                )
                                time.sleep(wait_time)
                            else:
                                logger.error(
                                    f" CRITICAL: Failed to save incremental cache after {max_save_retries} attempts: {save_e}"
                                )
                                logger.error(
                                    f" Data loss risk: {len(calls_to_save)} calls not saved to disk"
                                )

                    if save_success:
                        last_incremental_save_time = time.time()
                        st.session_state._last_incremental_save_time = (
                            last_incremental_save_time
                        )
                        batches_since_save = 0  # Reset counter
                        logger.info(
                            f" Incremental save: Saved {len(calls_to_save)} calls to disk cache ({processed_count}/{total_new} = {processed_count * 100 // total_new if total_new > 0 else 0}% complete)"
                        )

                        # SAFE MEMORY OPTIMIZATION: Clear old batches after successful save, keep last 2 saves as buffer
                        # This reduces memory while maintaining safety buffer
                        batches_to_keep = (
                            SAVE_INTERVAL_BATCHES * 2
                        )  # Keep last 2 saves worth
                        calls_per_save = batches_to_keep * BATCH_SIZE

                        # Only clear if we have significantly more calls than we want to keep
                        if (
                            len(existing_calls) > calls_per_save * 1.5
                        ):  # Only clear if 50% more than buffer
                            calls_to_keep = existing_calls[-calls_per_save:]
                            cleared_count = len(existing_calls) - len(calls_to_keep)
                            existing_calls = calls_to_keep

                            # Update existing_cache_keys to match
                            # Use consistent key extraction logic
                            existing_cache_keys = {
                                key
                                for call in existing_calls
                                if (key := extract_cache_key(call)) is not None
                            }

                            logger.info(
                                f" Cleared {cleared_count} old calls from memory (kept {len(existing_calls)} as safety buffer)"
                            )

                            # Force garbage collection after clearing
                            import gc

                            gc.collect()

                        # Clear intermediate variables after successful save
                        del calls_to_save
                        import gc

                        gc.collect()
                    else:
                        # All retries failed - log critical error but continue processing
                        # The next save attempt might succeed, and we don't want to lose all progress
                        logger.error(
                            f" CRITICAL: All incremental save retries failed. Last error: {save_error}"
                        )
                        logger.error(
                            "Processing will continue, but data may be lost if app crashes"
                        )
                except Exception as e:
                    logger.error(f" Unexpected error during incremental save: {e}")
                    import traceback

                    logger.error(traceback.format_exc())

            # Update dashboard every 500 calls
            if processed_count >= last_dashboard_update + DASHBOARD_UPDATE_INTERVAL:
                logger.info(
                    f" Dashboard update: {processed_count} calls processed, updating progress..."
                )
                last_dashboard_update = processed_count

                # Update session state with progress (no rerun to prevent cache corruption)
                # Progress will be visible on next natural rerun (user interaction or completion)
                st.session_state["last_refresh_progress_update"] = {
                    "processed": processed_count,
                    "total": total_new,
                    "timestamp": time.time(),
                }

                # Note: Removed st.rerun() to prevent cache corruption from concurrent reads/writes
                # Progress updates will be visible when refresh completes or user interacts

        elapsed_total = time.time() - processing_start_time
        logger.info(
            f" Refresh completed: Processed {total_new} new files in {elapsed_total / 60:.1f} minutes. Success: {len(new_calls)}, Errors: {len(errors)}. Cache updated with {len(new_calls)} new calls."
        )

        return new_calls, errors if errors else None, len(new_calls)

    except Exception as e:
        return [], f"Error loading new calls: {e}", 0


credentials = st.secrets["credentials"].to_dict()
cookie = st.secrets["cookie"]
auto_hash = st.secrets.get("auto_hash", False)

# Initialize authenticator with retry logic for CookieManager loading issues
authenticator = None
max_retries = 7  # Increased retries for better reliability
retry_delay = 4  # Increased initial delay
skip_auth_after_failures = st.session_state.get("skip_auth_after_failures", False)

# Check if user has chosen to skip auth after persistent failures
if skip_auth_after_failures:
    logger.warning(
        "Skipping authentication due to persistent CookieManager failures (user choice)"
    )
    authenticator = None
else:
    for attempt in range(max_retries):
        try:
            # Add progressive delay before each attempt (except first)
            if attempt > 0:
                wait_before = retry_delay * (1.5 ** (attempt - 1))
                logger.info(
                    f"Waiting {wait_before:.1f}s before authentication attempt {attempt + 1}/{max_retries}"
                )
                time.sleep(min(wait_before, 10))  # Cap at 10 seconds

            authenticator = stauth.Authenticate(
                credentials,
                cookie["name"],
                cookie["key"],
                cookie["expiry_days"],
                auto_hash=auto_hash,
            )
            # Test if CookieManager actually loaded by checking if we can access it
            # This helps catch cases where initialization appears to succeed but component isn't ready
            try:
                # Longer delay to let component fully initialize
                time.sleep(1.0)
            except Exception:
                pass
            logger.info(
                f"Authentication component initialized successfully on attempt {attempt + 1}"
            )
            break  # Success, exit retry loop
        except Exception as e:
            error_msg = str(e).lower()
            is_component_error = (
                "cookiemanager" in error_msg
                or "component" in error_msg
                or "frontend" in error_msg
                or "extra_streamlit_components" in error_msg
                or "trouble loading" in error_msg
                or "cookie_manager" in error_msg
                or "cannot assemble" in error_msg
                or "cookie manager" in error_msg
                or "frontend assets" in error_msg
                or "network latency" in error_msg
                or "proxy settings" in error_msg
            )

            if is_component_error and attempt < max_retries - 1:
                # Retry with exponential backoff (longer delays)
                wait_time = retry_delay * (2**attempt)
                wait_time = min(wait_time, 15)  # Cap at 15 seconds
                logger.warning(
                    f"CookieManager initialization failed (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time}s... Error: {e}"
                )
                # Show a loading message to user
                if attempt == 0:
                    with st.spinner(
                        f"Loading authentication component... (attempt {attempt + 1}/{max_retries})"
                    ):
                        time.sleep(wait_time)
                else:
                    time.sleep(wait_time)
                continue
            else:
                # Final attempt failed or non-component error
                logger.error(
                    f"Failed to initialize authenticator after {attempt + 1} attempts: {e}"
                )
                if is_component_error:
                    # Will be handled in login section below
                    authenticator = None
                else:
                    st.error("**Authentication System Error**")
                    st.error(f"Failed to initialize authentication: {str(e)}")
                    logger.exception("Authentication initialization error")
                    st.stop()

# --- LOGIN GUARD ---
auth_status = st.session_state.get("authentication_status")

# If they've never submitted the form, show it
if auth_status is None:
    if authenticator is None:
        # CookieManager failed to initialize
        st.error(" **Authentication Component Loading Issue**")
        st.warning(
            "The authentication component is having trouble loading. This is usually a temporary network or CDN issue."
        )
        st.markdown("### 🔧 **Quick Fixes (try in order):**")
        st.markdown("1. **Wait 10-15 seconds** and refresh the page (F5 or Cmd+R)")
        st.markdown("2. **Hard refresh** the page:")
        st.code("Windows/Linux: Ctrl+Shift+R\nMac: Cmd+Shift+R", language=None)
        st.markdown("3. **Clear browser cache** and cookies for this site")
        st.markdown("4. **Try a different browser** or incognito/private mode")
        st.markdown(
            "5. **Check your network connection** - ensure you can access external CDNs"
        )
        st.markdown(
            "6. **Verify installation** - ensure `extra-streamlit-components` is installed:"
        )
        st.code("pip install --upgrade extra-streamlit-components", language="bash")
        st.markdown("---")
        st.info(
            " **If the issue persists:** This may be a network/proxy/CDN issue. Contact your administrator or check if your deployment environment allows access to Streamlit component CDNs."
        )
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "**🔄 Retry Authentication**", type="primary", width="stretch"
            ):
                # Clear any cached state and retry
                if "authenticator_retry_count" not in st.session_state:
                    st.session_state.authenticator_retry_count = 0
                st.session_state.authenticator_retry_count += 1
                # Clear skip flag if user wants to retry
                if "skip_auth_after_failures" in st.session_state:
                    del st.session_state["skip_auth_after_failures"]
                st.rerun()
        with col2:
            if st.button(
                "**⚠️ Continue Without Auth**",
                type="secondary",
                width="stretch",
                help="Skip authentication for this session (not recommended for production)",
            ):
                st.session_state["skip_auth_after_failures"] = True
                st.session_state["authentication_status"] = True  # Allow access
                st.session_state["username"] = "guest"
                st.rerun()
        st.stop()

    try:
        # Add a small delay before calling login to give component time to load
        time.sleep(0.5)
        authenticator.login("main", "Login")
    except Exception as e:
        # Handle CookieManager component loading issues gracefully
        error_msg = str(e).lower()
        is_component_error = (
            "cookiemanager" in error_msg
            or "component" in error_msg
            or "frontend" in error_msg
            or "extra_streamlit_components" in error_msg
            or "trouble loading" in error_msg
            or "cookie_manager" in error_msg
            or "cannot assemble" in error_msg
            or "cookie manager" in error_msg
        )

        if is_component_error:
            st.error(" **Authentication Component Loading Issue**")
            st.warning(
                "The authentication component (CookieManager) is having trouble loading its frontend assets. "
                "This is usually a temporary network or CDN issue."
            )
            st.markdown("### 🔧 **Quick Fixes (try in order):**")
            st.markdown("1. **Wait 15-30 seconds** and refresh the page (F5 or Cmd+R)")
            st.markdown("2. **Hard refresh** the page to clear cached assets:")
            st.code("Windows/Linux: Ctrl+Shift+R\nMac: Cmd+Shift+R", language=None)
            st.markdown("3. **Clear browser cache** and cookies for this site")
            st.markdown("4. **Try a different browser** or incognito/private mode")
            st.markdown(
                "5. **Check your network connection** - ensure you can access external CDNs (jsDelivr, unpkg, etc.)"
            )
            st.markdown(
                "6. **Verify installation** - ensure `extra-streamlit-components` is installed:"
            )
            st.code(
                "pip install --upgrade extra-streamlit-components streamlit-authenticator",
                language="bash",
            )
            st.markdown(
                "7. **Check firewall/proxy settings** - ensure CDN access is not blocked"
            )
            st.markdown("---")
            st.info(
                " **If the issue persists:** This may be a network/proxy/CDN issue. "
                "The CookieManager component loads JavaScript from external CDNs. "
                "Contact your administrator or check if your deployment environment allows access to Streamlit component CDNs."
            )
            logger.warning(f"CookieManager component loading issue: {e}")
            # Don't stop immediately - let user try to refresh
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("**🔄 Retry Now**", type="primary", width="stretch"):
                    # Clear session state to force re-initialization
                    if "skip_auth_after_failures" in st.session_state:
                        del st.session_state["skip_auth_after_failures"]
                    if "authenticator_retry_count" not in st.session_state:
                        st.session_state.authenticator_retry_count = 0
                    st.session_state.authenticator_retry_count += 1
                    st.rerun()
            with col2:
                if st.button("**🔄 Hard Refresh**", width="stretch"):
                    # Clear more aggressively
                    st.session_state.clear()
                    st.rerun()
            with col3:
                if st.button(
                    "**⚠️ Skip Auth**",
                    type="secondary",
                    width="stretch",
                    help="Skip authentication for this session (not recommended for production)",
                ):
                    st.session_state["skip_auth_after_failures"] = True
                    st.session_state["authentication_status"] = True  # Allow access
                    st.session_state["username"] = "guest"
                    st.rerun()
        else:
            st.error("**Authentication Error**")
            st.error(f"Error: {str(e)}")
            logger.exception("Authentication error")
        st.stop()
    st.stop()

# If they submitted bad creds, show error and stay on login
if auth_status is False:
    st.error(" Username or password is incorrect")
    st.stop()

# Get current user info
current_username = st.session_state.get("username")
current_name = st.session_state.get("name")

# Check if user is anonymous (needed early for view mode display)
is_anonymous_user = current_username and current_username.lower() == "anonymous"

# Session management - track activity and timeout
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()
else:
    st.session_state.last_activity = time.time()  # Update on each interaction

# Check for session timeout (30 minutes of inactivity)
SESSION_TIMEOUT_MINUTES = 30
if check_session_timeout(st.session_state.last_activity, SESSION_TIMEOUT_MINUTES):
    st.warning("Your session has expired due to inactivity. Please log in again.")
    st.session_state.authentication_status = None
    st.session_state.last_activity = 0
    st.rerun()

# Show session timeout warning (5 minutes before timeout)
time_remaining = SESSION_TIMEOUT_MINUTES - (
    (time.time() - st.session_state.last_activity) / 60
)
if 0 < time_remaining <= 5:
    st.sidebar.warning(f"Session expires in {int(time_remaining)} minute(s)")

# Audit logging (all users)
if current_username:
    log_audit_event(
        current_username,
        "page_access",
        f"Accessed dashboard at {datetime.now().isoformat()}",
    )

# Check if user is an agent (has agent_id mapping) or admin
user_agent_id = None


# Helper functions for clear admin distinction
def is_regular_admin():
    """Check if current user is a regular admin (can view all agent data, but not super admin features).

    Regular admins can:
    - View all agent data and analytics
    - Access dark mode
    - See all charts and reports

    Regular admins cannot:
    - Access refresh controls
    - Access system monitoring
    - Access data quality validation
    """
    if not current_username:
        return False

    # Try to get agent_id from secrets
    try:
        user_mapping = st.secrets.get("user_mapping", {})
        if current_username in user_mapping:
            agent_id_value = user_mapping[current_username].get("agent_id", "")
            # No agent_id means admin
            if not agent_id_value:
                return True
            else:
                # User has agent_id - they are an agent, not an admin
                return False
        elif current_username.lower() in ["chloe", "shannon", "jerson"]:
            # Super admins are also regular admins
            return True
        else:
            # No mapping found - default to admin view for now
            # You can add mappings in secrets.toml to restrict access
            return True
    except Exception:
        # If mapping doesn't exist, default to admin view
        return True


def is_super_admin():
    """Check if current user is a super admin (has access to refresh controls and monitoring features).

    Super admins can:
    - Everything regular admins can do, PLUS:
    - Refresh Data controls
    - System Monitoring & Metrics
    - Data Quality Validation

    Currently only: Chloe, Shannon, and Jerson
    """
    if not current_username:
        return False
    allowed_users = ["chloe", "shannon", "jerson"]
    return current_username.lower() in allowed_users


# Set user_agent_id and is_admin for backward compatibility
# Normalize agent ID immediately: extract first two digits after 'bpagent'
# This same normalized value is used everywhere (sidebar, heading, filtering)
try:
    user_mapping = st.secrets.get("user_mapping", {})
    if current_username and current_username in user_mapping:
        agent_id_value = user_mapping[current_username].get("agent_id", "")
        if agent_id_value:
            user_agent_id = normalize_agent_id(agent_id_value)
except Exception:
    pass

# Set is_admin for backward compatibility (but prefer using is_regular_admin() function)
is_admin = is_regular_admin()

st.sidebar.success(f"Welcome, {current_name} 👋")

# Show view mode
if is_anonymous_user:
    st.sidebar.info(" Anonymous View: De-identified Data")
elif user_agent_id:
    st.sidebar.info(f"Agent View: {user_agent_id}")
elif is_regular_admin():
    st.sidebar.info("Admin View: All Data")
else:
    st.sidebar.info("User View: All Data")

# Logout button
st.sidebar.markdown("---")
if st.sidebar.button("Logout", help="Log out of your account", type="secondary"):
    try:
        # Clear authentication state first
        st.session_state.authentication_status = None
        st.session_state.username = None
        st.session_state.name = None
        st.session_state.user_agent_id = None

        # Clear data-related session state to prevent issues
        if "reload_all_triggered" in st.session_state:
            del st.session_state["reload_all_triggered"]
        if "_s3_cache_result" in st.session_state:
            del st.session_state["_s3_cache_result"]
        if "_s3_cache_timestamp" in st.session_state:
            del st.session_state["_s3_cache_timestamp"]

        # Call authenticator logout (may fail if CookieManager has issues, but that's OK)
        try:
            authenticator.logout("Logout", "sidebar")
        except Exception as logout_error:
            logger.warning(
                f"Authenticator logout had an issue (non-critical): {logout_error}"
            )

        # Use rerun to refresh the page
        st.rerun()
    except Exception as e:
        logger.exception(f"Logout error: {e}")
        st.sidebar.error("Error logging out. Please refresh the page manually.")
        # Force clear authentication state even on error
        st.session_state.authentication_status = None
        st.session_state.username = None
        st.session_state.name = None


def check_for_new_csvs_lightweight():
    """
    Lightweight check: Just counts new CSV files without downloading (PDFs are ignored).
    Cached for 60 seconds to prevent excessive S3 pagination calls.
    Returns: (new_count, error_message)
    """
    try:
        # Get already processed keys from disk cache (survives restarts)
        # Also check session state as a fallback
        processed_keys = set()

        # First, try to load from disk cache to get already processed keys
        disk_result = load_cached_data_from_disk()
        # CRITICAL FIX: Check if disk_result is None before accessing its elements
        if disk_result and disk_result[0] is not None and len(disk_result[0]) > 0:
            cached_calls = disk_result[0]
            logger.info(
                f" Checking disk cache for processed keys: {len(cached_calls)} calls"
            )

            # Extract all S3 keys from cached calls
            keys_found = 0
            keys_missing = 0
            for call in cached_calls:
                # Try multiple ways to get the S3 key
                s3_key = call.get("_s3_key") or call.get("_id")

                # If still no key, try to get from filename
                if not s3_key:
                    filename = call.get("Filename") or call.get("Call ID")
                    if filename:
                        # Reconstruct S3 key from filename
                        if s3_prefix:
                            # Handle both cases: filename with or without prefix
                            if filename.startswith(s3_prefix):
                                s3_key = filename
                            else:
                                s3_key = f"{s3_prefix.rstrip('/')}/{filename}"
                        else:
                            s3_key = filename
                        # Ensure it ends with .csv (legacy format handling)
                        if not s3_key.lower().endswith(".csv"):
                            s3_key = f"{s3_key}.csv"

                # Normalize the key (remove leading/trailing slashes for comparison)
                if s3_key:
                    s3_key = s3_key.strip("/")
                    processed_keys.add(s3_key)
                    keys_found += 1
                else:
                    keys_missing += 1

            logger.info(
                f" Found {keys_found} processed S3 keys in disk cache ({keys_missing} calls missing keys)"
            )
        else:
            logger.info(
                " No disk cache found in count_new_csvs - all files will be treated as new"
            )

        # Also check session state (for files processed in current session)
        session_keys = st.session_state.get("processed_s3_keys", set())
        if session_keys:
            processed_keys.update(session_keys)
            logger.info(f" Found {len(session_keys)} additional files in session state")

        logger.info(f" Total {len(processed_keys)} files already processed")

        # Configure S3 client with short timeout for quick checks
        import botocore.config

        config = botocore.config.Config(
            connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}
        )
        s3_client_quick = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
            region_name=st.secrets["s3"].get("region_name", "us-east-1"),
            config=config,
        )

        # List all CSV files in S3 (quick check - just count, PDFs are ignored)
        paginator = s3_client_quick.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=s3_bucket_name, Prefix=s3_prefix, MaxKeys=1000
        )

        # Count new CSV files (not in processed_keys)
        new_count = 0
        total_csvs = 0
        processed_keys_normalized = {
            key.strip("/") for key in processed_keys
        }  # Consistent normalization

        # Track pagination progress
        page_count = 0
        is_truncated = False

        for page in pages:
            page_count += 1

            if isinstance(page, dict) and "Contents" in page:
                for obj in page["Contents"]:
                    key_raw = obj.get("Key")
                    if not key_raw:
                        continue
                    key = key_raw.strip(
                        "/"
                    )  # Normalize S3 key (consistent with cache normalization)
                    if key.lower().endswith(".csv"):
                        total_csvs += 1
                        if key not in processed_keys_normalized:
                            new_count += 1
            # Check if pagination is truncated
            if "IsTruncated" in page:
                is_truncated = page["IsTruncated"]
                # Log IsTruncated status every 10 pages or if truncated
                if page_count % 10 == 0 or is_truncated:
                    logger.info(f" Page {page_count}: IsTruncated={is_truncated}")

        # Log pagination completion with final IsTruncated status (combined)
        logger.info(
            f" Lightweight check: {page_count} pages, {total_csvs} CSV files, IsTruncated={is_truncated} (False=complete, True=may be incomplete)"
        )

        # Verify pagination completed (warn if suspicious)
        if total_csvs > 0 and total_csvs % 1000 == 0:
            logger.warning(
                f" Total files ({total_csvs}) is exactly divisible by 1000 - pagination might be incomplete!"
            )
        if is_truncated:
            logger.error(
                "CRITICAL: Last page had IsTruncated=True! Pagination may be incomplete - expected more pages!"
            )

        # Validate S3 listing completeness
        if len(processed_keys_normalized) > total_csvs:
            logger.error(
                f" CRITICAL: Cache has MORE keys ({len(processed_keys_normalized)}) than S3 ({total_csvs})!"
            )
            logger.error(
                "This suggests S3 listing is incomplete or cache has orphaned entries."
            )

        logger.info(
            f" CSV Count: {total_csvs} total in S3, {len(processed_keys_normalized)} processed, {new_count} new"
        )

        # Note: For lightweight check, we don't do full exhaustive comparison to save time
        # The full exhaustive comparison happens in load_new_calls_only()

        return new_count, None

    except Exception as e:
        return 0, f"Error checking for new CSV files: {e}"


# Initialize notification count (for manual refresh button)
if "new_csvs_notification_count" not in st.session_state:
    st.session_state.new_csvs_notification_count = 0

# Prominent refresh button for when new data is added (Chloe, Shannon, and Jerson only)
if is_super_admin():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Refresh Data")


# Smart refresh button (Chloe, Shannon, and Jerson only) - only loads new CSV files
# Note: files_to_load will be defined later, but we'll use None here to get all cached data
if is_super_admin():
    if st.sidebar.button(
        " Refresh New Data",
        help="Only processes new CSV files added since last refresh. Fast and efficient!",
        type="primary",
    ):
        log_audit_event(current_username, "refresh_data", "Refreshed new data from S3")

        # Set flag to prevent main data loading during refresh (prevents conflicts and crashes)
        st.session_state["refresh_in_progress"] = True

        # IMPORTANT: Preserve Streamlit cache BEFORE refresh to avoid calling load_all_calls_cached() during refresh
        # This prevents crashes from triggering a full S3 reload during the refresh operation
        previous_streamlit_cache = None
        previous_streamlit_errors = []
        try:
            # Get Streamlit cache BEFORE refresh (safe to call here, won't trigger reload)
            # Use cache_version when preserving Streamlit cache
            cache_version = st.session_state.get("_cache_version", 0)
            streamlit_result = load_all_calls_cached(cache_version=cache_version)
            previous_streamlit_cache = (
                streamlit_result[0] if streamlit_result[0] else []
            )
            previous_streamlit_errors = (
                streamlit_result[1] if streamlit_result[1] else []
            )
            logger.info(
                f" Preserved Streamlit cache: {len(previous_streamlit_cache)} calls before refresh"
            )
        except Exception as e:
            logger.warning(
                f" Could not preserve Streamlit cache: {e} - will use disk cache only"
            )

        # CRITICAL FIX: Wrap load_new_calls_only() in try/except to ensure refresh_in_progress flag is always cleared
        refresh_failed = False
        try:
            with st.spinner(" Checking for new CSV files..."):
                new_calls, new_errors, new_count = load_new_calls_only()
        except Exception as e:
            # If load_new_calls_only() crashes, clear flag and show error
            refresh_failed = True
            st.session_state["refresh_in_progress"] = False
            logger.exception(f" CRITICAL: load_new_calls_only() crashed: {e}")
            st.error(f" Refresh failed with unexpected error: {e}")
            st.info(
                " Please try again or use 'Reload ALL Data' button if the issue persists"
            )
            # Return early to prevent further processing
            new_calls, new_errors, new_count = [], f"Unexpected error: {e}", 0

        # Check if there was an overall error (returns string instead of list)
        if isinstance(new_errors, str):
            # Overall error occurred (e.g., network timeout, S3 access issue)
            refresh_failed = True
            st.session_state["refresh_in_progress"] = False  # Clear flag on error
            st.error(f" Error refreshing data: {new_errors}")
            st.info(" Try using 'Reload ALL Data' button if the issue persists")
            # Rerun to update UI and prevent further processing
            st.rerun()
        elif new_count > 0 and not refresh_failed:
            # Successfully found and processed new files
            # CRITICAL FIX: load_new_calls_only() already merged new calls with existing_calls incrementally
            # and saved to disk during processing. Don't merge again - just load from disk and verify.

            # Load disk cache once (should already contain merged data from incremental saves)
            disk_result = load_cached_data_from_disk()
            # CRITICAL FIX: Check if disk_result is None before accessing its elements
            disk_cached_calls = (
                disk_result[0] if (disk_result and disk_result[0] is not None) else []
            )
            disk_cached_errors = (
                disk_result[1] if (disk_result and disk_result[1] is not None) else []
            )

            # The disk cache should already contain all merged data from incremental saves
            # Verify that incremental saves worked correctly by checking if new_calls keys are in disk cache
            disk_cache_keys = {
                call.get("_s3_key") or call.get("_id")
                for call in disk_cached_calls
                if call.get("_s3_key") or call.get("_id")
            }
            new_calls_keys = {
                call.get("_s3_key") or call.get("_id")
                for call in new_calls
                if call.get("_s3_key") or call.get("_id")
            }

            # Check if incremental saves worked: all new_calls keys should be in disk cache
            missing_keys = new_calls_keys - disk_cache_keys
            if missing_keys:
                # Incremental saves didn't work properly - merge as fallback
                logger.warning(
                    f" Disk cache missing {len(missing_keys)} keys from new_calls - incremental saves may have failed, merging as fallback"
                )
                logger.warning(f" Sample missing keys: {list(missing_keys)[:5]}")
                all_calls_merged = disk_cached_calls + new_calls
                all_calls_merged = deduplicate_calls(all_calls_merged)
                logger.info(
                    f" Fallback merge: {len(disk_cached_calls)} + {len(new_calls)} = {len(all_calls_merged)} unique calls"
                )
            elif not disk_cached_calls and new_calls:
                # Disk cache is empty but we have new calls - merge as fallback
                logger.warning(
                    f" Disk cache is empty but {len(new_calls)} new calls were processed - merging as fallback"
                )
                # CRITICAL FIX: Deduplicate before assignment to match what save_cached_data_to_disk() will do
                # This ensures verification check compares post-dedup counts correctly
                all_calls_merged = deduplicate_calls(new_calls)
                logger.info(
                    f" Fallback merge: Using {len(all_calls_merged)} unique calls from {len(new_calls)} new calls (disk cache was empty)"
                )
            else:
                # Use disk cache directly (already merged from incremental saves)
                all_calls_merged = disk_cached_calls
                if new_calls_keys and disk_cache_keys:
                    overlap_count = len(new_calls_keys & disk_cache_keys)
                    logger.info(
                        f" Using disk cache directly: {len(all_calls_merged)} calls (verified: {overlap_count}/{len(new_calls_keys)} new keys in cache)"
                    )
                else:
                    logger.info(
                        f" Using disk cache directly: {len(all_calls_merged)} calls (already merged from incremental saves)"
                    )

            # Mark cache as complete (non-partial) now that refresh is done
            try:
                save_cached_data_to_disk(
                    all_calls_merged,
                    new_errors if new_errors else disk_cached_errors,
                    partial=False,
                )
                logger.info(f" Marked cache as complete: {len(all_calls_merged)} calls")
            except Exception as save_error:
                logger.error(
                    f" CRITICAL: Failed to mark cache as complete: {save_error}"
                )
                logger.error(
                    "Cache may still be marked as partial - this could cause issues on next refresh"
                )
                # Continue anyway - verification will catch this

            # Verify disk cache was saved correctly (use single load for verification)
            # CRITICAL FIX: Add error handling for verification load to prevent crashes
            disk_result_verify = None
            disk_cache_count = 0
            verification_failed = False
            try:
                disk_result_verify = load_cached_data_from_disk()
                if disk_result_verify and disk_result_verify[0] is not None:
                    disk_cache_count = len(disk_result_verify[0])
                else:
                    logger.warning(
                        "Verification load returned None - checking previous disk_result as fallback"
                    )
                    # CRITICAL FIX: Check if disk_result is None before using as fallback
                    if disk_result is not None and disk_result[0] is not None:
                        disk_result_verify = disk_result
                        disk_cache_count = len(disk_result[0])
                    else:
                        logger.error(
                            "Both verification load and disk_result are None - using empty lists as fallback"
                        )
                        disk_result_verify = ([], [])
                        disk_cache_count = 0
                        verification_failed = True
            except Exception as verify_error:
                logger.error(f" CRITICAL: Verification load failed: {verify_error}")
                logger.error(
                    "Checking previous disk_result as fallback for verification"
                )
                verification_failed = True
                # CRITICAL FIX: Check if disk_result is None before using as fallback
                if disk_result is not None and disk_result[0] is not None:
                    disk_result_verify = disk_result
                    disk_cache_count = len(disk_result[0])
                    verification_failed = False  # We have valid fallback data
                else:
                    logger.error(
                        "Both verification load and disk_result are None - using empty lists as fallback"
                    )
                    disk_result_verify = ([], [])
                    disk_cache_count = 0

            if disk_cache_count >= len(all_calls_merged):
                # Disk cache was saved successfully
                # CRITICAL FIX: Check if disk_result_verify and its elements are valid before using
                # Also check if verification failed and we're using empty fallback lists
                if (
                    disk_result_verify is None
                    or disk_result_verify[0] is None
                    or (verification_failed and disk_result_verify == ([], []))
                ):
                    logger.error(
                        "CRITICAL: disk_result_verify is None or has None data - cannot proceed with merge"
                    )
                    logger.error("Using all_calls_merged as fallback")
                    # Use all_calls_merged as fallback since verification failed
                    st.session_state["_merged_cache_data"] = all_calls_merged
                    st.session_state["_merged_cache_errors"] = (
                        new_errors if new_errors else []
                    )
                # Merge preserved Streamlit cache with disk cache (preserving all data)
                elif previous_streamlit_cache and len(previous_streamlit_cache) > 0:
                    # CRITICAL FIX: Only merge if disk_result_verify[0] is not None
                    # Defensive check: ensure disk_result_verify is not None before accessing [0]
                    if disk_result_verify and disk_result_verify[0] is not None:
                        # CRITICAL FIX: Check if caches are identical before merging
                        # If they have the same keys, skip merge to avoid crash (no need to merge identical data)
                        # Extract keys from both caches for comparison
                        previous_keys = set()
                        for call in previous_streamlit_cache:
                            key = (
                                call.get("_s3_key")
                                or call.get("_id")
                                or call.get("Call ID")
                            )
                            if key:
                                previous_keys.add(key)

                        disk_keys = set()
                        for call in disk_result_verify[0]:
                            key = (
                                call.get("_s3_key")
                                or call.get("_id")
                                or call.get("Call ID")
                            )
                            if key:
                                disk_keys.add(key)

                        if previous_keys == disk_keys:
                            # Caches are identical - no merge needed, use all_calls_merged directly
                            logger.info(
                                f" Streamlit cache and disk cache have identical keys ({len(previous_keys)} calls) - skipping merge to avoid crash"
                            )
                            logger.info(
                                f" Using all_calls_merged ({len(all_calls_merged)} calls) directly - no data loss"
                            )
                            st.session_state["_merged_cache_data"] = all_calls_merged
                            verify_errors = (
                                disk_result_verify[1]
                                if (
                                    disk_result_verify
                                    and len(disk_result_verify) > 1
                                    and disk_result_verify[1] is not None
                                )
                                else []
                            )
                            st.session_state["_merged_cache_errors"] = (
                                new_errors if new_errors else verify_errors
                            )
                            # BUG FIX: Store timestamp when _merged_cache_data is set so we can detect stale data
                            st.session_state["_merged_cache_data_timestamp"] = (
                                datetime.now().isoformat()
                            )
                        else:
                            # Caches are different - merge to preserve all data
                            # Calculate difference for logging
                            only_in_streamlit = previous_keys - disk_keys
                            only_in_disk = disk_keys - previous_keys

                            logger.info(
                                f" Caches differ: {len(only_in_streamlit)} calls only in Streamlit, {len(only_in_disk)} calls only in disk"
                            )
                            logger.info(
                                f" Merging preserved Streamlit cache ({len(previous_streamlit_cache)} calls) with disk cache ({disk_cache_count} calls)"
                            )

                            # Merge preserved Streamlit cache with disk cache
                            merged_data = (
                                previous_streamlit_cache + disk_result_verify[0]
                            )
                        merged_data = deduplicate_calls(merged_data)
                        merged_count = len(merged_data)

                        logger.info(
                            f"Merged result: {merged_count} unique calls (removed {len(previous_streamlit_cache) + len(disk_result_verify[0]) - merged_count} duplicates)"
                        )

                        # Save merged result to disk
                        merge_save_succeeded = False
                        try:
                            # CRITICAL FIX: Check if disk_result_verify[1] exists before using
                            verify_errors = (
                                disk_result_verify[1]
                                if (
                                    disk_result_verify
                                    and len(disk_result_verify) > 1
                                    and disk_result_verify[1] is not None
                                )
                                else []
                            )
                            save_cached_data_to_disk(
                                merged_data,
                                previous_streamlit_errors
                                if previous_streamlit_errors
                                else verify_errors,
                            )
                            logger.info(
                                f"Saved merged cache to disk: {merged_count} calls"
                            )
                            merge_save_succeeded = True
                        except Exception as merge_save_error:
                            logger.error(
                                f"CRITICAL: Failed to save merged cache: {merge_save_error}"
                            )
                            logger.error(
                                "Using disk cache without Streamlit cache merge - some data may be lost"
                            )

                        # CRITICAL FIX: Only store merged_data in session state if save succeeded
                        # If save failed, use disk cache (fallback) to prevent data loss
                        if merge_save_succeeded:
                            # Store merged data in session state so it's used after cache clear
                            st.session_state["_merged_cache_data"] = merged_data
                            verify_errors = (
                                disk_result_verify[1]
                                if (
                                    disk_result_verify
                                    and len(disk_result_verify) > 1
                                    and disk_result_verify[1] is not None
                                )
                                else []
                            )
                            st.session_state["_merged_cache_errors"] = (
                                previous_streamlit_errors
                                if previous_streamlit_errors
                                else verify_errors
                            )
                            # BUG FIX: Store timestamp when _merged_cache_data is set so we can detect stale data
                            st.session_state["_merged_cache_data_timestamp"] = (
                                datetime.now().isoformat()
                            )
                        else:
                            # Fallback: use disk cache without merge (save failed, merged_data not on disk)
                            # CRITICAL FIX: Skip using disk_result_verify entirely if verification failed
                            # Don't use potentially corrupted cache data when verification fails
                            if (
                                not verification_failed
                                and disk_result_verify
                                and disk_result_verify[0] is not None
                            ):
                                st.session_state["_merged_cache_data"] = (
                                    disk_result_verify[0]
                                )
                                verify_errors = (
                                    disk_result_verify[1]
                                    if (
                                        disk_result_verify
                                        and len(disk_result_verify) > 1
                                        and disk_result_verify[1] is not None
                                    )
                                    else []
                                )
                                st.session_state["_merged_cache_errors"] = verify_errors
                                # BUG FIX: Store timestamp when _merged_cache_data is set so we can detect stale data
                                st.session_state["_merged_cache_data_timestamp"] = (
                                    datetime.now().isoformat()
                                )
                            else:
                                logger.error(
                                    "CRITICAL: Cannot use disk cache (verification failed or invalid) - using all_calls_merged as fallback"
                                )
                                st.session_state["_merged_cache_data"] = (
                                    all_calls_merged
                                )
                                st.session_state["_merged_cache_errors"] = (
                                    new_errors if new_errors else []
                                )
                                # BUG FIX: Store timestamp when _merged_cache_data is set so we can detect stale data
                                st.session_state["_merged_cache_data_timestamp"] = (
                                    datetime.now().isoformat()
                                )
                    else:
                        # disk_result_verify[0] is None - use previous_streamlit_cache only
                        logger.warning(
                            "disk_result_verify[0] is None - using previous_streamlit_cache only"
                        )
                        st.session_state["_merged_cache_data"] = (
                            previous_streamlit_cache
                        )
                        st.session_state["_merged_cache_errors"] = (
                            previous_streamlit_errors
                            if previous_streamlit_errors
                            else []
                        )
                        # BUG FIX: Store timestamp when _merged_cache_data is set so we can detect stale data
                        st.session_state["_merged_cache_data_timestamp"] = (
                            datetime.now().isoformat()
                        )
                else:
                    # No previous Streamlit cache - just use disk cache
                    logger.info(
                        f"No previous Streamlit cache to merge - using disk cache ({disk_cache_count} calls)"
                    )
                    # CRITICAL FIX: Skip using disk_result_verify entirely if verification failed
                    # Don't use potentially corrupted cache data when verification fails
                    if (
                        not verification_failed
                        and disk_result_verify
                        and disk_result_verify[0] is not None
                    ):
                        st.session_state["_merged_cache_data"] = disk_result_verify[0]
                        # CRITICAL FIX: Check if disk_result_verify is None before calling len()
                        verify_errors = (
                            disk_result_verify[1]
                            if (
                                disk_result_verify
                                and len(disk_result_verify) > 1
                                and disk_result_verify[1] is not None
                            )
                            else []
                        )
                        st.session_state["_merged_cache_errors"] = verify_errors
                        # BUG FIX: Store timestamp when _merged_cache_data is set so we can detect stale data
                        st.session_state["_merged_cache_data_timestamp"] = (
                            datetime.now().isoformat()
                        )
                    else:
                        logger.error(
                            "CRITICAL: disk_result_verify is invalid (None or failed verification) - using all_calls_merged as fallback"
                        )
                        st.session_state["_merged_cache_data"] = all_calls_merged
                        st.session_state["_merged_cache_errors"] = (
                            new_errors if new_errors else []
                        )
                        # BUG FIX: Store timestamp when _merged_cache_data is set so we can detect stale data
                        st.session_state["_merged_cache_data_timestamp"] = (
                            datetime.now().isoformat()
                        )

                # CRITICAL FIX: Don't clear Streamlit cache - let it update naturally
                # The merged data is stored in session state (_merged_cache_data)
                # When load_all_calls_cached() is called next, it will return that data
                # and Streamlit's @st.cache_data will automatically cache it
                # This ensures disk cache is backed up in Streamlit cache without risk of data loss
                # Clearing cache here could cause data loss if something goes wrong before next call
                logger.info(
                    f" Disk cache saved ({disk_cache_count} calls) - Streamlit cache will update with latest data on next access"
                )
                logger.info(
                    "Merged data stored in session state - will be cached automatically when load_all_calls_cached() is called"
                )

                # REMOVED: load_all_calls_cached.clear() - causes crashes during refresh/rerun
                # Instead, rely on _merged_cache_data in session state which load_all_calls_cached()
                # will use on next call (line 1118-1126), and Streamlit will cache that result automatically.
                # The disk cache is already persisted to logs/cached_calls_data.json, so data is safe.
            else:
                logger.warning(
                    f" Disk cache verification failed: expected {len(all_calls_merged)} calls, found {disk_cache_count} - NOT clearing Streamlit cache"
                )
                # CRITICAL FIX: When verification fails, use verified disk cache data if available
                # Don't store unverified all_calls_merged that may not have been persisted to disk
                if (
                    not verification_failed
                    and disk_result_verify
                    and disk_result_verify[0] is not None
                ):
                    logger.info(
                        f" Using verified disk cache data ({disk_cache_count} calls) instead of unverified merged data ({len(all_calls_merged)} calls)"
                    )
                    all_calls_merged = disk_result_verify[0]
                    # Use verified errors if available
                    if (
                        disk_result_verify
                        and len(disk_result_verify) > 1
                        and disk_result_verify[1] is not None
                    ):
                        new_errors = disk_result_verify[1]
                else:
                    logger.error(
                        "CRITICAL: Verification failed and no valid disk cache available - data may not be persisted"
                    )
                    # Still use all_calls_merged as last resort, but log the risk
                    logger.warning(
                        "Using unverified all_calls_merged as fallback - data may be lost on restart"
                    )

            # We need to manually update the cache - store in session state temporarily
            st.session_state["merged_calls"] = all_calls_merged
            st.session_state["merged_errors"] = new_errors if new_errors else []
            # Update processed keys tracking
            if "processed_s3_keys" not in st.session_state:
                st.session_state["processed_s3_keys"] = set()
            new_keys = {
                call.get("_s3_key") for call in new_calls if call.get("_s3_key")
            }
            st.session_state["processed_s3_keys"].update(new_keys)
            st.success(
                f" Added {new_count} new call(s)! Total: {len(all_calls_merged)} calls"
            )
            if new_errors:
                st.warning(f" {len(new_errors)} file(s) had errors")
            # Clear notification count after successful refresh
            st.session_state.new_csvs_notification_count = 0

            # CRITICAL FIX: Increment cache version and clear Streamlit cache before rerun
            # This forces Streamlit cache to reload from S3 cache (source of truth) on next call
            try:
                # Increment cache version to force cache refresh
                current_version = st.session_state.get("_cache_version", 0)
                st.session_state["_cache_version"] = current_version + 1

                # Store S3 cache timestamp after saving (so we can detect when it's updated)
                s3_client, s3_bucket = get_s3_client_and_bucket()
                if s3_client and s3_bucket:
                    try:
                        response = s3_client.get_object(
                            Bucket=s3_bucket, Key=S3_CACHE_KEY
                        )
                        s3_data = json.loads(response["Body"].read().decode("utf-8"))
                        s3_timestamp = s3_data.get("timestamp", None)
                        if s3_timestamp:
                            st.session_state["_s3_cache_timestamp"] = s3_timestamp
                            logger.info(f" Stored S3 cache timestamp: {s3_timestamp}")
                    except Exception as s3_read_error:
                        logger.debug(
                            f"Could not read S3 cache timestamp: {s3_read_error}"
                        )

                # Clear Streamlit cache to force reload from S3 cache
                load_all_calls_cached.clear()
                logger.info(
                    f" Cleared Streamlit cache - will reload {len(all_calls_merged)} calls from S3 cache (source of truth)"
                )
            except Exception as clear_error:
                logger.warning(
                    f" Could not clear Streamlit cache: {clear_error} - will rely on cache_version parameter"
                )
                # Continue anyway - cache_version parameter will force refresh

            # Clear refresh flag before rerun
            st.session_state["refresh_in_progress"] = False
            # Rerun to show updated data (Streamlit cache will reload from _merged_cache_data)
            st.rerun()
        else:
            # No new files found and no errors
            st.session_state["refresh_in_progress"] = False  # Clear flag
            st.info(" No new CSV files found. All data is up to date!")
# Admin-only: Full reload button (Super admins only)
if is_super_admin():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Admin: Full Reload")

    # Check if reload is already in progress
    reload_in_progress = st.session_state.get("reload_all_triggered", False)

    if reload_in_progress:
        st.sidebar.warning("⏳ Reload in progress... Please wait.")
        st.sidebar.info(
            "This may take 10-20 minutes. The page will refresh automatically when complete."
        )
    else:
        if st.sidebar.button(
            " Reload ALL Data (Admin Only)",
            help=" Clears cache and reloads ALL CSV files from S3. This may take 10-20 minutes.",
            type="secondary",
        ):
            reload_success = False
            try:
                if is_super_admin():
                    log_audit_event(
                        current_username,
                        "reload_all_data",
                        "Cleared cache and reloaded all data from S3",
                    )

                # Clear Streamlit cache with error handling
                try:
                    st.cache_data.clear()
                    logger.info(" Cleared Streamlit cache")
                except Exception as e:
                    logger.warning(f"Failed to clear Streamlit cache: {e}")
                    # Continue anyway - other caches can still be cleared

                # Clear persistent disk cache with timeout handling
                if CACHE_FILE.exists():
                    try:
                        with cache_file_lock(CACHE_FILE, timeout=5):
                            CACHE_FILE.unlink()
                            logger.info(f" Cleared persistent disk cache: {CACHE_FILE}")
                    except LockTimeoutError:
                        logger.warning(
                            f"Timeout clearing disk cache (file may be locked)"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to clear disk cache: {e}")

                # Clear S3 cache with retry logic
                try:
                    s3_client, s3_bucket = get_s3_client_and_bucket()
                    if s3_client and s3_bucket:
                        try:
                            s3_client.delete_object(Bucket=s3_bucket, Key=S3_CACHE_KEY)
                            logger.info(
                                f" Cleared S3 cache: s3://{s3_bucket}/{S3_CACHE_KEY}"
                            )
                        except ClientError as e:
                            if e.response.get("Error", {}).get("Code") != "NoSuchKey":
                                logger.warning(f" Could not delete S3 cache: {e}")
                            else:
                                logger.info(
                                    " S3 cache does not exist (already deleted)"
                                )
                except Exception as e:
                    logger.warning(f"Failed to clear S3 cache: {e}")

                # Clear session state cache keys safely
                cache_keys_to_clear = [
                    "processed_s3_keys",
                    "_s3_cache_result",
                    "_s3_cache_timestamp",
                    "_merged_cache_data",
                    "_merged_cache_errors",
                    "_merged_cache_data_timestamp",
                ]
                for key in cache_keys_to_clear:
                    try:
                        if key in st.session_state:
                            del st.session_state[key]
                    except Exception as e:
                        logger.warning(
                            f"Failed to clear session state key '{key}': {e}"
                        )

                # Mark that full dataset should be cached after this reload
                st.session_state["full_dataset_cached"] = (
                    False  # Will be set to True after load completes
                )

                # Clear load in progress flag to allow new load
                try:
                    if "_data_load_in_progress" in st.session_state:
                        del st.session_state["_data_load_in_progress"]
                except Exception as e:
                    logger.warning(f"Failed to clear load in progress flag: {e}")

                # Set flag to trigger full load
                st.session_state["reload_all_triggered"] = True
                reload_success = True

                st.success(" Cache cleared! Reloading all data from S3...")
                logger.info("Reload ALL Data button clicked - starting full reload")
                st.rerun()
            except Exception as e:
                logger.exception(f"Error initiating reload: {e}")
                st.error(f"❌ Failed to initiate reload: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")
            finally:
                # Clear all flags if there was an error to prevent retry loops
                if not reload_success:
                    try:
                        if "reload_all_triggered" in st.session_state:
                            st.session_state["reload_all_triggered"] = False
                        if "_data_load_in_progress" in st.session_state:
                            del st.session_state["_data_load_in_progress"]
                        logger.info(" Cleared reload flags due to error")
                    except Exception as clear_error:
                        logger.warning(
                            f"Failed to clear flags in finally block: {clear_error}"
                        )
                # Clear all flags if there was an error
                if "reload_all_triggered" in st.session_state:
                    st.session_state["reload_all_triggered"] = False
                if "_data_load_in_progress" in st.session_state:
                    del st.session_state["_data_load_in_progress"]


# --- Load Rubric Reference ---
@st.cache_data
def load_rubric():
    """Load the rubric JSON file."""
    try:
        import json
        import os

        rubric_path = os.path.join(os.path.dirname(__file__), "Rubric_v33.json")
        if os.path.exists(rubric_path):
            with open(rubric_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except Exception:
        return None


rubric_data = load_rubric()

# --- Rubric Reference Link in Sidebar ---
st.sidebar.markdown("---")
if rubric_data:
    st.sidebar.info(" View full rubric reference in the main dashboard below")
else:
    st.sidebar.warning(" Rubric file not found")

# --- Fetch Call Metadata ---
# Only test S3 connection AFTER user is logged in (moved from before login)
status_text = st.empty()

# Check if data is already loaded - skip S3 test if so
data_already_loaded = (
    "merged_calls" in st.session_state
    or st.session_state.get("_s3_cache_result") is not None
    or st.session_state.get("_data_load_in_progress", False)
)

if not data_already_loaded:
    status_text.text(" Preparing to load data...")
    logger.info("Starting data load process (user is authenticated)")

    try:
        # Quick connection test with aggressive timeouts (only after login)
        # Only test once per session to avoid unnecessary tests
        s3_test_key = "_s3_connection_tested"
        if not st.session_state.get(s3_test_key, False):
            import botocore.config

            config = botocore.config.Config(
                connect_timeout=5,  # 5 seconds max
                read_timeout=10,  # 10 seconds max
                retries={"max_attempts": 1},  # No retries for faster failure
            )
            test_client = boto3.client(
                "s3",
                aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
                aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
                region_name=st.secrets["s3"].get("region_name", "us-east-1"),
                config=config,
            )

            status_text.text(" Testing S3 connection...")
            logger.debug(f"Testing connection to bucket: {s3_bucket_name}")

            # Quick test - just check if we can access the bucket with timeout
            try:
                test_client.head_bucket(Bucket=s3_bucket_name)
                logger.debug("S3 connection test successful")
                status_text.text(" Connected! Loading data...")
                st.session_state[s3_test_key] = True  # Mark as tested
            except Exception as bucket_error:
                logger.error(f"S3 bucket access failed: {bucket_error}")
                status_text.empty()
                st.warning(" **S3 Connection Issue**")
                st.warning(
                    "Could not connect to S3. The app will try to use cached data if available."
                )
                st.info(
                    " **If you see cached data below, you can continue using the app.**"
                )
                st.info(" **If not, check your S3 credentials or network connection.**")
                # Don't stop - try to load from cache instead
                status_text.text(" Attempting to load from cache...")
                st.session_state[s3_test_key] = True  # Mark as tested even on failure
        else:
            # Already tested, just show loading message
            status_text.text(" Loading data...")
            logger.debug("S3 connection already tested, skipping test")

        # Skip CSV count for faster startup - just load data directly
        pdf_count = None
        logger.debug(
            "Skipping PDF count for faster startup - proceeding to data loading..."
        )

    except ClientError as e:
        status_text.empty()
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "404":
            st.error(f" S3 bucket '{s3_bucket_name}' not found.")
            st.error("Please check the bucket name in your secrets.toml")
        elif error_code == "403":
            st.error(f" Access denied to S3 bucket '{s3_bucket_name}'.")
            st.error("Please check your AWS credentials and IAM permissions")
        else:
            st.error(f" S3 connection error: {e}")
        st.stop()
    except Exception as e:
        status_text.empty()
        logger.exception(f"Error during S3 connection test: {e}")
        # Don't stop - try to load from cache instead
        st.warning(" **S3 Connection Issue**")
        st.warning(
            "Could not connect to S3. The app will try to use cached data if available."
        )
        st.info(" **If you see cached data below, you can continue using the app.**")
        status_text.text(" Attempting to load from cache...")
        st.session_state["_s3_connection_tested"] = (
            True  # Mark as tested to prevent retry
        )
else:
    # Data already loaded, skip S3 test
    logger.debug("Data already loaded, skipping S3 connection test")
    status_text.empty()

# Always load all files - caching handles performance
# First load will process all CSV files, then cached indefinitely for instant access

# Now load the actual data
logger.debug("Entering data loading section...")
# Initialize call_data and errors to prevent undefined variable errors
call_data = []
errors = []
try:
    status_text.text(" Loading CSV files from S3...")
    logger.debug("Status text updated, starting timer...")

    t0 = time.time()
    was_processing = False  # Track if we actually processed files
    logger.debug(f"Timer started at {t0}")

    # Check if we have merged data from smart refresh
    logger.debug("Checking for merged calls in session state...")
    if "merged_calls" in st.session_state:
        logger.info("Found merged calls in session state, using cached data")
        # Use merged data from smart refresh
        call_data = st.session_state["merged_calls"]
        errors = st.session_state.get("merged_errors", [])
        # Clear the temporary session state
        del st.session_state["merged_calls"]
        if "merged_errors" in st.session_state:
            del st.session_state["merged_errors"]
        # Note: Disk cache already has the merged data from refresh, Streamlit cache will update on next access
        elapsed = time.time() - t0
        status_text.empty()
        logger.info(f"Merged data loaded in {elapsed:.2f} seconds")
    else:
        # Check if refresh is in progress - skip main data loading to prevent conflicts
        if st.session_state.get("refresh_in_progress", False):
            logger.info(
                " Refresh in progress - skipping main data load to prevent conflicts"
            )
            # Use disk cache if available, but don't trigger a new load
            try:
                disk_result = load_cached_data_from_disk()
                # CRITICAL FIX: Check if disk_result is None before accessing its elements
                if disk_result and disk_result[0] and len(disk_result[0]) > 0:
                    call_data = disk_result[0]
                    # CRITICAL FIX: Check if disk_result is None before accessing disk_result[1]
                    errors = (
                        disk_result[1]
                        if (disk_result and disk_result[1] is not None)
                        else []
                    )
                    logger.info(
                        f" Using disk cache during refresh: {len(call_data)} calls"
                    )
                else:
                    call_data = []
                    errors = []
                    logger.info(
                        " No disk cache available during refresh - will load after refresh completes"
                    )
            except Exception as e:
                logger.warning(f"Could not load disk cache during refresh: {e}")
                call_data = []
                errors = []
            elapsed = time.time() - t0
            status_text.empty()
        else:
            logger.debug(
                "No merged calls found, proceeding with normal load from cache or S3"
            )
            # Check if data is already loaded and not stale - skip reload if so
            # Only use cached data if it's substantial (at least 100 calls) to avoid using stale/partial data
            if (
                "_s3_cache_result" in st.session_state
                and st.session_state["_s3_cache_result"] is not None
                and not st.session_state.get("reload_all_triggered", False)
                and not st.session_state.get("_data_load_in_progress", False)
            ):
                logger.debug(
                    "Data already loaded in session state, checking cached result"
                )
                cached_result = st.session_state["_s3_cache_result"]
                # Only use cached data if it's substantial (at least 100 calls)
                # This prevents using stale/partial data from previous sessions
                if cached_result and len(cached_result) >= 100:
                    call_data = cached_result
                    errors = st.session_state.get("_last_load_errors", [])
                    elapsed = time.time() - t0
                    status_text.empty()
                    logger.info(
                        f"Using cached data from session: {len(call_data)} calls"
                    )
                else:
                    # Cached result is too small or empty, proceed with normal load
                    logger.debug(
                        f"Cached result is too small ({len(cached_result) if cached_result else 0} calls), proceeding with normal load"
                    )
                    # Clear the invalid cache and proceed to normal load
                    if "_s3_cache_result" in st.session_state:
                        del st.session_state["_s3_cache_result"]
            else:
                # Normal load from cache or S3
                # Load all files - first load will process all CSV files, then cached indefinitely for instant access
                # After first load, data is CACHED indefinitely - subsequent loads will be INSTANT until you manually refresh
                logger.debug("Setting up progress tracking...")

                # Initialize progress tracking
                if "csv_processing_progress" not in st.session_state:
                    st.session_state.csv_processing_progress = {
                        "processed": 0,
                        "total": 0,
                        "errors": 0,
                    }
                    logger.debug("Initialized new progress tracking in session state")
                else:
                    logger.debug("Using existing progress tracking from session state")

                # Create progress bar placeholder
                progress_placeholder = st.empty()
                progress_bar = None
                logger.debug("Progress placeholder created")

                # Show progress if we're processing files
                def update_progress():
                    if st.session_state.csv_processing_progress["total"] > 0:
                        processed = st.session_state.csv_processing_progress[
                            "processed"
                        ]
                        total = st.session_state.csv_processing_progress["total"]
                        errors = st.session_state.csv_processing_progress["errors"]
                        progress = processed / total if total > 0 else 0
                        progress_placeholder.progress(
                            progress,
                            text=f"Processing CSV files: {processed}/{total} ({errors} errors)",
                        )

                # Load data (this will trigger processing if not cached)
                # SIMPLE approach: Just call the cached function. Streamlit's cache handles everything.
                # If cache exists, it's instant. If not, it loads from S3 (only happens once, then cached).
                logger.debug(
                    "Loading data - Streamlit cache will handle it automatically"
                )

                try:
                    # Add timeout wrapper
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Data loading timed out after 5 minutes")

                    # Try to load with better error visibility
                    loading_placeholder = st.empty()
                    with loading_placeholder.container():
                        st.spinner(
                            "Loading PDFs from S3... This may take a few minutes for large datasets."
                        )

                    # Use cache_version to force cache refresh when refresh completes
                    cache_version = st.session_state.get("_cache_version", 0)
                    call_data, errors = load_all_calls_cached(
                        cache_version=cache_version
                    )
                    # Store data and errors in session state for reuse
                    st.session_state["_s3_cache_result"] = call_data
                    st.session_state["_last_load_errors"] = errors
                    if call_data:
                        st.session_state["_s3_cache_timestamp"] = time.time()
                    logger.info(
                        f"Data loaded. Got {len(call_data) if call_data else 0} calls"
                    )

                    # Clear loading messages
                    loading_placeholder.empty()
                except TimeoutError:
                    logger.exception("Timeout during data loading")
                    loading_placeholder.empty()
                    status_text.empty()
                    st.error(" **Loading Timeout**")
                    st.error(
                        "The data loading is taking too long. This might be due to:"
                    )
                    st.error("1. Slow S3 connection")
                    st.error("2. Large number of files to process")
                    st.error("3. Network issues")
                    st.info(" **Quick Fixes:**")
                    st.info(
                        "1. **Refresh the page** - if cache exists, it will load instantly"
                    )
                    st.info(
                        "2. **Wait 2-3 minutes** and refresh - the cache may be building"
                    )
                    st.info("3. **Check your internet connection**")
                    st.info(
                        "4. If you're an admin, try the ' Reload ALL Data' button after refresh"
                    )
                    st.stop()
                except Exception as e:
                    logger.exception("Error during data loading")
                    loading_placeholder.empty()
                    status_text.empty()
                    st.error(" **Error Loading Data**")
                    st.error(f"**Error:** {str(e)}")
                    st.error("The app may be trying to load too many files at once.")
                    st.info(" **Try this:**")
                    st.info(
                        "1. **Refresh the page** - if cache exists, it will load instantly"
                    )
                    st.info(
                        "2. Clear the cache by clicking ' Reload ALL Data (Admin Only)' button (if you're an admin)"
                    )
                    st.info("3. Wait a few minutes and refresh the page")
                    st.info("4. Check the terminal/logs for detailed errors")
                    with st.expander("Show full error details"):
                        import traceback

                        st.code(traceback.format_exc())
                    st.stop()

        # Clear progress after loading (only if progress_placeholder was created)
        was_processing = st.session_state.csv_processing_progress.get("total", 0) > 0
        if was_processing and "progress_placeholder" in locals():
            progress_placeholder.empty()
            st.session_state.csv_processing_progress = {
                "processed": 0,
                "total": 0,
                "errors": 0,
                "processing_start_time": None,
            }

        elapsed = time.time() - t0
        status_text.empty()

    # Check if we got valid data
    logger.debug(
        f"Checking call_data: type={type(call_data)}, len={len(call_data) if call_data else 0}, truthy={bool(call_data)}"
    )
    if not call_data and not errors:
        status_text.empty()
        st.warning(
            " No data loaded. This might be the first time loading, or there may be an issue."
        )
        st.info(
            " Try refreshing the page or clicking ' Reload ALL Data (Admin Only)' if you're an admin."
        )
        st.stop()

    # Handle errors - could be a tuple (errors_list, info_message) or just errors
    if errors:
        if isinstance(errors, tuple) and len(errors) == 2:
            # New format: (errors_list, info_message)
            errors_list, info_msg = errors
            st.info(f" {info_msg}")
            if errors_list:
                if len(errors_list) <= 10:
                    for error in errors_list:
                        st.warning(f" {error}")
                else:
                    st.warning(
                        f" {len(errors_list)} files had errors. Showing first 10:"
                    )
                    for error in errors_list[:10]:
                        st.warning(f" {error}")
        elif isinstance(errors, list) and len(errors) <= 10:
            for error in errors:
                st.warning(f" {error}")
        elif isinstance(errors, list) and len(errors) > 10:
            st.warning(f" {len(errors)} files had errors. Showing first 10:")
            for error in errors[:10]:
                st.warning(f" {error}")
        elif isinstance(errors, str):
            # Single error message
            st.error(f" {errors}")
            st.stop()

    logger.debug(
        f"Before final check: call_data type={type(call_data)}, len={len(call_data) if call_data else 0}, truthy={bool(call_data)}"
    )
    if call_data:
        # Only show cache messages if we actually processed files or refresh was triggered
        refresh_in_progress = st.session_state.get("refresh_in_progress", False)
        if was_processing and "last_actual_processing_time" in st.session_state:
            # We actually processed files - show actual processing time
            actual_time = st.session_state["last_actual_processing_time"]
            file_count = st.session_state.get(
                "last_processing_file_count", len(call_data)
            )
            if actual_time > 60:
                time_str = f"{actual_time / 60:.1f} minutes"
            else:
                time_str = f"{actual_time:.1f}s"
            st.success(f" Loaded {file_count} calls in {time_str}")
        elif refresh_in_progress and "last_actual_processing_time" in st.session_state:
            # Refresh was triggered - show when it was last processed
            last_time = st.session_state["last_actual_processing_time"]
            if last_time > 60:
                time_str = f"{last_time / 60:.1f} minutes"
            else:
                time_str = f"{last_time:.1f}s"
            if current_username and current_username.lower() in ["chloe", "shannon"]:
                st.success(
                    f" Loaded {len(call_data)} calls (from cache, originally processed in {time_str})"
                )
        # If call_data exists but we don't need to show processing messages, just continue silently
        # The data is loaded and will be used below
    else:
        # call_data is empty or falsy - show error
        st.error(" No call data found!")
        st.error("Possible issues:")
        st.error("1. No CSV files in S3 bucket (check bucket name and prefix)")
        st.error("2. CSV files couldn't be parsed")
        st.error("3. Check the prefix path if CSV files are in a subfolder")
        st.stop()
except Exception as e:
    status_text.empty()
    st.error(f" Error loading data: {e}")
    st.error("Please check:")
    st.error("1. S3 credentials in secrets.toml")
    st.error("2. Bucket name and region")
    st.error("3. AWS permissions")
    import traceback

    with st.expander("Show full error"):
        st.code(traceback.format_exc())
    st.stop()

# CRITICAL: Normalize all agent IDs in call_data BEFORE creating DataFrame
# This ensures cached data with old agent IDs gets normalized consistently
# This fixes the issue where cached DataFrames have wrong agent IDs
if call_data:
    agent_normalized_count = 0
    for call in call_data:
        if isinstance(call, dict) and "agent" in call:
            original_agent = call.get("agent")
            normalized_agent = normalize_agent_id(original_agent)
            if original_agent != normalized_agent:
                call["agent"] = normalized_agent
                agent_normalized_count += 1
        # Also check for "Agent" key (capitalized)
        if isinstance(call, dict) and "Agent" in call and "agent" not in call:
            original_agent = call.get("Agent")
            normalized_agent = normalize_agent_id(original_agent)
            if original_agent != normalized_agent:
                call["Agent"] = normalized_agent
                agent_normalized_count += 1

    if agent_normalized_count > 0:
        logger.info(
            f" Normalized {agent_normalized_count} agent IDs before DataFrame creation"
        )

meta_df = pd.DataFrame(call_data)

# --- ANONYMIZATION FUNCTIONS ---
# Note: is_anonymous_user is already defined earlier in the code


def create_anonymous_mapping(values, prefix="ID"):
    """
    Create a consistent mapping from real values to anonymous identifiers.
    Same input always produces same output for consistency.
    Uses sorted order to ensure deterministic mapping.
    """
    if not is_anonymous_user:
        return {}

    # Filter out null/empty values and get unique sorted list for consistency
    unique_values = sorted(
        [str(v).strip() for v in values if pd.notna(v) and str(v).strip()]
    )

    # Create mapping: real value -> anonymous ID (Agent-001, Agent-002, etc.)
    mapping = {}
    for idx, value in enumerate(unique_values, start=1):
        mapping[value] = f"{prefix}-{idx:03d}"

    return mapping


# Store anonymization mappings in session state for consistency across the app
if "anonymization_mappings" not in st.session_state:
    st.session_state.anonymization_mappings = {}


def anonymize_dataframe(df, create_mappings_from=None):
    """
    Anonymize identifying columns in the dataframe using vectorized operations for speed.

    Args:
        df: DataFrame to anonymize
        create_mappings_from: Optional DataFrame to use for creating mappings (for consistency)
    """
    if not is_anonymous_user:
        return df

    df = df.copy()

    # Use provided dataframe for mapping creation, or use current df
    mapping_source = create_mappings_from if create_mappings_from is not None else df

    # Create or reuse mappings from session state for consistency
    if "Agent" in mapping_source.columns:
        if "agent_mapping" not in st.session_state.anonymization_mappings:
            logger.info("Creating agent anonymization mapping...")
            st.session_state.anonymization_mappings["agent_mapping"] = (
                create_anonymous_mapping(
                    mapping_source["Agent"].dropna().unique(), "Agent"
                )
            )
        agent_mapping = st.session_state.anonymization_mappings["agent_mapping"]
        if "Agent" in df.columns and agent_mapping:
            # Use vectorized replace for much faster performance
            # Handle NaN values properly
            mask = df["Agent"].notna()
            df.loc[mask, "Agent"] = (
                df.loc[mask, "Agent"].astype(str).str.strip().replace(agent_mapping)
            )

    if "Company" in mapping_source.columns:
        if "company_mapping" not in st.session_state.anonymization_mappings:
            logger.info("Creating company anonymization mapping...")
            st.session_state.anonymization_mappings["company_mapping"] = (
                create_anonymous_mapping(
                    mapping_source["Company"].dropna().unique(), "Company"
                )
            )
        company_mapping = st.session_state.anonymization_mappings["company_mapping"]
        if "Company" in df.columns and company_mapping:
            # Use vectorized replace for much faster performance
            # Handle NaN values properly
            mask = df["Company"].notna()
            df.loc[mask, "Company"] = (
                df.loc[mask, "Company"].astype(str).str.strip().replace(company_mapping)
            )

    if "Call ID" in mapping_source.columns:
        if "call_id_mapping" not in st.session_state.anonymization_mappings:
            logger.info("Creating call ID anonymization mapping...")
            st.session_state.anonymization_mappings["call_id_mapping"] = (
                create_anonymous_mapping(
                    mapping_source["Call ID"].dropna().unique(), "Call"
                )
            )
        call_id_mapping = st.session_state.anonymization_mappings["call_id_mapping"]
        if "Call ID" in df.columns and call_id_mapping:
            # Use vectorized replace for much faster performance
            # Handle NaN values properly
            mask = df["Call ID"].notna()
            df.loc[mask, "Call ID"] = (
                df.loc[mask, "Call ID"].astype(str).str.strip().replace(call_id_mapping)
            )

    return df


# Convert call_date to datetime if it's not already
if "call_date" in meta_df.columns:
    # If call_date is already datetime, keep it; otherwise try to parse
    if meta_df["call_date"].dtype == "object":
        meta_df["call_date"] = pd.to_datetime(meta_df["call_date"], errors="coerce")


# --- Product Extraction Function ---
def extract_products_from_text(text):
    """
    Extract products from text fields (Summary, Reason, Outcome).
    Uses keyword matching to identify watch brands, jewelry types, and product categories.

    Args:
        text: Text string to extract products from (can be None or empty)

    Returns:
        List of extracted products, or empty list if none found
    """
    if not text or pd.isna(text):
        return []

    text_lower = str(text).lower()
    products = []

    # Watch brands (common luxury and fashion watch brands)
    watch_brands = [
        "rolex",
        "omega",
        "tag heuer",
        "breitling",
        "cartier",
        "patek philippe",
        "audemars piguet",
        "jaeger-lecoultre",
        "vacheron constantin",
        "panerai",
        "iwc",
        "zenith",
        "tudor",
        "longines",
        "tissot",
        "hamilton",
        "seiko",
        "citizen",
        "casio",
        "fossil",
        "michael kors",
        "bulova",
        "movado",
        "baume & mercier",
        "montblanc",
        "hublot",
        "richard mille",
        "ap",
        "patek",
        "jlc",
        "vc",
        "iwc schaffhausen",
        "grand seiko",
    ]

    # Product categories
    product_categories = [
        "watch",
        "watches",
        "timepiece",
        "timepieces",
        "wristwatch",
        "wristwatches",
        "bracelet",
        "bracelets",
        "necklace",
        "necklaces",
        "ring",
        "rings",
        "earrings",
        "pendant",
        "pendants",
        "jewelry",
        "jewellery",
        "accessories",
    ]

    # Check for watch brands
    for brand in watch_brands:
        if brand in text_lower:
            # Normalize brand name (capitalize properly)
            brand_normalized = brand.title()
            if brand == "ap":
                brand_normalized = "Audemars Piguet"
            elif brand == "jlc":
                brand_normalized = "Jaeger-LeCoultre"
            elif brand == "vc":
                brand_normalized = "Vacheron Constantin"
            elif brand == "patek":
                brand_normalized = "Patek Philippe"
            elif brand == "iwc schaffhausen":
                brand_normalized = "IWC"
            elif brand == "grand seiko":
                brand_normalized = "Grand Seiko"

            if brand_normalized not in products:
                products.append(brand_normalized)

    # Check for product categories
    for category in product_categories:
        if category in text_lower:
            category_normalized = (
                category.title() if category != "jewellery" else "Jewelry"
            )
            if category_normalized not in products:
                products.append(category_normalized)

    # Look for SKU patterns (alphanumeric codes, model numbers)
    # Pattern: alphanumeric codes that might be SKUs or model numbers
    sku_pattern = r"\b[A-Z]{2,}\d{3,}\b|\b\d{4,}[A-Z]{2,}\b"
    sku_matches = re.findall(sku_pattern, str(text), re.IGNORECASE)
    for sku in sku_matches:
        if len(sku) >= 6 and sku.upper() not in products:  # Only add substantial codes
            products.append(f"SKU: {sku.upper()}")

    # Look for model numbers (patterns like "Submariner", "Speedmaster", etc.)
    # Common watch model patterns
    model_keywords = [
        "submariner",
        "gmt",
        "daytona",
        "yacht-master",
        "explorer",
        "datejust",
        "speedmaster",
        "seamaster",
        "constellation",
        "de ville",
        "aqua terra",
        "carrera",
        "monaco",
        "aquaracer",
        "formula 1",
        "link",
        "calibre",
        "navitimer",
        "chronomat",
        "superocean",
        "avenger",
        "transocean",
    ]

    for model in model_keywords:
        if model in text_lower:
            model_normalized = model.title().replace("-", " ")
            if model_normalized not in products:
                products.append(f"Model: {model_normalized}")

    return products


# --- Normalize QA fields ---
meta_df.rename(
    columns={
        "company": "Company",
        "agent": "Agent",
        "call_date": "Call Date",
        "date_raw": "Date Raw",
        "time": "Call Time",
        "call_id": "Call ID",
        "qa_score": "QA Score",
        "label": "Label",
        "reason": "Reason",
        "outcome": "Outcome",
        "summary": "Summary",
        "strengths": "Strengths",
        "challenges": "Challenges",
        "coaching_suggestions": "Coaching Suggestions",
        "rubric_details": "Rubric Details",
        "rubric_pass_count": "Rubric Pass Count",
        "rubric_fail_count": "Rubric Fail Count",
    },
    inplace=True,
)


def normalize_categories_in_dataframe(df, column_name):
    """
    Normalize categories in a DataFrame column, handling case-insensitive duplicates.

    This function:
    1. Applies category-specific normalization rules
    2. Merges case-insensitive duplicates (keeps the most common capitalization)

    Args:
        df: DataFrame to normalize
        column_name: Name of the column to normalize

    Returns:
        DataFrame with normalized column
    """
    if column_name not in df.columns:
        return df

    df = df.copy()

    # First, apply category-specific normalization rules
    df[column_name] = df[column_name].apply(normalize_category)

    # Then, handle case-insensitive duplicates
    # Group by lowercase version and use the most common capitalization
    if df[column_name].notna().any():
        # Create a mapping: lowercase -> most common capitalization
        category_counts = df[column_name].value_counts()
        lowercase_to_canonical = {}

        for category in category_counts.index:
            if pd.notna(category):
                cat_lower = str(category).lower()
                # If we haven't seen this lowercase version, or this capitalization is more common
                if cat_lower not in lowercase_to_canonical:
                    lowercase_to_canonical[cat_lower] = category
                else:
                    # Use the one with higher count
                    current_count = category_counts.get(category, 0)
                    existing_cat = lowercase_to_canonical[cat_lower]
                    existing_count = category_counts.get(existing_cat, 0)
                    if current_count > existing_count:
                        lowercase_to_canonical[cat_lower] = category

        # Apply the mapping
        def apply_case_normalization(val):
            if pd.isna(val):
                return val
            val_str = str(val)
            val_lower = val_str.lower()
            if val_lower in lowercase_to_canonical:
                return lowercase_to_canonical[val_lower]
            return val_str

        df[column_name] = df[column_name].apply(apply_case_normalization)

    return df


# Normalize Reason and Outcome columns to merge duplicates
# NOTE: Basic category normalization (shipping merge, refund rename) happens during CSV parsing,
# but case-insensitive duplicate merging requires seeing all values, so it happens here
if "Reason" in meta_df.columns:
    meta_df = normalize_categories_in_dataframe(meta_df, "Reason")
if "Outcome" in meta_df.columns:
    meta_df = normalize_categories_in_dataframe(meta_df, "Outcome")

# Handle call duration
if "speaking_time_per_speaker" in meta_df.columns:

    def compute_speaking_time(row):
        speaking_times = row["speaking_time_per_speaker"]
        if isinstance(speaking_times, dict):
            total = 0
            for t in speaking_times.values():
                if isinstance(t, str) and ":" in t:
                    try:
                        parts = t.split(":")
                        if len(parts) == 2:
                            minutes, seconds = map(int, parts)
                            total += minutes * 60 + seconds
                    except Exception:
                        pass
            return total
        return None

    meta_df["Call Duration (s)"] = meta_df.apply(compute_speaking_time, axis=1)
else:
    meta_df["Call Duration (s)"] = None

meta_df["Call Duration (min)"] = (
    meta_df["Call Duration (s)"] / 60
    if "Call Duration (s)" in meta_df.columns
    else None
)

# Ensure QA Score is numeric
if "QA Score" in meta_df.columns:
    meta_df["QA Score"] = pd.to_numeric(meta_df["QA Score"], errors="coerce")

# Date handling
if ("Call Date" not in meta_df.columns) or meta_df["Call Date"].isna().all():
    if "Date Raw" in meta_df.columns:
        meta_df["Call Date"] = pd.to_datetime(
            meta_df["Date Raw"], format="%m%d%Y", errors="coerce"
        )
    else:
        st.sidebar.error(
            " Neither Call Date nor Date Raw found—cannot parse any dates."
        )
        st.stop()

meta_df.dropna(subset=["Call Date"], inplace=True)

# Normalize agent IDs AFTER column rename (works for both cached and new data)
# This ensures normalization works regardless of whether data came from cache or fresh load
# NOTE: This is a verification pass - normalization should have already happened before DataFrame creation
agent_ids_updated = False
if "Agent" in meta_df.columns:
    # Store original agent IDs to check if any changed
    original_agents = meta_df["Agent"].copy()
    meta_df["Agent"] = meta_df["Agent"].apply(normalize_agent_id)

    # Check if any agent IDs were actually changed
    # Count how many actually changed (not just NaN differences)
    changed_mask = (original_agents != meta_df["Agent"]) & (
        original_agents.notna() | meta_df["Agent"].notna()
    )
    changed_count = changed_mask.sum()

    # VERIFICATION: Check for any non-normalized agent IDs (should not happen after pre-DataFrame normalization)
    non_normalized_mask = meta_df["Agent"].notna() & ~meta_df["Agent"].astype(
        str
    ).str.match(r"^Agent \d+$", na=False)
    non_normalized_count = non_normalized_mask.sum()

    if non_normalized_count > 0:
        logger.warning(
            f" Found {non_normalized_count} non-normalized agent IDs after DataFrame creation. "
            "These should have been normalized earlier. Re-normalizing..."
        )
        # Re-normalize any that slipped through
        for idx in meta_df[non_normalized_mask].index:
            original = meta_df.loc[idx, "Agent"]
            normalized = normalize_agent_id(original)
            meta_df.loc[idx, "Agent"] = normalized
            if original != normalized:
                changed_count += 1
                logger.debug(f" Re-normalized agent ID: {original} -> {normalized}")

    if changed_count > 0:
        agent_ids_updated = True
        logger.info(
            f" Agent IDs normalized - updating cache with corrected agent assignments ({changed_count} calls updated)"
        )

        # Update call_data with normalized agent IDs
        for i, call in enumerate(call_data):
            if i < len(meta_df):
                # Update the agent field in the call data
                if "agent" in call:
                    call["agent"] = meta_df.iloc[i]["Agent"]
                elif "Agent" in call:
                    call["Agent"] = meta_df.iloc[i]["Agent"]
                else:
                    # Add agent field if it doesn't exist
                    call["agent"] = meta_df.iloc[i]["Agent"]

        # Save updated call_data back to cache
        try:
            # Get errors list (empty if not available)
            errors = getattr(st.session_state, "_last_load_errors", [])
            save_cached_data_to_disk(call_data, errors)
            logger.info(
                f" Updated cache with normalized agent IDs ({len(call_data)} calls)"
            )
        except Exception as e:
            logger.warning(f" Failed to update cache with normalized agent IDs: {e}")
            logger.warning(
                " Normalization will still work, but cache won't be updated until next save"
            )

# Apply anonymization if user is anonymous
# Create mappings from full dataset before any filtering
if is_anonymous_user:
    import time

    anonymize_start = time.time()
    logger.info(
        " Anonymous user detected - creating anonymization mappings from full dataset"
    )
    # Create mappings from the full meta_df to ensure consistency
    # Only create mappings if they don't exist (faster on subsequent loads)
    if not st.session_state.anonymization_mappings:
        _ = anonymize_dataframe(meta_df, create_mappings_from=meta_df)
    # Now apply anonymization to meta_df (this is fast with vectorized operations)
    meta_df = anonymize_dataframe(meta_df, create_mappings_from=meta_df)
    anonymize_duration = time.time() - anonymize_start
    logger.info(f" Anonymization completed in {anonymize_duration:.2f}s")
    st.sidebar.warning(
        " **Anonymous Mode**: All identifying information has been anonymized"
    )

# --- Determine if agent view or admin view ---
# user_agent_id is already normalized above (when set from user mapping)

# If user has agent_id, automatically filter to their data
if user_agent_id:
    # Agent view - filter to their calls only
    agent_calls_df = meta_df[meta_df["Agent"] == user_agent_id].copy()

    if agent_calls_df.empty:
        st.warning(f" No calls found for agent: {user_agent_id}")
        st.info(
            "If this is incorrect, please contact your administrator to update your agent ID mapping."
        )
        st.stop()

    # Use agent's data for filtering
    filter_df = agent_calls_df
    show_comparison = True  # Always show comparison for agents
    st.sidebar.info(f" Showing your calls only ({len(agent_calls_df)} calls)")
else:
    # Admin/All data view
    filter_df = meta_df
    show_comparison = False
    st.sidebar.info(f" Showing all data ({len(meta_df)} calls)")

# --- Sidebar Filters ---
st.sidebar.header(" Filter Data")

# Date filter
min_date = filter_df["Call Date"].min()
max_date = filter_df["Call Date"].max()
dates = filter_df["Call Date"].dropna().sort_values().dt.date.unique().tolist()

if not dates:
    st.warning(" No calls with valid dates to display.")
    st.stop()

# Remember last filter settings
if "last_date_preset" not in st.session_state:
    st.session_state.last_date_preset = "All Time"
if "last_date_range" not in st.session_state:
    st.session_state.last_date_range = None
if "last_agents" not in st.session_state:
    st.session_state.last_agents = []
if "last_score_range" not in st.session_state:
    st.session_state.last_score_range = None
if "last_labels" not in st.session_state:
    st.session_state.last_labels = []
if "last_search" not in st.session_state:
    st.session_state.last_search = ""
if "last_preset_filter" not in st.session_state:
    st.session_state.last_preset_filter = "None"
if "selected_rubric_codes" not in st.session_state:
    st.session_state.selected_rubric_codes = []
if "rubric_filter_type" not in st.session_state:
    st.session_state.rubric_filter_type = "Any Status"

# Dark mode toggle (admin only)
if is_regular_admin():
    st.sidebar.markdown("---")
    dark_mode = st.sidebar.toggle(
        "🌙 Dark Mode", value=False, help="Toggle dark mode (requires page refresh)"
    )
    if dark_mode:
        st.markdown(
            """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

# Keyboard shortcuts info
with st.sidebar.expander(" Keyboard Shortcuts"):
    st.markdown("""
    **Navigation:**
    - `Ctrl/Cmd + R` - Refresh page
    - `Ctrl/Cmd + F` - Focus search box
    
    **Tips:**
    - Use filter presets for quick filtering
    - Select multiple calls for batch export
    - Use full-text search across all call details
    """)

preset_option = st.sidebar.selectbox(
    "📆 Date Range",
    options=["All Time", "This Week", "Last 7 Days", "Last 30 Days", "Custom"],
    index=["All Time", "This Week", "Last 7 Days", "Last 30 Days", "Custom"].index(
        st.session_state.last_date_preset
    )
    if st.session_state.last_date_preset
    in ["All Time", "This Week", "Last 7 Days", "Last 30 Days", "Custom"]
    else 0,
)

if preset_option != "Custom":
    today = datetime.today().date()
    if preset_option == "All Time":
        selected_dates = (min(dates), max(dates))
    elif preset_option == "This Week":
        selected_dates = (today - timedelta(days=today.weekday()), today)
    elif preset_option == "Last 7 Days":
        selected_dates = (today - timedelta(days=7), today)
    elif preset_option == "Last 30 Days":
        selected_dates = (today - timedelta(days=30), today)
    st.session_state.last_date_preset = preset_option  # Save selection
else:
    # Restore last custom date range or use default
    default_date_range = (
        st.session_state.last_date_range
        if st.session_state.last_date_range
        and isinstance(st.session_state.last_date_range, tuple)
        else (min(dates), max(dates))
    )
    custom_input = st.sidebar.date_input("Select Date Range", value=default_date_range)
    if isinstance(custom_input, tuple) and len(custom_input) == 2:
        selected_dates = custom_input
        st.session_state.last_date_range = custom_input  # Save selection
    elif isinstance(custom_input, date):
        selected_dates = (custom_input, custom_input)
        st.session_state.last_date_range = selected_dates  # Save selection
    else:
        st.warning(" Please select a valid date range.")
        st.stop()

# Extract start_date and end_date from selected_dates (works for both preset and custom)
start_date, end_date = selected_dates

# Agent filter (only for admin view)
if not user_agent_id:
    available_agents = filter_df["Agent"].dropna().unique().tolist()
    # Restore last selection or use all agents
    default_agents = (
        st.session_state.last_agents
        if st.session_state.last_agents
        and all(a in available_agents for a in st.session_state.last_agents)
        else available_agents
    )
    selected_agents = st.sidebar.multiselect(
        "Select Agents", available_agents, default=default_agents
    )
    st.session_state.last_agents = selected_agents  # Save selection
else:
    # For agents, they only see their own data
    selected_agents = [user_agent_id]

# QA Score filter
if "QA Score" in meta_df.columns and not meta_df["QA Score"].isna().all():
    min_score = float(meta_df["QA Score"].min())
    max_score = float(meta_df["QA Score"].max())
    # Restore last score range or use full range
    default_score_range = (
        st.session_state.last_score_range
        if st.session_state.last_score_range
        and st.session_state.last_score_range[0] >= min_score
        and st.session_state.last_score_range[1] <= max_score
        else (min_score, max_score)
    )
    score_range = st.sidebar.slider(
        " QA Score Range",
        min_value=min_score,
        max_value=max_score,
        value=default_score_range,
        step=1.0,
    )
    st.session_state.last_score_range = score_range  # Save selection
else:
    score_range = None

# Label filter
if "Label" in meta_df.columns:
    available_labels = meta_df["Label"].dropna().unique().tolist()
    # Restore last selection or use all labels
    default_labels = (
        st.session_state.last_labels
        if st.session_state.last_labels
        and all(label in available_labels for label in st.session_state.last_labels)
        else available_labels
    )
    selected_labels = st.sidebar.multiselect(
        " Select Labels", available_labels, default=default_labels
    )
    st.session_state.last_labels = selected_labels  # Save selection
else:
    selected_labels = []

# Filter Presets
st.sidebar.markdown("---")
st.sidebar.markdown("###  Filter Presets")

# Load saved filter presets
if "saved_filter_presets" not in st.session_state:
    st.session_state.saved_filter_presets = {}

# Quick preset filters
preset_options = [
    "None",
    "High Performers (90%+)",
    "Needs Improvement (<70%)",
    "Failed Rubric Items",
    "Recent Issues (Last 7 Days)",
]
preset_index = (
    preset_options.index(st.session_state.last_preset_filter)
    if st.session_state.last_preset_filter in preset_options
    else 0
)
preset_filters = st.sidebar.radio(
    "Quick Filters",
    options=preset_options,
    index=preset_index,
    help="Apply common filter combinations quickly",
)
st.session_state.last_preset_filter = preset_filters  # Save selection

# Saved filter presets management
st.sidebar.markdown("---")
with st.sidebar.expander(" Saved Filter Presets"):
    if st.session_state.saved_filter_presets:
        st.write("**Saved Presets:**")
        for preset_name in st.session_state.saved_filter_presets.keys():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"{preset_name}", key=f"load_{preset_name}"):
                    # Load preset
                    preset = st.session_state.saved_filter_presets[preset_name]
                    st.session_state.last_date_preset = preset.get(
                        "date_preset", "All Time"
                    )
                    st.session_state.last_date_range = preset.get("date_range", None)
                    st.session_state.last_agents = preset.get("agents", [])
                    st.session_state.last_score_range = preset.get("score_range", None)
                    st.session_state.last_labels = preset.get("labels", [])
                    st.session_state.last_search = preset.get("search", "")
                    st.session_state.last_preset_filter = preset.get(
                        "preset_filter", "None"
                    )
                    st.session_state.selected_rubric_codes = preset.get(
                        "rubric_codes", []
                    )
                    st.session_state.rubric_filter_type = preset.get(
                        "rubric_filter_type", "Any Status"
                    )
                    st.rerun()
            with col2:
                if st.button(
                    "", key=f"delete_{preset_name}", help=f"Delete {preset_name}"
                ):
                    del st.session_state.saved_filter_presets[preset_name]
                    st.rerun()

    # Save current filter as preset
    st.markdown("---")
    preset_name = st.text_input(
        "Save current filters as:",
        placeholder="e.g., 'Weekly Review'",
        key="new_preset_name",
    )
    if st.button("Save Preset") and preset_name:
        if preset_name in st.session_state.saved_filter_presets:
            st.warning(f"Preset '{preset_name}' already exists. Overwrite?")
        else:
            # Save current filter state
            current_preset = {
                "date_preset": st.session_state.last_date_preset,
                "date_range": st.session_state.last_date_range,
                "agents": st.session_state.last_agents,
                "score_range": st.session_state.last_score_range,
                "labels": st.session_state.last_labels,
                "search": st.session_state.last_search,
                "preset_filter": st.session_state.last_preset_filter,
                "rubric_codes": st.session_state.get("selected_rubric_codes", []),
                "rubric_filter_type": st.session_state.get(
                    "rubric_filter_type", "Any Status"
                ),
                "created_at": datetime.now().isoformat(),
            }
            st.session_state.saved_filter_presets[preset_name] = current_preset
            st.success(f" Preset '{preset_name}' saved!")
            st.rerun()

# Search functionality - Enhanced full-text search
st.sidebar.markdown("---")
search_text = st.sidebar.text_input(
    " Full-Text Search",
    st.session_state.last_search,
    help="Search across Reason, Summary, Outcome, Strengths, Challenges, and Coaching Suggestions",
)
st.session_state.last_search = search_text  # Save search

# Rubric code filter - Enhanced to support multiple codes (pass/fail/any)
st.sidebar.markdown("---")
st.sidebar.markdown("###  Rubric Code Filters")

# Collect all rubric codes (both pass and fail)
all_rubric_codes = []
if "Rubric Details" in meta_df.columns:
    for idx, row in meta_df.iterrows():
        rubric_details = row.get("Rubric Details", {})
        if isinstance(rubric_details, dict):
            for code, details in rubric_details.items():
                if code not in all_rubric_codes:
                    all_rubric_codes.append(code)

    all_rubric_codes.sort()

    if all_rubric_codes:
        rubric_filter_type = st.sidebar.radio(
            "Filter Type",
            options=["Any Status", "Failed Only", "Passed Only"],
            index=0,
            horizontal=True,
        )

        selected_rubric_codes = st.sidebar.multiselect(
            f"Select Rubric Codes ({rubric_filter_type})",
            options=all_rubric_codes,
            default=st.session_state.selected_rubric_codes
            if st.session_state.selected_rubric_codes
            else [],
            help="Show calls that match these rubric codes",
        )
        st.session_state.selected_rubric_codes = selected_rubric_codes
        st.session_state.rubric_filter_type = rubric_filter_type

        # Also collect failed codes for backward compatibility
        failed_rubric_codes = [code for code in all_rubric_codes]
        selected_failed_codes = (
            selected_rubric_codes if rubric_filter_type == "Failed Only" else []
        )
    else:
        selected_rubric_codes = []
        selected_failed_codes = []
        rubric_filter_type = "Any Status"
else:
    selected_rubric_codes = []
    selected_failed_codes = []
    rubric_filter_type = "Any Status"

# Performance alerts threshold
st.sidebar.markdown("---")
alert_threshold = st.sidebar.slider(
    " Alert Threshold (QA Score)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=5.0,
    help="Agents/calls below this score will be highlighted",
)

# Apply preset filters
if preset_filters == "High Performers (90%+)":
    filter_df = filter_df[filter_df["QA Score"] >= 90].copy()
elif preset_filters == "Needs Improvement (<70%)":
    filter_df = filter_df[filter_df["QA Score"] < 70].copy()
elif preset_filters == "Failed Rubric Items":
    # Filter for calls with any failed rubric items
    if "Rubric Fail Count" in filter_df.columns:
        filter_df = filter_df[filter_df["Rubric Fail Count"] > 0].copy()
elif preset_filters == "Recent Issues (Last 7 Days)":
    seven_days_ago = datetime.today().date() - timedelta(days=7)
    filter_df = filter_df[filter_df["Call Date"].dt.date >= seven_days_ago].copy()
    if "QA Score" in filter_df.columns:
        filter_df = filter_df[filter_df["QA Score"] < 70].copy()

# Apply basic filters
filtered_df = filter_df[
    (filter_df["Agent"].isin(selected_agents))
    & (filter_df["Call Date"].dt.date >= start_date)
    & (filter_df["Call Date"].dt.date <= end_date)
].copy()

# Apply QA Score filter
if score_range:
    filtered_df = filtered_df[
        (filtered_df["QA Score"] >= score_range[0])
        & (filtered_df["QA Score"] <= score_range[1])
    ].copy()

# Apply Label filter
if selected_labels:
    filtered_df = filtered_df[filtered_df["Label"].isin(selected_labels)].copy()

# Apply enhanced full-text search
if search_text:
    search_lower = search_text.lower()
    search_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)

    # Search across multiple fields
    search_fields = [
        "Reason",
        "Summary",
        "Outcome",
        "Strengths",
        "Challenges",
        "Coaching Suggestions",
    ]
    for field in search_fields:
        if field in filtered_df.columns:
            field_mask = (
                filtered_df[field]
                .astype(str)
                .str.lower()
                .str.contains(search_lower, na=False)
            )
            search_mask = search_mask | field_mask

    filtered_df = filtered_df[search_mask].copy()

# Apply enhanced rubric code filter
if selected_rubric_codes:
    rubric_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)

    for idx, row in filtered_df.iterrows():
        rubric_details = row.get("Rubric Details", {})
        if isinstance(rubric_details, dict):
            for code in selected_rubric_codes:
                if code in rubric_details:
                    details = rubric_details[code]
                    if isinstance(details, dict):
                        status = details.get("status", "").lower()
                        if rubric_filter_type == "Any Status":
                            rubric_mask[idx] = True
                            break
                        elif rubric_filter_type == "Failed Only" and status == "fail":
                            rubric_mask[idx] = True
                            break
                        elif rubric_filter_type == "Passed Only" and status == "pass":
                            rubric_mask[idx] = True
                            break

    filtered_df = filtered_df[rubric_mask].copy()

# Apply legacy failed rubric codes filter (for backward compatibility)
if selected_failed_codes:
    failed_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)

    for idx, row in filtered_df.iterrows():
        rubric_details = row.get("Rubric Details", {})
        if isinstance(rubric_details, dict):
            for code in selected_failed_codes:
                if code in rubric_details:
                    details = rubric_details[code]
                    if isinstance(details, dict) and details.get("status") == "Fail":
                        failed_mask[idx] = True
                        break

    filtered_df = filtered_df[failed_mask].copy()

# Calculate overall averages for comparison (from all data, not filtered)
if show_comparison and user_agent_id:
    # Get overall averages from all agents' data in the same date range
    overall_df = meta_df[
        (meta_df["Call Date"].dt.date >= start_date)
        & (meta_df["Call Date"].dt.date <= end_date)
    ].copy()

    # Calculate averages per agent first, then average those agent averages
    # This gives equal weight to each agent rather than weighting by call volume
    # Exclude the current user's agent from peer comparison
    agent_metrics = []

    for agent in overall_df["Agent"].unique():
        # Skip if this is the current user's agent (we want peer average)
        if agent == user_agent_id:
            continue

        agent_data = overall_df[overall_df["Agent"] == agent]
        agent_metric = {}

        # Average QA Score per agent
        if "QA Score" in agent_data.columns:
            agent_avg_score = agent_data["QA Score"].mean()
            if not pd.isna(agent_avg_score):
                agent_metric["avg_score"] = agent_avg_score

        # Pass rate per agent
        if (
            "Rubric Pass Count" in agent_data.columns
            and "Rubric Fail Count" in agent_data.columns
        ):
            total_pass = agent_data["Rubric Pass Count"].sum()
            total_fail = agent_data["Rubric Fail Count"].sum()
            if (total_pass + total_fail) > 0:
                agent_metric["pass_rate"] = total_pass / (total_pass + total_fail) * 100

        # Average AHT per agent
        if "Call Duration (min)" in agent_data.columns:
            aht_values = agent_data["Call Duration (min)"].dropna()
            if len(aht_values) > 0:
                agent_avg_aht = aht_values.mean()
                if not pd.isna(agent_avg_aht):
                    agent_metric["avg_aht"] = agent_avg_aht

        if agent_metric:
            agent_metrics.append(agent_metric)

    # Now average the agent averages
    if agent_metrics:
        # Average QA Score (average of agent averages)
        scores = [m.get("avg_score") for m in agent_metrics if "avg_score" in m]
        overall_avg_score = sum(scores) / len(scores) if scores else 0

        # Average Pass Rate (average of agent pass rates)
        pass_rates = [m.get("pass_rate") for m in agent_metrics if "pass_rate" in m]
        overall_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0

        # Average AHT (average of agent averages)
        ahts = [m.get("avg_aht") for m in agent_metrics if "avg_aht" in m]
        overall_avg_aht = sum(ahts) / len(ahts) if ahts else None
    else:
        overall_avg_score = 0
        overall_pass_rate = 0
        overall_avg_aht = None

    overall_total_calls = len(overall_df)
else:
    overall_avg_score = None
    overall_pass_rate = None
    overall_avg_aht = None
    overall_total_calls = None

if score_range:
    filtered_df = filtered_df[
        (filtered_df["QA Score"] >= score_range[0])
        & (filtered_df["QA Score"] <= score_range[1])
    ]

if selected_labels:
    filtered_df = filtered_df[filtered_df["Label"].isin(selected_labels)]

# Search filter
if search_text:
    search_lower = search_text.lower()
    mask = (
        filtered_df["Reason"]
        .astype(str)
        .str.lower()
        .str.contains(search_lower, na=False)
        | filtered_df["Summary"]
        .astype(str)
        .str.lower()
        .str.contains(search_lower, na=False)
        | filtered_df["Outcome"]
        .astype(str)
        .str.lower()
        .str.contains(search_lower, na=False)
    )
    filtered_df = filtered_df[mask]

# Failed rubric code filter
if selected_failed_codes:

    def has_failed_code(row, codes):
        rubric_details = row.get("Rubric Details", {})
        if isinstance(rubric_details, dict):
            for code in codes:
                if code in rubric_details:
                    details = rubric_details[code]
                    if isinstance(details, dict) and details.get("status") == "Fail":
                        return True
        return False

    filtered_df = filtered_df[
        filtered_df.apply(
            lambda row: has_failed_code(row, selected_failed_codes), axis=1
        )
    ]

if filtered_df.empty:
    st.warning(" No data matches the current filter selection.")
    st.stop()

# --- Main Dashboard ---
if user_agent_id:
    st.title(f" My QA Performance Dashboard - {user_agent_id}")
else:
    st.title(" QA Rubric Dashboard")

# Monitoring Dashboard (Chloe, Shannon, and Jerson only)
if is_super_admin():
    with st.expander(
        "System Monitoring & Metrics",
        expanded=False,
    ):
        st.markdown("### Usage Metrics")
        metrics = load_metrics()

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Sessions", metrics.get("sessions", 0))
        with metric_col2:
            total_errors = sum(
                e.get("count", 0) for e in metrics.get("errors", {}).values()
            )
            st.metric("Total Errors", total_errors)
        with metric_col3:
            unique_features = len(metrics.get("features_used", {}))
            st.metric("Features Used", unique_features)

        # Show repeated failures
        repeated_failures = {
            k: v for k, v in metrics.get("errors", {}).items() if v.get("count", 0) >= 5
        }
        if repeated_failures:
            st.warning(f" **{len(repeated_failures)} repeated failure(s) detected:**")
            for error_key, error_data in sorted(
                repeated_failures.items(), key=lambda x: x[1]["count"], reverse=True
            ):
                st.markdown(
                    f"- **{error_key}**: {error_data['count']} occurrences (First: {error_data.get('first_seen', 'N/A')}, Last: {error_data.get('last_seen', 'N/A')})"
                )

        # Show feature usage
        if metrics.get("features_used"):
            st.markdown("### Feature Usage")
            feature_df = pd.DataFrame(
                [
                    {"Feature": k, "Usage Count": v}
                    for k, v in sorted(
                        metrics.get("features_used", {}).items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ]
            )
            st.dataframe(feature_df, hide_index=True)

        # Show recent errors
        if metrics.get("errors"):
            st.markdown("### Recent Errors")
            error_df = pd.DataFrame(
                [
                    {
                        "Error": k.split(":")[0],
                        "Message": k.split(":", 1)[1] if ":" in k else k,
                        "Count": v.get("count", 0),
                        "Last Seen": v.get("last_seen", "N/A"),
                    }
                    for k, v in sorted(
                        metrics.get("errors", {}).items(),
                        key=lambda x: x[1].get("last_seen", ""),
                        reverse=True,
                    )[:10]
                ]
            )
            st.dataframe(error_df, hide_index=True)

        if st.button("Refresh Metrics"):
            st.rerun()

        # Audit Log Viewer (Chloe, Shannon, and Jerson only)
        if current_username and current_username.lower() in [
            "chloe",
            "shannon",
            "jerson",
        ]:
            st.markdown("---")
            st.markdown("###  Audit Log")
            audit_file = Path("logs/audit_log.json")
            if audit_file.exists():
                try:
                    with open(audit_file, "r") as f:
                        audit_log = json.load(f)

                    if audit_log:
                        # Show recent audit entries
                        st.write(f"**Total audit entries:** {len(audit_log)}")
                        recent_entries = audit_log[-50:]  # Last 50 entries
                        audit_df = pd.DataFrame(recent_entries)
                        audit_df["timestamp"] = pd.to_datetime(audit_df["timestamp"])
                        audit_df = audit_df.sort_values("timestamp", ascending=False)
                        st.dataframe(audit_df, hide_index=True)

                        # Filter by action type
                        action_types = audit_df["action"].unique().tolist()
                        selected_action = st.selectbox(
                            "Filter by action:", ["All"] + action_types
                        )
                        if selected_action != "All":
                            filtered_audit = audit_df[
                                audit_df["action"] == selected_action
                            ]
                            st.dataframe(filtered_audit, hide_index=True)
                    else:
                        st.info("No audit entries yet.")
                except Exception as e:
                    st.error(f"Error loading audit log: {e}")
            else:
                st.info(
                    "Audit log file not found. Audit entries will be created as you use the system."
                )

# Data Validation Dashboard (Chloe, Shannon, and Jerson only)
if is_super_admin():
    with st.expander("Data Quality Validation", expanded=False):
        st.markdown("### Data Quality Metrics")

        validation_issues = []
        validation_stats = {}

        # Check for missing required fields
        required_fields = ["Agent", "Call Date", "QA Score", "Label"]
        for field in required_fields:
            if field in meta_df.columns:
                missing_count = meta_df[field].isna().sum()
                total_count = len(meta_df)
                missing_pct = (
                    (missing_count / total_count * 100) if total_count > 0 else 0
                )
                validation_stats[field] = {
                    "missing": missing_count,
                    "total": total_count,
                    "pct": missing_pct,
                }
                if missing_count > 0:
                    validation_issues.append(
                        f"**{field}**: {missing_count} missing ({missing_pct:.1f}%)"
                    )

        # Check for invalid QA scores
        if "QA Score" in meta_df.columns:
            invalid_scores = meta_df[
                (meta_df["QA Score"] < 0) | (meta_df["QA Score"] > 100)
            ]
            invalid_count = len(invalid_scores)
            if invalid_count > 0:
                validation_issues.append(
                    f"**QA Score**: {invalid_count} invalid scores (outside 0-100%)"
                )
            validation_stats["Invalid QA Scores"] = invalid_count

        # Check for missing rubric details
        if "Rubric Details" in meta_df.columns:
            missing_rubric = meta_df["Rubric Details"].isna().sum()
            if missing_rubric > 0:
                validation_issues.append(
                    f"**Rubric Details**: {missing_rubric} calls missing rubric data"
                )
            validation_stats["Missing Rubric"] = missing_rubric

        # Check for duplicate call IDs
        if "Call ID" in meta_df.columns:
            duplicates = meta_df[meta_df["Call ID"].duplicated(keep=False)]
            duplicate_count = len(duplicates)
            if duplicate_count > 0:
                validation_issues.append(
                    f"**Call ID**: {duplicate_count} duplicate call IDs found"
                )
            validation_stats["Duplicate Call IDs"] = duplicate_count

        # Display validation results
        if validation_issues:
            st.warning(f" Found {len(validation_issues)} data quality issue(s):")
            for issue in validation_issues:
                st.markdown(f"- {issue}")
        else:
            st.success(" No data quality issues detected!")

        # Show detailed stats table
        if validation_stats:
            st.markdown("### Detailed Statistics")
            stats_df = pd.DataFrame(
                [
                    {
                        "Field": k,
                        "Missing/Invalid Count": v.get("missing", v)
                        if isinstance(v, dict)
                        else v,
                        "Total": str(v.get("total", "N/A"))
                        if isinstance(v, dict)
                        else "N/A",
                        "Percentage": f"{v.get('pct', 0):.1f}%"
                        if isinstance(v, dict)
                        else "N/A",
                    }
                    for k, v in validation_stats.items()
                ]
            )
            st.dataframe(stats_df, hide_index=True)

# Summary Metrics
if show_comparison and user_agent_id:
    # Agent view with comparison
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        my_calls = len(filtered_df)
        st.metric(
            "My Calls",
            my_calls,
        )

    with col2:
        my_avg_score = (
            filtered_df["QA Score"].mean() if "QA Score" in filtered_df.columns else 0
        )
        delta_score = my_avg_score - overall_avg_score if overall_avg_score else None
        st.metric(
            "My Avg Score",
            f"{my_avg_score:.1f}%",
            delta=f"{delta_score:+.1f}%" if delta_score is not None else None,
            delta_color="normal" if delta_score and delta_score >= 0 else "inverse",
        )

    with col3:
        my_total_pass = (
            filtered_df["Rubric Pass Count"].sum()
            if "Rubric Pass Count" in filtered_df.columns
            else 0
        )
        my_total_fail = (
            filtered_df["Rubric Fail Count"].sum()
            if "Rubric Fail Count" in filtered_df.columns
            else 0
        )
        my_pass_rate = (
            (my_total_pass / (my_total_pass + my_total_fail) * 100)
            if (my_total_pass + my_total_fail) > 0
            else 0
        )
        delta_pass = my_pass_rate - overall_pass_rate if overall_pass_rate else None
        st.metric(
            "My Pass Rate",
            f"{my_pass_rate:.1f}%",
            delta=f"{delta_pass:+.1f}%" if delta_pass is not None else None,
            delta_color="normal" if delta_pass and delta_pass >= 0 else "inverse",
        )

    with col4:
        my_avg_aht = (
            filtered_df["Call Duration (min)"].mean()
            if "Call Duration (min)" in filtered_df.columns
            and filtered_df["Call Duration (min)"].notna().any()
            else None
        )
        # overall_avg_aht is already calculated above (average of agent averages, excluding current agent)
        delta_aht = (
            my_avg_aht - overall_avg_aht
            if my_avg_aht is not None and overall_avg_aht is not None
            else None
        )
        delta_value = f"{delta_aht:+.1f} min" if delta_aht is not None else None
        delta_color_value = None
        if delta_aht is not None:
            delta_color_value = "normal" if delta_aht < 0 else "inverse"

        st.metric(
            "My Avg AHT",
            f"{my_avg_aht:.1f} min" if my_avg_aht is not None else "N/A",
            delta=delta_value,
            delta_color=delta_color_value,
        )

    with col5:
        st.metric(
            "Overall Avg Score",
            f"{overall_avg_score:.1f}%" if overall_avg_score else "N/A",
        )

    with col6:
        st.metric(
            "Overall Pass Rate",
            f"{overall_pass_rate:.1f}%" if overall_pass_rate else "N/A",
        )

    # Comparison section
    st.subheader("My Performance vs. Team Average")
    comp_col1, comp_col2, comp_col3 = st.columns(3)

    with comp_col1:
        st.write("**QA Score Comparison**")
        comparison_data = pd.DataFrame(
            {
                "Metric": ["My Average", "Team Average"],
                "QA Score": [my_avg_score, overall_avg_score],
            }
        )
        fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
        comparison_data.plot(
            x="Metric",
            y="QA Score",
            kind="bar",
            ax=ax_comp,
            color=["steelblue", "orange"],
        )
        ax_comp.set_ylabel("QA Score (%)")
        ax_comp.set_title("My Score vs Team Average")
        ax_comp.axhline(
            y=alert_threshold,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Threshold ({alert_threshold}%)",
        )
        ax_comp.legend()
        plt.xticks(rotation=0)
        plt.tight_layout()
        st_pyplot_safe(fig_comp)

    with comp_col2:
        st.write("**Pass Rate Comparison**")
        pass_comparison = pd.DataFrame(
            {
                "Metric": ["My Pass Rate", "Team Pass Rate"],
                "Pass Rate": [my_pass_rate, overall_pass_rate],
            }
        )
        fig_pass, ax_pass = plt.subplots(figsize=(8, 5))
        pass_comparison.plot(
            x="Metric",
            y="Pass Rate",
            kind="bar",
            ax=ax_pass,
            color=["green", "lightgreen"],
        )
        ax_pass.set_ylabel("Pass Rate (%)")
        ax_pass.set_title("My Pass Rate vs Team Average")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st_pyplot_safe(fig_pass)

    with comp_col3:
        st.write("**AHT Comparison**")
        if my_avg_aht is not None and overall_avg_aht is not None:
            aht_comparison = pd.DataFrame(
                {
                    "Metric": ["My Avg AHT", "Team Avg AHT"],
                    "AHT (min)": [my_avg_aht, overall_avg_aht],
                }
            )
            fig_aht, ax_aht = plt.subplots(figsize=(8, 5))
            aht_comparison.plot(
                x="Metric",
                y="AHT (min)",
                kind="bar",
                ax=ax_aht,
                color=["purple", "lavender"],
            )
            ax_aht.set_ylabel("Average Handle Time (min)")
            ax_aht.set_title("My AHT vs Team Average")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st_pyplot_safe(fig_aht)
        else:
            st.info("AHT data not available")

else:
    # Admin/All data view
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Calls", len(filtered_df))
    with col2:
        avg_score = (
            filtered_df["QA Score"].mean() if "QA Score" in filtered_df.columns else 0
        )
        st.metric("Avg QA Score", f"{avg_score:.1f}%" if avg_score else "N/A")
    with col3:
        total_pass = (
            filtered_df["Rubric Pass Count"].sum()
            if "Rubric Pass Count" in filtered_df.columns
            else 0
        )
        total_fail = (
            filtered_df["Rubric Fail Count"].sum()
            if "Rubric Fail Count" in filtered_df.columns
            else 0
        )
        pass_rate = (
            (total_pass / (total_pass + total_fail) * 100)
            if (total_pass + total_fail) > 0
            else 0
        )
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col4:
        avg_aht = (
            filtered_df["Call Duration (min)"].mean()
            if "Call Duration (min)" in filtered_df.columns
            and filtered_df["Call Duration (min)"].notna().any()
            else None
        )
        st.metric(
            "Avg AHT",
            f"{avg_aht:.1f} min" if avg_aht is not None else "N/A",
        )
    with col5:
        st.metric("Agents", filtered_df["Agent"].nunique())

# --- Historical Baseline Comparisons (Benchmarking) ---
with st.expander(" Historical Baseline Comparisons", expanded=False):
    st.markdown("### Compare Current Performance to Historical Baselines")

    # Calculate baselines - convert date objects to pandas Timestamp for consistent comparison
    start_date_dt = (
        pd.Timestamp(start_date)
        if not isinstance(start_date, pd.Timestamp)
        else start_date
    )
    end_date_dt = (
        pd.Timestamp(end_date) if not isinstance(end_date, pd.Timestamp) else end_date
    )

    # Calculate baselines
    baselines = calculate_historical_baselines(meta_df, start_date_dt, end_date_dt)

    if baselines:
        current_avg_score = (
            filtered_df["QA Score"].mean()
            if "QA Score" in filtered_df.columns
            else None
        )
        current_pass_rate = calculate_pass_rate(filtered_df)

        baseline_col1, baseline_col2, baseline_col3 = st.columns(3)

        # Last 30 days comparison
        if "last_30_days" in baselines:
            with baseline_col1:
                baseline_30 = baselines["last_30_days"]
                if current_avg_score and baseline_30["avg_score"]:
                    score_change_30 = current_avg_score - baseline_30["avg_score"]
                    st.metric(
                        "vs Last 30 Days",
                        f"{current_avg_score:.1f}%",
                        delta=f"{score_change_30:+.1f}%",
                        delta_color="normal" if score_change_30 >= 0 else "inverse",
                        help=f"Baseline: {baseline_30['avg_score']:.1f}%",
                    )

        # Last 90 days comparison
        if "last_90_days" in baselines:
            with baseline_col2:
                baseline_90 = baselines["last_90_days"]
                if current_avg_score and baseline_90["avg_score"]:
                    score_change_90 = current_avg_score - baseline_90["avg_score"]
                    st.metric(
                        "vs Last 90 Days",
                        f"{current_avg_score:.1f}%",
                        delta=f"{score_change_90:+.1f}%",
                        delta_color="normal" if score_change_90 >= 0 else "inverse",
                        help=f"Baseline: {baseline_90['avg_score']:.1f}%",
                    )

        # Year-over-year comparison
        if "year_over_year" in baselines:
            with baseline_col3:
                baseline_yoy = baselines["year_over_year"]
                if current_avg_score and baseline_yoy["avg_score"]:
                    score_change_yoy = current_avg_score - baseline_yoy["avg_score"]
                    st.metric(
                        "vs Same Period Last Year",
                        f"{current_avg_score:.1f}%",
                        delta=f"{score_change_yoy:+.1f}%",
                        delta_color="normal" if score_change_yoy >= 0 else "inverse",
                        help=f"Baseline: {baseline_yoy['avg_score']:.1f}%",
                    )
    else:
        st.info(" Insufficient historical data for baseline comparisons")

    # Benchmark visualization chart
    if baselines and current_avg_score:
        st.markdown("### Benchmark Comparison Chart")
        baseline_names = []
        baseline_scores = []

        if "last_30_days" in baselines and baselines["last_30_days"]["avg_score"]:
            baseline_names.append("Last 30 Days")
            baseline_scores.append(baselines["last_30_days"]["avg_score"])

        if "last_90_days" in baselines and baselines["last_90_days"]["avg_score"]:
            baseline_names.append("Last 90 Days")
            baseline_scores.append(baselines["last_90_days"]["avg_score"])

        if "year_over_year" in baselines and baselines["year_over_year"]["avg_score"]:
            baseline_names.append("Same Period Last Year")
            baseline_scores.append(baselines["year_over_year"]["avg_score"])

        if baseline_names:
            baseline_names.append("Current Period")
            baseline_scores.append(current_avg_score)

            fig_bench, ax_bench = plt.subplots(figsize=(8, 5))
            colors = [
                "steelblue" if i < len(baseline_names) - 1 else "orange"
                for i in range(len(baseline_names))
            ]
            bars = ax_bench.bar(baseline_names, baseline_scores, color=colors)
            ax_bench.set_ylabel("Average QA Score (%)")
            ax_bench.set_title("Current Performance vs Historical Baselines")
            ax_bench.axhline(
                y=alert_threshold,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Threshold ({alert_threshold}%)",
            )
            ax_bench.legend()
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st_pyplot_safe(fig_bench)

# --- Agent Leaderboard ---
if not user_agent_id:
    # Admin view - show all agents
    st.subheader("Agent Leaderboard")

    # Ensure Agent column is normalized (safety check)
    if "Agent" in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df["Agent"] = filtered_df["Agent"].apply(normalize_agent_id)

    agent_performance = (
        filtered_df.groupby("Agent")
        .agg(
            Total_Calls=("Call ID", "count"),
            Avg_QA_Score=("QA Score", "mean"),
            Total_Pass=("Rubric Pass Count", "sum"),
            Total_Fail=("Rubric Fail Count", "sum"),
            Avg_Call_Duration=("Call Duration (min)", "mean"),
        )
        .reset_index()
    )

    # Calculate pass rate
    agent_performance["Pass_Rate"] = (
        agent_performance["Total_Pass"]
        / (agent_performance["Total_Pass"] + agent_performance["Total_Fail"])
        * 100
    ).fillna(0)

    # Add percentile rankings
    percentile_rankings = calculate_percentile_rankings(filtered_df, "QA Score")
    if not percentile_rankings.empty:
        agent_performance = agent_performance.merge(
            percentile_rankings[["Agent", "percentile"]], on="Agent", how="left"
        )
        agent_performance["percentile"] = agent_performance["percentile"].fillna(0)

        # Add percentile badges
        def get_percentile_badge(pct):
            if pct >= 90:
                return " Top 10%"
            elif pct >= 75:
                return " Top 25%"
            elif pct >= 50:
                return " Top 50%"
            elif pct >= 25:
                return " Bottom 50%"
            else:
                return " Bottom 25%"

        agent_performance["Percentile_Rank"] = agent_performance["percentile"].apply(
            get_percentile_badge
        )
        agent_performance = agent_performance.sort_values(
            "Avg_QA_Score", ascending=False
        )

        # Display with percentile column
        display_cols = [
            "Agent",
            "Total_Calls",
            "Avg_QA_Score",
            "Pass_Rate",
            "Percentile_Rank",
            "Avg_Call_Duration",
        ]
        st.dataframe(agent_performance[display_cols].round(1), hide_index=True)
    else:
        agent_performance = agent_performance.sort_values(
            "Avg_QA_Score", ascending=False
        )
        st.dataframe(agent_performance.round(1), hide_index=True)

# --- At-Risk Agent Detection (Predictive Analytics) ---
if not user_agent_id:  # Admin view only
    with st.expander("At-Risk Agent Detection", expanded=False):
        st.markdown("### Early Warning System for Agents at Risk")
        st.caption(
            "Identifies agents who may drop below threshold based on recent trends, volatility, and proximity to threshold"
        )

        at_risk_agents = identify_at_risk_agents(filtered_df, threshold=alert_threshold)

        if at_risk_agents:
            st.warning(f" Found {len(at_risk_agents)} agent(s) at risk")

            risk_data = []
            for agent_info in at_risk_agents:
                risk_data.append(
                    {
                        "Agent": agent_info["agent"],
                        "Risk Score": f"{agent_info['risk_score']:.0f}/100",
                        "Recent Avg Score": f"{agent_info['recent_avg']:.1f}%",
                        "Trend": " Declining"
                        if agent_info["trend_slope"] < 0
                        else " Improving",
                        "Volatility": f"{agent_info['volatility']:.1f}",
                        "Distance to Threshold": f"{agent_info['proximity_to_threshold']:.1f}%",
                        "Recent Calls": agent_info["recent_calls"],
                    }
                )

            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, hide_index=True)

            # Show risk factors for top at-risk agent
            if at_risk_agents:
                top_risk = at_risk_agents[0]
                st.markdown(f"**Why is {top_risk['agent']} at risk?**")
                risk_factors = []
                if top_risk["trend_slope"] < -0.5:
                    risk_factors.append(
                        f" Declining trend (slope: {top_risk['trend_slope']:.2f})"
                    )
                if top_risk["volatility"] > 10:
                    risk_factors.append(
                        f" High volatility ({top_risk['volatility']:.1f})"
                    )
                if top_risk["proximity_to_threshold"] <= 10:
                    risk_factors.append(
                        f" Close to threshold ({top_risk['proximity_to_threshold']:.1f}% away)"
                    )

                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"- {factor}")
        else:
            st.success(" No agents currently identified as at risk")
else:
    # Agent view - show only their performance summary
    st.subheader("My Performance Summary")
    my_performance = (
        filtered_df.groupby("Agent")
        .agg(
            Total_Calls=("Call ID", "count"),
            Avg_QA_Score=("QA Score", "mean"),
            Total_Pass=("Rubric Pass Count", "sum"),
            Total_Fail=("Rubric Fail Count", "sum"),
            Avg_Call_Duration=("Call Duration (min)", "mean"),
        )
        .reset_index()
    )

    my_performance["Pass_Rate"] = (
        my_performance["Total_Pass"]
        / (my_performance["Total_Pass"] + my_performance["Total_Fail"])
        * 100
    ).fillna(0)

    # Create comparison table
    comparison_table = pd.DataFrame(
        {
            "Metric": [
                "Total Calls",
                "Avg QA Score",
                "Pass Rate",
                "Avg Call Duration (min)",
            ],
            "My Performance": [
                str(int(my_performance["Total_Calls"].iloc[0])),
                f"{my_performance['Avg_QA_Score'].iloc[0]:.1f}%",
                f"{my_performance['Pass_Rate'].iloc[0]:.1f}%",
                f"{my_performance['Avg_Call_Duration'].iloc[0]:.1f}"
                if not pd.isna(my_performance["Avg_Call_Duration"].iloc[0])
                else "N/A",
            ],
            "Team Average": [
                str(overall_total_calls) if overall_total_calls else "0",
                f"{overall_avg_score:.1f}%" if overall_avg_score else "0.0%",
                f"{overall_pass_rate:.1f}%" if overall_pass_rate else "0.0%",
                f"{overall_avg_aht:.1f}" if overall_avg_aht is not None else "N/A",
            ],
        }
    )

    st.dataframe(comparison_table, hide_index=True)

# --- Call Reason & Outcome Analysis ---
with st.expander("Call Reason & Outcome Analysis", expanded=False):
    st.subheader("Call Reason & Outcome Analysis")
    if (
        "Reason" in filtered_df.columns
        or "Outcome" in filtered_df.columns
        or "Summary" in filtered_df.columns
    ):
        reason_tab1, reason_tab2, reason_tab3 = st.tabs(
            ["Reasons", "Outcomes", "Products"]
        )

        with reason_tab1:
            if "Reason" in filtered_df.columns:
                reason_col1, reason_col2 = st.columns(2)

                with reason_col1:
                    st.write("**Most Common Call Reasons**")
                    reason_counts = filtered_df["Reason"].value_counts().head(10)
                    if len(reason_counts) > 0:
                        fig_reason, ax_reason = plt.subplots(figsize=(8, 6))
                        reason_counts.plot(kind="barh", ax=ax_reason, color="steelblue")
                        ax_reason.set_xlabel("Number of Calls")
                        ax_reason.set_title("Top 10 Call Reasons")
                        plt.tight_layout()
                        st_pyplot_safe(fig_reason)

                with reason_col2:
                    st.write("**Reason Distribution**")
                    reason_counts_all = filtered_df["Reason"].value_counts()
                    if len(reason_counts_all) > 0:
                        # Show top 10 in pie chart, rest as "Other"
                        top_reasons = reason_counts_all.head(10)
                        other_count = (
                            reason_counts_all.iloc[10:].sum()
                            if len(reason_counts_all) > 10
                            else 0
                        )

                        if other_count > 0:
                            pie_data = pd.concat(
                                [top_reasons, pd.Series({"Other": other_count})]
                            )
                        else:
                            pie_data = top_reasons

                        fig_reason_pie, ax_reason_pie = plt.subplots(figsize=(8, 6))
                        wedges, texts, autotexts = ax_reason_pie.pie(
                            pie_data.values,
                            labels=None,  # Remove labels from pie chart
                            autopct="%1.1f%%",
                            startangle=90,
                        )
                        ax_reason_pie.set_title("Call Reasons Distribution")
                        # Add legend with labels
                        ax_reason_pie.legend(
                            wedges,
                            pie_data.index,
                            title="Call Reasons",
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1),
                        )
                        plt.tight_layout()
                        st_pyplot_safe(fig_reason_pie)

                # Trend over time
                if "Call Date" in filtered_df.columns and len(filtered_df) > 0:
                    st.write("**Reason Trends Over Time**")
                    # Get top 5 reasons for trend
                    top_5_reasons = (
                        filtered_df["Reason"].value_counts().head(5).index.tolist()
                    )
                    filtered_df_reason = filtered_df[
                        filtered_df["Reason"].isin(top_5_reasons)
                    ].copy()
                    filtered_df_reason["Call Date"] = pd.to_datetime(
                        filtered_df_reason["Call Date"], errors="coerce"
                    )
                    filtered_df_reason = filtered_df_reason.dropna(subset=["Call Date"])

                    if len(filtered_df_reason) > 0:
                        # Group by date and reason
                        reason_trend = (
                            filtered_df_reason.groupby(
                                [filtered_df_reason["Call Date"].dt.date, "Reason"]
                            )
                            .size()
                            .reset_index()
                        )
                        reason_trend.columns = ["Date", "Reason", "Count"]
                        reason_trend_pivot = reason_trend.pivot(
                            index="Date", columns="Reason", values="Count"
                        ).fillna(0)

                        fig_reason_trend, ax_reason_trend = plt.subplots(
                            figsize=(12, 6)
                        )
                        for reason in top_5_reasons:
                            if reason in reason_trend_pivot.columns:
                                ax_reason_trend.plot(
                                    reason_trend_pivot.index,
                                    reason_trend_pivot[reason],
                                    marker="o",
                                    label=reason,
                                    linewidth=2,
                                )
                        ax_reason_trend.set_xlabel("Date")
                        ax_reason_trend.set_ylabel("Number of Calls")
                        ax_reason_trend.set_title("Top 5 Call Reasons Over Time")
                        ax_reason_trend.legend()
                        ax_reason_trend.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st_pyplot_safe(fig_reason_trend)

        with reason_tab2:
            if "Outcome" in filtered_df.columns:
                outcome_col1, outcome_col2 = st.columns(2)

                with outcome_col1:
                    st.write("**Most Common Outcomes**")
                    outcome_counts = filtered_df["Outcome"].value_counts().head(10)
                    if len(outcome_counts) > 0:
                        fig_outcome, ax_outcome = plt.subplots(figsize=(8, 6))
                        outcome_counts.plot(kind="barh", ax=ax_outcome, color="green")
                        ax_outcome.set_xlabel("Number of Calls")
                        ax_outcome.set_title("Top 10 Outcomes")
                        plt.tight_layout()
                        st_pyplot_safe(fig_outcome)

                with outcome_col2:
                    st.write("**Outcome Distribution**")
                    outcome_counts_all = filtered_df["Outcome"].value_counts()
                    if len(outcome_counts_all) > 0:
                        # Show top 10 in pie chart, rest as "Other"
                        top_outcomes = outcome_counts_all.head(10)
                        other_count = (
                            outcome_counts_all.iloc[10:].sum()
                            if len(outcome_counts_all) > 10
                            else 0
                        )

                        if other_count > 0:
                            pie_data = pd.concat(
                                [top_outcomes, pd.Series({"Other": other_count})]
                            )
                        else:
                            pie_data = top_outcomes

                        fig_outcome_pie, ax_outcome_pie = plt.subplots(figsize=(8, 6))
                        wedges, texts, autotexts = ax_outcome_pie.pie(
                            pie_data.values,
                            labels=None,  # Remove labels from pie chart
                            autopct="%1.1f%%",
                            startangle=90,
                        )
                        ax_outcome_pie.set_title("Outcomes Distribution")
                        # Add legend with labels
                        ax_outcome_pie.legend(
                            wedges,
                            pie_data.index,
                            title="Call Outcomes",
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1),
                        )
                        plt.tight_layout()
                        st_pyplot_safe(fig_outcome_pie)

                # Trend over time
                if "Call Date" in filtered_df.columns and len(filtered_df) > 0:
                    st.write("**Outcome Trends Over Time**")
                    # Get top 5 outcomes for trend
                    top_5_outcomes = (
                        filtered_df["Outcome"].value_counts().head(5).index.tolist()
                    )
                    filtered_df_outcome = filtered_df[
                        filtered_df["Outcome"].isin(top_5_outcomes)
                    ].copy()
                    filtered_df_outcome["Call Date"] = pd.to_datetime(
                        filtered_df_outcome["Call Date"], errors="coerce"
                    )
                    filtered_df_outcome = filtered_df_outcome.dropna(
                        subset=["Call Date"]
                    )

                    if len(filtered_df_outcome) > 0:
                        # Group by date and outcome
                        outcome_trend = (
                            filtered_df_outcome.groupby(
                                [filtered_df_outcome["Call Date"].dt.date, "Outcome"]
                            )
                            .size()
                            .reset_index()
                        )
                        outcome_trend.columns = ["Date", "Outcome", "Count"]
                        outcome_trend_pivot = outcome_trend.pivot(
                            index="Date", columns="Outcome", values="Count"
                        ).fillna(0)

                        fig_outcome_trend, ax_outcome_trend = plt.subplots(
                            figsize=(12, 6)
                        )
                        for outcome in top_5_outcomes:
                            if outcome in outcome_trend_pivot.columns:
                                ax_outcome_trend.plot(
                                    outcome_trend_pivot.index,
                                    outcome_trend_pivot[outcome],
                                    marker="o",
                                    label=outcome,
                                    linewidth=2,
                                )
                        ax_outcome_trend.set_xlabel("Date")
                        ax_outcome_trend.set_ylabel("Number of Calls")
                        ax_outcome_trend.set_title("Top 5 Outcomes Over Time")
                        ax_outcome_trend.legend()
                        ax_outcome_trend.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st_pyplot_safe(fig_outcome_trend)

        with reason_tab3:
            # Extract products from Summary, Reason, and Outcome fields
            if (
                "Summary" in filtered_df.columns
                or "Reason" in filtered_df.columns
                or "Outcome" in filtered_df.columns
            ):
                # Combine text from all three fields for product extraction
                product_data = []
                for idx, row in filtered_df.iterrows():
                    combined_text = " ".join(
                        [
                            str(row.get("Summary", "") or ""),
                            str(row.get("Reason", "") or ""),
                            str(row.get("Outcome", "") or ""),
                        ]
                    )
                    products = extract_products_from_text(combined_text)
                    product_data.extend(products)

                if len(product_data) > 0:
                    product_col1, product_col2 = st.columns(2)

                    with product_col1:
                        st.write("**Most Discussed Products**")
                        product_counts = pd.Series(product_data).value_counts().head(10)
                        if len(product_counts) > 0:
                            fig_product, ax_product = plt.subplots(figsize=(8, 6))
                            product_counts.plot(
                                kind="barh", ax=ax_product, color="orange"
                            )
                            ax_product.set_xlabel("Number of Mentions")
                            ax_product.set_title("Top 10 Products Discussed")
                            plt.tight_layout()
                            st_pyplot_safe(fig_product)

                    with product_col2:
                        st.write("**Product Distribution**")
                        product_counts_all = pd.Series(product_data).value_counts()
                        if len(product_counts_all) > 0:
                            # Show top 10 in pie chart, rest as "Other"
                            top_products = product_counts_all.head(10)
                            other_count = (
                                product_counts_all.iloc[10:].sum()
                                if len(product_counts_all) > 10
                                else 0
                            )

                            if other_count > 0:
                                pie_data = pd.concat(
                                    [top_products, pd.Series({"Other": other_count})]
                                )
                            else:
                                pie_data = top_products

                            fig_product_pie, ax_product_pie = plt.subplots(
                                figsize=(8, 6)
                            )
                            wedges, texts, autotexts = ax_product_pie.pie(
                                pie_data.values,
                                labels=None,  # Remove labels from pie chart
                                autopct="%1.1f%%",
                                startangle=90,
                            )
                            ax_product_pie.set_title("Products Distribution")
                            # Add legend with labels
                            ax_product_pie.legend(
                                wedges,
                                pie_data.index,
                                title="Products",
                                loc="center left",
                                bbox_to_anchor=(1, 0, 0.5, 1),
                            )
                            plt.tight_layout()
                            st_pyplot_safe(fig_product_pie)

                    # Trend over time
                    if "Call Date" in filtered_df.columns and len(filtered_df) > 0:
                        st.write("**Product Mentions Over Time**")
                        # Create a DataFrame with products per call
                        call_products = []
                        for idx, row in filtered_df.iterrows():
                            combined_text = " ".join(
                                [
                                    str(row.get("Summary", "") or ""),
                                    str(row.get("Reason", "") or ""),
                                    str(row.get("Outcome", "") or ""),
                                ]
                            )
                            products = extract_products_from_text(combined_text)
                            call_date = row.get("Call Date")
                            if call_date and len(products) > 0:
                                for product in products:
                                    call_products.append(
                                        {"Date": call_date, "Product": product}
                                    )

                        if len(call_products) > 0:
                            products_df = pd.DataFrame(call_products)
                            products_df["Date"] = pd.to_datetime(
                                products_df["Date"], errors="coerce"
                            )
                            products_df = products_df.dropna(subset=["Date"])

                            if len(products_df) > 0:
                                # Get top 5 products for trend
                                top_5_products = (
                                    pd.Series(product_data)
                                    .value_counts()
                                    .head(5)
                                    .index.tolist()
                                )
                                products_df_filtered = products_df[
                                    products_df["Product"].isin(top_5_products)
                                ].copy()

                                if len(products_df_filtered) > 0:
                                    # Group by date and product
                                    product_trend = (
                                        products_df_filtered.groupby(
                                            [
                                                products_df_filtered["Date"].dt.date,
                                                "Product",
                                            ]
                                        )
                                        .size()
                                        .reset_index()
                                    )
                                    product_trend.columns = ["Date", "Product", "Count"]
                                    product_trend_pivot = product_trend.pivot(
                                        index="Date", columns="Product", values="Count"
                                    ).fillna(0)

                                    fig_product_trend, ax_product_trend = plt.subplots(
                                        figsize=(12, 6)
                                    )
                                    for product in top_5_products:
                                        if product in product_trend_pivot.columns:
                                            ax_product_trend.plot(
                                                product_trend_pivot.index,
                                                product_trend_pivot[product],
                                                marker="o",
                                                label=product,
                                                linewidth=2,
                                            )
                                    ax_product_trend.set_xlabel("Date")
                                    ax_product_trend.set_ylabel("Number of Mentions")
                                    ax_product_trend.set_title(
                                        "Top 5 Products Over Time"
                                    )
                                    ax_product_trend.legend()
                                    ax_product_trend.grid(True, alpha=0.3)
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st_pyplot_safe(fig_product_trend)
                else:
                    st.info(
                        "No products found in call data. Products are extracted from Summary, Reason, and Outcome fields."
                    )

# --- AHT Root Cause Analysis ---
if (
    "Call Duration (min)" in filtered_df.columns
    and filtered_df["Call Duration (min)"].notna().any()
):
    with st.expander(" Average Handle Time (AHT) Root Cause Analysis", expanded=False):
        st.markdown("### Analyze Factors Contributing to Long Call Duration")
        st.caption(
            "Identifies patterns and correlations in calls with extended handle times"
        )

        # Calculate threshold for "long" calls (75th percentile)
        aht_threshold = filtered_df["Call Duration (min)"].quantile(0.75)
        long_aht_calls = filtered_df[
            filtered_df["Call Duration (min)"] >= aht_threshold
        ]

        if len(long_aht_calls) > 0:
            st.info(
                f" Analyzing {len(long_aht_calls)} calls with AHT ≥ {aht_threshold:.1f} min (75th percentile)"
            )

            # Initialize variables for insights section (calculated later in analysis)
            high_aht_coaching_categories = []
            rubric_code_aht_analysis = {}

            # Function to categorize coaching suggestions and challenges (needed for insights)
            def categorize_text(text, text_type="coaching"):
                """Categorize coaching suggestions or challenges into themes"""
                if not text or pd.isna(text):
                    return "Other"

                text_lower = str(text).lower()

                # Product knowledge keywords
                product_keywords = [
                    "product",
                    "item",
                    "inventory",
                    "stock",
                    "availability",
                    "specification",
                    "feature",
                    "model",
                    "brand",
                    "catalog",
                    "knowledge",
                    "unfamiliar",
                    "unaware",
                    "doesn't know",
                ]

                # Communication keywords
                communication_keywords = [
                    "communication",
                    "clarity",
                    "explain",
                    "understand",
                    "confusing",
                    "unclear",
                    "misunderstand",
                    "listening",
                    "tone",
                    "empathy",
                    "rapport",
                    "connection",
                    "interrupt",
                ]

                # Hold time keywords
                hold_keywords = [
                    "hold",
                    "wait",
                    "transfer",
                    "escalate",
                    "supervisor",
                    "on hold",
                    "put on hold",
                    "long hold",
                    "excessive hold",
                ]

                # System/Technical keywords
                system_keywords = [
                    "system",
                    "technical",
                    "error",
                    "bug",
                    "glitch",
                    "software",
                    "platform",
                    "tool",
                    "database",
                    "slow",
                    "loading",
                    "crash",
                    "down",
                    "issue",
                    "problem",
                ]

                # Process/Procedure keywords
                process_keywords = [
                    "process",
                    "procedure",
                    "policy",
                    "protocol",
                    "workflow",
                    "step",
                    "order",
                    "sequence",
                    "method",
                    "approach",
                    "guideline",
                    "standard",
                    "compliance",
                ]

                # Efficiency keywords
                efficiency_keywords = [
                    "efficient",
                    "quick",
                    "fast",
                    "speed",
                    "time",
                    "streamline",
                    "optimize",
                    "reduce time",
                    "faster",
                ]

                # Count matches for each category
                product_score = sum(1 for kw in product_keywords if kw in text_lower)
                comm_score = sum(1 for kw in communication_keywords if kw in text_lower)
                hold_score = sum(1 for kw in hold_keywords if kw in text_lower)
                system_score = sum(1 for kw in system_keywords if kw in text_lower)
                process_score = sum(1 for kw in process_keywords if kw in text_lower)
                efficiency_score = sum(
                    1 for kw in efficiency_keywords if kw in text_lower
                )

                # Return category with highest score
                scores = {
                    "Product Knowledge": product_score,
                    "Communication": comm_score,
                    "Hold Time": hold_score,
                    "System/Technical": system_score,
                    "Process/Procedure": process_score,
                    "Efficiency": efficiency_score,
                }

                max_score = max(scores.values())
                if max_score > 0:
                    return max(scores, key=scores.get)
                return "Other"

            # Calculate data needed for insights (quick calculation)
            # Calculate coaching categories for high-AHT calls
            if "Coaching Suggestions" in filtered_df.columns:
                for idx, row in long_aht_calls.iterrows():
                    coaching = row.get("Coaching Suggestions", [])
                    if isinstance(coaching, list):
                        for item in coaching:
                            if item:
                                high_aht_coaching_categories.append(
                                    categorize_text(item, "coaching")
                                )
                    elif coaching:
                        high_aht_coaching_categories.append(
                            categorize_text(coaching, "coaching")
                        )

            # Calculate rubric code analysis
            if "Rubric Details" in filtered_df.columns:
                for idx, row in filtered_df.iterrows():
                    aht = row.get("Call Duration (min)", None)
                    if pd.isna(aht):
                        continue

                    rubric_details = row.get("Rubric Details", {})
                    if isinstance(rubric_details, dict):
                        for code, details in rubric_details.items():
                            if (
                                isinstance(details, dict)
                                and details.get("status") == "Fail"
                            ):
                                if code not in rubric_code_aht_analysis:
                                    rubric_code_aht_analysis[code] = {
                                        "total_fails": 0,
                                        "high_aht_fails": 0,
                                        "aht_sum": 0,
                                        "aht_count": 0,
                                    }

                                rubric_code_aht_analysis[code]["total_fails"] += 1
                                rubric_code_aht_analysis[code]["aht_sum"] += aht
                                rubric_code_aht_analysis[code]["aht_count"] += 1

                                if aht >= aht_threshold:
                                    rubric_code_aht_analysis[code][
                                        "high_aht_fails"
                                    ] += 1

            # --- Actionable Insights Summary (moved to top) ---
            st.markdown("###  Actionable Insights")
            insights = []

            # Coaching insights
            if (
                "Coaching Suggestions" in filtered_df.columns
                and high_aht_coaching_categories
                and len(high_aht_coaching_categories) > 0
            ):
                from collections import Counter

                high_aht_coaching_counts = Counter(high_aht_coaching_categories)
                top_coaching = high_aht_coaching_counts.most_common(1)
                if top_coaching:
                    category, count = top_coaching[0]
                    total_high_aht = len(high_aht_coaching_categories)
                    pct = (count / total_high_aht * 100) if total_high_aht > 0 else 0
                    insights.append(
                        f" **{category}** issues appear in {pct:.1f}% of coaching suggestions for high-AHT calls"
                    )

            # Rubric insights
            if rubric_code_aht_analysis:
                # Find rubric code with highest percentage of high-AHT failures
                max_high_aht_pct = 0
                top_rubric_code = None
                for code, stats in rubric_code_aht_analysis.items():
                    if (
                        stats["total_fails"] >= 5
                    ):  # Only consider codes with at least 5 failures
                        high_aht_pct = (
                            (stats["high_aht_fails"] / stats["total_fails"] * 100)
                            if stats["total_fails"] > 0
                            else 0
                        )
                        if high_aht_pct > max_high_aht_pct:
                            max_high_aht_pct = high_aht_pct
                            top_rubric_code = code

                if top_rubric_code:
                    stats = rubric_code_aht_analysis[top_rubric_code]
                    insights.append(
                        f" **Rubric Code {top_rubric_code}** fails in {max_high_aht_pct:.1f}% of high-AHT calls ({stats['high_aht_fails']} out of {stats['total_fails']} failures)"
                    )

            # AHT difference insight
            avg_all_aht = filtered_df["Call Duration (min)"].mean()
            avg_high_aht = long_aht_calls["Call Duration (min)"].mean()
            aht_difference = avg_high_aht - avg_all_aht
            insights.append(
                f" High-AHT calls average **{aht_difference:.1f} min longer** than all calls ({avg_high_aht:.1f} min vs {avg_all_aht:.1f} min)"
            )

            # Fail count insight
            if "Rubric Fail Count" in filtered_df.columns:
                avg_fails_all = filtered_df["Rubric Fail Count"].mean()
                avg_fails_high = long_aht_calls["Rubric Fail Count"].mean()
                if not pd.isna(avg_fails_all) and not pd.isna(avg_fails_high):
                    fail_diff = avg_fails_high - avg_fails_all
                    insights.append(
                        f" High-AHT calls have **{fail_diff:.1f} more failed rubric items** on average ({avg_fails_high:.1f} vs {avg_fails_all:.1f})"
                    )

            # Display insights
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.info("Run the analysis below to generate insights")

            # --- Summary Statistics (moved to second position) ---
            st.markdown("---")
            st.markdown("###  Summary Statistics: Long AHT Calls")
            summary_stats = {
                "Metric": [
                    "Total Long AHT Calls",
                    "Average AHT (Long Calls)",
                    "Average AHT (All Calls)",
                    "Average QA Score (Long Calls)",
                    "Average QA Score (All Calls)",
                    "Avg Fail Count (Long Calls)",
                    "Avg Fail Count (All Calls)",
                ],
                "Value": [
                    str(
                        len(long_aht_calls)
                    ),  # Convert to string to avoid Arrow serialization error
                    f"{long_aht_calls['Call Duration (min)'].mean():.1f} min",
                    f"{filtered_df['Call Duration (min)'].mean():.1f} min",
                    f"{long_aht_calls['QA Score'].mean():.1f}%"
                    if "QA Score" in long_aht_calls.columns
                    else "N/A",
                    f"{filtered_df['QA Score'].mean():.1f}%"
                    if "QA Score" in filtered_df.columns
                    else "N/A",
                    f"{long_aht_calls['Rubric Fail Count'].mean():.1f}"
                    if "Rubric Fail Count" in long_aht_calls.columns
                    else "N/A",
                    f"{filtered_df['Rubric Fail Count'].mean():.1f}"
                    if "Rubric Fail Count" in filtered_df.columns
                    else "N/A",
                ],
            }
            st.dataframe(pd.DataFrame(summary_stats), hide_index=True)

            # --- Visualizations and Detailed Analysis ---
            st.markdown("---")
            st.markdown("###  Detailed Analysis")

            rca_col1, rca_col2 = st.columns(2)

            with rca_col1:
                st.write("**AHT Distribution**")
                fig_aht_dist, ax_aht_dist = plt.subplots(figsize=(10, 5))
                ax_aht_dist.hist(
                    filtered_df["Call Duration (min)"].dropna(),
                    bins=30,
                    color="steelblue",
                    alpha=0.7,
                    edgecolor="black",
                )
                ax_aht_dist.axvline(
                    x=aht_threshold,
                    color="r",
                    linestyle="--",
                    linewidth=2,
                    label=f"75th Percentile ({aht_threshold:.1f} min)",
                )
                ax_aht_dist.set_xlabel("Average Handle Time (min)")
                ax_aht_dist.set_ylabel("Number of Calls")
                ax_aht_dist.set_title("AHT Distribution")
                ax_aht_dist.legend()
                plt.tight_layout()
                st_pyplot_safe(fig_aht_dist)

            with rca_col2:
                st.write("**AHT vs QA Score Correlation**")
                if "QA Score" in filtered_df.columns:
                    # Scatter plot
                    fig_aht_score, ax_aht_score = plt.subplots(figsize=(10, 5))
                    ax_aht_score.scatter(
                        filtered_df["QA Score"],
                        filtered_df["Call Duration (min)"],
                        alpha=0.5,
                        color="steelblue",
                    )
                    ax_aht_score.scatter(
                        long_aht_calls["QA Score"],
                        long_aht_calls["Call Duration (min)"],
                        alpha=0.7,
                        color="red",
                        label="Long AHT Calls",
                    )
                    ax_aht_score.set_xlabel("QA Score (%)")
                    ax_aht_score.set_ylabel("AHT (min)")
                    ax_aht_score.set_title("AHT vs QA Score")
                    ax_aht_score.legend()
                    ax_aht_score.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st_pyplot_safe(fig_aht_score)

                    # Calculate correlation
                    correlation = filtered_df["QA Score"].corr(
                        filtered_df["Call Duration (min)"]
                    )
                    if not pd.isna(correlation):
                        st.metric(
                            "Correlation",
                            f"{correlation:.3f}",
                            help="Negative = longer calls tend to have lower scores",
                        )

            # Analysis by Agent
            st.write("**Top Agents by Long AHT Calls**")
            agent_aht_analysis = (
                long_aht_calls.groupby("Agent")
                .agg(
                    Long_AHT_Calls=("Call ID", "count"),
                    Avg_AHT=("Call Duration (min)", "mean"),
                    Avg_QA_Score=("QA Score", "mean"),
                )
                .reset_index()
                .sort_values("Long_AHT_Calls", ascending=False)
                .head(10)
            )
            if len(agent_aht_analysis) > 0:
                st.dataframe(agent_aht_analysis.round(2), hide_index=True)

            # --- Root Cause Analysis: Coaching Suggestions & Challenges ---
            st.markdown("---")
            st.markdown("###  Root Cause Analysis: Text-Based Factors")

            # Function to categorize coaching suggestions and challenges
            def categorize_text(text, text_type="coaching"):
                """Categorize coaching suggestions or challenges into themes"""
                if not text or pd.isna(text):
                    return "Other"

                text_lower = str(text).lower()

                # Product knowledge keywords
                product_keywords = [
                    "product",
                    "item",
                    "inventory",
                    "stock",
                    "availability",
                    "specification",
                    "feature",
                    "model",
                    "brand",
                    "catalog",
                    "knowledge",
                    "unfamiliar",
                    "unaware",
                    "doesn't know",
                ]

                # Communication keywords
                communication_keywords = [
                    "communication",
                    "clarity",
                    "explain",
                    "understand",
                    "confusing",
                    "unclear",
                    "misunderstand",
                    "listening",
                    "tone",
                    "empathy",
                    "rapport",
                    "connection",
                    "interrupt",
                ]

                # Hold time keywords
                hold_keywords = [
                    "hold",
                    "wait",
                    "transfer",
                    "escalate",
                    "supervisor",
                    "on hold",
                    "put on hold",
                    "long hold",
                    "excessive hold",
                ]

                # System/Technical keywords
                system_keywords = [
                    "system",
                    "technical",
                    "error",
                    "bug",
                    "glitch",
                    "software",
                    "platform",
                    "tool",
                    "database",
                    "slow",
                    "loading",
                    "crash",
                    "down",
                    "issue",
                    "problem",
                ]

                # Process/Procedure keywords
                process_keywords = [
                    "process",
                    "procedure",
                    "policy",
                    "protocol",
                    "workflow",
                    "step",
                    "order",
                    "sequence",
                    "method",
                    "approach",
                    "guideline",
                    "standard",
                    "compliance",
                ]

                # Efficiency keywords
                efficiency_keywords = [
                    "efficient",
                    "quick",
                    "fast",
                    "speed",
                    "time",
                    "streamline",
                    "optimize",
                    "reduce time",
                    "faster",
                ]

                # Count matches for each category
                product_score = sum(1 for kw in product_keywords if kw in text_lower)
                comm_score = sum(1 for kw in communication_keywords if kw in text_lower)
                hold_score = sum(1 for kw in hold_keywords if kw in text_lower)
                system_score = sum(1 for kw in system_keywords if kw in text_lower)
                process_score = sum(1 for kw in process_keywords if kw in text_lower)
                efficiency_score = sum(
                    1 for kw in efficiency_keywords if kw in text_lower
                )

                # Return category with highest score
                scores = {
                    "Product Knowledge": product_score,
                    "Communication": comm_score,
                    "Hold Time": hold_score,
                    "System/Technical": system_score,
                    "Process/Procedure": process_score,
                    "Efficiency": efficiency_score,
                }

                max_score = max(scores.values())
                if max_score > 0:
                    return max(scores, key=scores.get)
                return "Other"

            # Analyze coaching suggestions in high-AHT calls
            if "Coaching Suggestions" in filtered_df.columns:
                st.write("**Coaching Suggestions Analysis: High AHT vs All Calls**")

                # Categorize coaching suggestions for all calls
                all_coaching_categories = []
                high_aht_coaching_categories = []  # Reset for this analysis
                for idx, row in filtered_df.iterrows():
                    coaching = row.get("Coaching Suggestions", [])
                    if isinstance(coaching, list):
                        for item in coaching:
                            if item:
                                all_coaching_categories.append(
                                    categorize_text(item, "coaching")
                                )
                    elif coaching:
                        all_coaching_categories.append(
                            categorize_text(coaching, "coaching")
                        )

                # Categorize coaching suggestions for high-AHT calls
                high_aht_coaching_categories = []
                for idx, row in long_aht_calls.iterrows():
                    coaching = row.get("Coaching Suggestions", [])
                    if isinstance(coaching, list):
                        for item in coaching:
                            if item:
                                high_aht_coaching_categories.append(
                                    categorize_text(item, "coaching")
                                )
                    elif coaching:
                        high_aht_coaching_categories.append(
                            categorize_text(coaching, "coaching")
                        )

                if high_aht_coaching_categories:
                    # Calculate frequencies
                    from collections import Counter

                    all_coaching_counts = Counter(all_coaching_categories)
                    high_aht_coaching_counts = Counter(high_aht_coaching_categories)

                    # Create comparison DataFrame
                    coaching_comparison = []
                    all_categories = set(
                        list(all_coaching_counts.keys())
                        + list(high_aht_coaching_counts.keys())
                    )

                    for category in sorted(all_categories):
                        all_count = all_coaching_counts.get(category, 0)
                        high_aht_count = high_aht_coaching_counts.get(category, 0)
                        all_pct = (
                            (all_count / len(all_coaching_categories) * 100)
                            if all_coaching_categories
                            else 0
                        )
                        high_aht_pct = (
                            (high_aht_count / len(high_aht_coaching_categories) * 100)
                            if high_aht_coaching_categories
                            else 0
                        )

                        coaching_comparison.append(
                            {
                                "Category": category,
                                "All Calls": f"{all_count} ({all_pct:.1f}%)",
                                "High AHT Calls": f"{high_aht_count} ({high_aht_pct:.1f}%)",
                                "Difference": f"{high_aht_pct - all_pct:+.1f}%",
                            }
                        )

                    coaching_df = pd.DataFrame(coaching_comparison)
                    st.dataframe(coaching_df, hide_index=True)

                    # Visualization
                    col_coach1, col_coach2 = st.columns(2)
                    with col_coach1:
                        st.write("**Top Categories in High AHT Calls**")
                        top_categories = high_aht_coaching_counts.most_common(8)
                        if top_categories:
                            fig_coach, ax_coach = plt.subplots(figsize=(10, 6))
                            categories, counts = zip(*top_categories)
                            ax_coach.barh(categories, counts, color="coral")
                            ax_coach.set_xlabel("Frequency")
                            ax_coach.set_title(
                                "Top Coaching Categories in High AHT Calls"
                            )
                            plt.tight_layout()
                            st_pyplot_safe(fig_coach)

                    with col_coach2:
                        st.write("**Category Comparison**")
                        comparison_categories = [cat for cat, _ in top_categories[:6]]
                        all_values = [
                            all_coaching_counts.get(cat, 0)
                            for cat in comparison_categories
                        ]
                        high_aht_values = [
                            high_aht_coaching_counts.get(cat, 0)
                            for cat in comparison_categories
                        ]

                        x = range(len(comparison_categories))
                        width = 0.35
                        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
                        ax_comp.barh(
                            [i - width / 2 for i in x],
                            all_values,
                            width,
                            label="All Calls",
                            color="steelblue",
                            alpha=0.7,
                        )
                        ax_comp.barh(
                            [i + width / 2 for i in x],
                            high_aht_values,
                            width,
                            label="High AHT Calls",
                            color="coral",
                            alpha=0.7,
                        )
                        ax_comp.set_yticks(x)
                        ax_comp.set_yticklabels(comparison_categories)
                        ax_comp.set_xlabel("Frequency")
                        ax_comp.set_title("Coaching Categories: All vs High AHT")
                        ax_comp.legend()
                        plt.tight_layout()
                        st_pyplot_safe(fig_comp)

            # Analyze Challenges in high-AHT calls
            if "Challenges" in filtered_df.columns:
                st.write("**Challenges Analysis: High AHT vs All Calls**")

                # Categorize challenges
                all_challenge_categories = []
                for idx, row in filtered_df.iterrows():
                    challenge = row.get("Challenges", "")
                    if challenge and pd.notna(challenge):
                        all_challenge_categories.append(
                            categorize_text(challenge, "challenge")
                        )

                high_aht_challenge_categories = []
                for idx, row in long_aht_calls.iterrows():
                    challenge = row.get("Challenges", "")
                    if challenge and pd.notna(challenge):
                        high_aht_challenge_categories.append(
                            categorize_text(challenge, "challenge")
                        )

                if high_aht_challenge_categories:
                    from collections import Counter

                    all_challenge_counts = Counter(all_challenge_categories)
                    high_aht_challenge_counts = Counter(high_aht_challenge_categories)

                    challenge_comparison = []
                    all_challenge_cats = set(
                        list(all_challenge_counts.keys())
                        + list(high_aht_challenge_counts.keys())
                    )

                    for category in sorted(all_challenge_cats):
                        all_count = all_challenge_counts.get(category, 0)
                        high_aht_count = high_aht_challenge_counts.get(category, 0)
                        all_pct = (
                            (all_count / len(all_challenge_categories) * 100)
                            if all_challenge_categories
                            else 0
                        )
                        high_aht_pct = (
                            (high_aht_count / len(high_aht_challenge_categories) * 100)
                            if high_aht_challenge_categories
                            else 0
                        )

                        challenge_comparison.append(
                            {
                                "Category": category,
                                "All Calls": f"{all_count} ({all_pct:.1f}%)",
                                "High AHT Calls": f"{high_aht_count} ({high_aht_pct:.1f}%)",
                                "Difference": f"{high_aht_pct - all_pct:+.1f}%",
                            }
                        )

                    challenge_df = pd.DataFrame(challenge_comparison)
                    st.dataframe(challenge_df, hide_index=True)

            # --- Rubric Code Correlation with AHT ---
            st.markdown("---")
            st.markdown("###  Rubric Code Correlation with High AHT")

            if "Rubric Details" in filtered_df.columns:
                # Analyze which rubric codes appear most in high-AHT calls
                # Note: rubric_code_aht_analysis already calculated above for insights
                # No need to recalculate - just use the existing data

                if rubric_code_aht_analysis:
                    # Create analysis DataFrame
                    rubric_aht_list = []
                    for code, stats in rubric_code_aht_analysis.items():
                        avg_aht = (
                            stats["aht_sum"] / stats["aht_count"]
                            if stats["aht_count"] > 0
                            else 0
                        )
                        high_aht_pct = (
                            (stats["high_aht_fails"] / stats["total_fails"] * 100)
                            if stats["total_fails"] > 0
                            else 0
                        )

                        rubric_aht_list.append(
                            {
                                "Rubric Code": code,
                                "Total Fails": stats["total_fails"],
                                "High AHT Fails": stats["high_aht_fails"],
                                "% High AHT": f"{high_aht_pct:.1f}%",
                                "Avg AHT (when failed)": f"{avg_aht:.1f} min",
                            }
                        )

                    rubric_aht_df = pd.DataFrame(rubric_aht_list)
                    rubric_aht_df = rubric_aht_df.sort_values(
                        "High AHT Fails", ascending=False
                    ).head(15)

                    st.dataframe(rubric_aht_df, hide_index=True)

                    # Visualization
                    if len(rubric_aht_df) > 0:
                        fig_rubric_aht, ax_rubric_aht = plt.subplots(figsize=(12, 6))
                        top_codes = rubric_aht_df.head(10)
                        ax_rubric_aht.barh(
                            top_codes["Rubric Code"],
                            top_codes["High AHT Fails"],
                            color="crimson",
                        )
                        ax_rubric_aht.set_xlabel("Number of High AHT Failures")
                        ax_rubric_aht.set_title(
                            "Top Rubric Codes Failing in High AHT Calls"
                        )
                        plt.tight_layout()
                        st_pyplot_safe(fig_rubric_aht)

        else:
            st.info("No calls found above the 75th percentile threshold")

# --- QA Score Trends Over Time ---
with st.expander("QA Score Trends Over Time", expanded=False):
    st.subheader("QA Score Trends Over Time")
    col_trend1, col_trend2 = st.columns(2)

    with col_trend1:
        st.write("**QA Score Trend**")
        if len(filtered_df) > 0 and "QA Score" in filtered_df.columns:
            # Daily average QA scores
            daily_scores = (
                filtered_df.groupby(filtered_df["Call Date"].dt.date)["QA Score"]
                .mean()
                .reset_index()
            )
            daily_scores.columns = ["Date", "Avg QA Score"]

            fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
            ax_trend.plot(
                daily_scores["Date"],
                daily_scores["Avg QA Score"],
                marker="o",
                linewidth=2,
                color="steelblue",
            )
            ax_trend.set_xlabel("Date")
            ax_trend.set_ylabel("Average QA Score (%)")
            ax_trend.set_title("QA Score Trend Over Time")
            ax_trend.grid(True, alpha=0.3)
            ax_trend.axhline(
                y=alert_threshold,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Threshold ({alert_threshold}%)",
            )
            ax_trend.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st_pyplot_safe(fig_trend)

    with col_trend2:
        # Pass/Fail Rate Trends
        st.write("**Pass/Fail Rate Trends**")
        if len(filtered_df) > 0:
            daily_stats = (
                filtered_df.groupby(filtered_df["Call Date"].dt.date)
                .agg(
                    Total_Pass=("Rubric Pass Count", "sum"),
                    Total_Fail=("Rubric Fail Count", "sum"),
                )
                .reset_index()
            )
            daily_stats.columns = ["Date", "Total_Pass", "Total_Fail"]
            daily_stats["Total"] = daily_stats["Total_Pass"] + daily_stats["Total_Fail"]
            daily_stats["Pass_Rate"] = (
                daily_stats["Total_Pass"] / daily_stats["Total"] * 100
            ).fillna(0)
            daily_stats["Fail_Rate"] = (
                daily_stats["Total_Fail"] / daily_stats["Total"] * 100
            ).fillna(0)

            # Create line chart
            fig_pf, ax_pf = plt.subplots(figsize=(10, 5))
            ax_pf.plot(
                daily_stats["Date"],
                daily_stats["Pass_Rate"],
                marker="o",
                linewidth=2,
                label="Pass Rate",
                color="green",
            )
            ax_pf.plot(
                daily_stats["Date"],
                daily_stats["Fail_Rate"],
                marker="s",
                linewidth=2,
                label="Fail Rate",
                color="red",
            )
            ax_pf.set_xlabel("Date")
            ax_pf.set_ylabel("Rate (%)")
            ax_pf.set_title("Pass/Fail Rate Trends Over Time")
            ax_pf.grid(True, alpha=0.3)
            ax_pf.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st_pyplot_safe(fig_pf)

# --- Rubric Code Analysis ---
with st.expander("Rubric Code Analysis", expanded=False):
    st.subheader("Rubric Code Analysis")
    if "Rubric Details" in filtered_df.columns:
        # Collect all rubric code statistics
        code_stats = {}
        for idx, row in filtered_df.iterrows():
            rubric_details = row.get("Rubric Details", {})
            if isinstance(rubric_details, dict):
                for code, details in rubric_details.items():
                    if isinstance(details, dict):
                        status = details.get("status", "N/A")
                        note = details.get("note", "")

                        if code not in code_stats:
                            code_stats[code] = {
                                "total": 0,
                                "pass": 0,
                                "fail": 0,
                                "na": 0,
                                "fail_notes": [],
                            }

                        code_stats[code]["total"] += 1
                        if status == "Pass":
                            code_stats[code]["pass"] += 1
                        elif status == "Fail":
                            code_stats[code]["fail"] += 1
                        if note:
                            code_stats[code]["fail_notes"].append(note)
                        elif status == "N/A":
                            code_stats[code]["na"] += 1

        if code_stats:
            rubric_analysis = pd.DataFrame(
                [
                    {
                        "Code": code,
                        "Total": stats["total"],
                        "Pass": stats["pass"],
                        "Fail": stats["fail"],
                        "N/A": stats["na"],
                        "Pass_Rate": (stats["pass"] / stats["total"] * 100)
                        if stats["total"] > 0
                        else 0,
                        "Fail_Rate": (stats["fail"] / stats["total"] * 100)
                        if stats["total"] > 0
                        else 0,
                        "Most_Common_Fail_Reason": max(
                            set(stats["fail_notes"]), key=stats["fail_notes"].count
                        )
                        if stats["fail_notes"]
                        else "N/A",
                    }
                    for code, stats in code_stats.items()
                ]
            )
            rubric_analysis = rubric_analysis.sort_values("Fail_Rate", ascending=False)

            # Top 10 Failed Rubric Codes - full width
            st.write("**Top 10 Failed Rubric Codes**")
            top_failed = rubric_analysis.head(10)
            st.dataframe(
                top_failed[
                    [
                        "Code",
                        "Total",
                        "Fail",
                        "Fail_Rate",
                        "Most_Common_Fail_Reason",
                    ]
                ],
                hide_index=True,
            )

            # Category-level analysis
            if code_stats:
                category_stats = {}
                for code, stats in code_stats.items():
                    # Extract category from code (e.g., "A1" from "A1.1")
                    category = (
                        code.split(".")[0]
                        if "." in code
                        else code[0]
                        if code
                        else "Other"
                    )

                    if category not in category_stats:
                        category_stats[category] = {
                            "total": 0,
                            "fail": 0,
                            "fail_rates": [],
                        }

                    category_stats[category]["total"] += stats["total"]
                    category_stats[category]["fail"] += stats["fail"]
                    if stats["total"] > 0:
                        fail_rate = (stats["fail"] / stats["total"]) * 100
                        category_stats[category]["fail_rates"].append(fail_rate)

                if category_stats:
                    category_df = pd.DataFrame(
                        [
                            {
                                "Category": cat,
                                "Total": stats["total"],
                                "Total_Fail": stats["fail"],
                                "Avg_Fail_Rate": (
                                    sum(stats["fail_rates"]) / len(stats["fail_rates"])
                                    if stats["fail_rates"]
                                    else 0
                                ),
                            }
                            for cat, stats in category_stats.items()
                        ]
                    )
                    category_df = category_df.sort_values(
                        "Avg_Fail_Rate", ascending=False
                    )

                    # Put Fail Rate Distribution and Fail Rate by Rubric Category side by side
                    rubric_col1, rubric_col2 = st.columns(2)

                    with rubric_col1:
                        st.write("**Fail Rate Distribution**")
                        fig_rubric, ax_rubric = plt.subplots(figsize=(8, 6))
                        top_failed.plot(
                            x="Code",
                            y="Fail_Rate",
                            kind="bar",
                            ax=ax_rubric,
                            color="red",
                        )
                        ax_rubric.set_ylabel("Fail Rate (%)")
                        ax_rubric.set_xlabel("Rubric Code")
                        ax_rubric.set_title("Top 10 Failed Rubric Codes")
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        st_pyplot_safe(fig_rubric)

                    with rubric_col2:
                        st.write("**Fail Rate by Rubric Category**")
                        fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
                        colors = [
                            "green" if x < 20 else "orange" if x < 40 else "red"
                            for x in category_df["Avg_Fail_Rate"]
                        ]
                        category_df.plot(
                            x="Category",
                            y="Avg_Fail_Rate",
                            kind="bar",
                            ax=ax_heat,
                            color=colors,
                        )
                        ax_heat.set_ylabel("Average Fail Rate (%)")
                        ax_heat.set_xlabel("Rubric Category")
                        ax_heat.set_title("Fail Rate by Rubric Category")
                        plt.xticks(rotation=0)
                        plt.tight_layout()
                        st_pyplot_safe(fig_heat)

# --- Trend Forecasting (Predictive Analytics) ---
with st.expander("Trend Forecasting", expanded=False):
    st.markdown("### Predict Future QA Scores")
    st.caption(
        "Forecasts future QA scores based on historical trends using time series analysis"
    )

    forecast_days = st.selectbox(
        "Forecast Period", [7, 14, 30], index=0, help="Number of days to forecast ahead"
    )

    if (
        len(filtered_df) > 0
        and "QA Score" in filtered_df.columns
        and "Call Date" in filtered_df.columns
    ):
        with st.spinner("Calculating forecast..."):
            forecast_result = predict_future_scores(
                filtered_df, days_ahead=forecast_days
            )

        if forecast_result:
            # Display forecast
            forecast_df = pd.DataFrame(
                {
                    "Date": forecast_result["dates"],
                    "Forecast": forecast_result["forecast"],
                    "Lower Bound": forecast_result["lower_bound"],
                    "Upper Bound": forecast_result["upper_bound"],
                }
            )

            # Create forecast chart
            fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))

            # Plot historical data
            daily_scores = (
                filtered_df.groupby(filtered_df["Call Date"].dt.date)["QA Score"]
                .mean()
                .reset_index()
            )
            daily_scores.columns = ["Date", "Avg QA Score"]
            ax_forecast.plot(
                daily_scores["Date"],
                daily_scores["Avg QA Score"],
                marker="o",
                linewidth=2,
                color="steelblue",
                label="Historical",
            )

            # Plot forecast
            forecast_dates = pd.to_datetime(forecast_df["Date"])
            ax_forecast.plot(
                forecast_dates,
                forecast_df["Forecast"],
                marker="s",
                linewidth=2,
                color="orange",
                label="Forecast",
            )
            ax_forecast.fill_between(
                forecast_dates,
                forecast_df["Lower Bound"],
                forecast_df["Upper Bound"],
                alpha=0.3,
                color="orange",
                label="95% Confidence Interval",
            )

            ax_forecast.axhline(
                y=alert_threshold,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Threshold ({alert_threshold}%)",
            )
            ax_forecast.set_xlabel("Date")
            ax_forecast.set_ylabel("Average QA Score (%)")
            ax_forecast.set_title(
                f"QA Score Forecast ({forecast_days} days ahead) - {forecast_result['method'].title()} Method"
            )
            ax_forecast.grid(True, alpha=0.3)
            ax_forecast.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st_pyplot_safe(fig_forecast)

            # Show forecast summary
            avg_forecast = forecast_df["Forecast"].mean()
            st.metric(
                "Average Forecasted Score",
                f"{avg_forecast:.1f}%",
                delta=f"{avg_forecast - daily_scores['Avg QA Score'].iloc[-1]:+.1f}%",
                delta_color="normal"
                if avg_forecast >= daily_scores["Avg QA Score"].iloc[-1]
                else "inverse",
                help="Average of forecasted scores vs most recent historical score",
            )
        else:
            st.warning(
                " Insufficient data for forecasting (need at least 7 days of historical data)"
            )
    else:
        st.info(" No data available for forecasting")

# --- Agent-Specific Trends ---
if user_agent_id:
    # Agent view - show their trend with team comparison
    with st.expander("My Performance Trend vs Team Average", expanded=False):
        st.subheader("My Performance Trend vs Team Average")

        # Add trajectory analysis
        agent_data = filtered_df[filtered_df["Agent"] == user_agent_id]
        if len(agent_data) > 0:
            trajectory = classify_trajectory(filtered_df, agent=user_agent_id)

        traj_col1, traj_col2, traj_col3, traj_col4 = st.columns(4)
        with traj_col1:
            traj_icon = {
                "improving": "",
                "declining": "",
                "stable": "",
                "volatile": "",
                "insufficient_data": "❓",
            }.get(trajectory["trajectory"], "❓")
            traj_label = {
                "improving": "Improving",
                "declining": "Declining",
                "stable": "Stable",
                "volatile": "Volatile",
                "insufficient_data": "Insufficient Data",
            }.get(trajectory["trajectory"], "Unknown")
            st.metric("Trajectory", f"{traj_icon} {traj_label}")

            with traj_col2:
                st.metric("Current Score", f"{trajectory.get('current_score', 0):.1f}%")

            with traj_col3:
                projected = trajectory.get("projected_score", 0)
                delta = (
                    projected - trajectory.get("current_score", 0) if projected else 0
                )
                st.metric(
                    "Projected (7 days)",
                    f"{projected:.1f}%",
                    delta=f"{delta:+.1f}%",
                    delta_color="normal" if delta >= 0 else "inverse",
                    help="Projected score if current trend continues",
                )

            with traj_col4:
                st.metric(
                    "Volatility",
                    f"{trajectory.get('volatility', 0):.1f}",
                    help="Standard deviation of scores (lower is more consistent)",
                )

        agent_trends_col1, agent_trends_col2 = st.columns(2)

        with agent_trends_col1:
            st.write("**My QA Score Trend**")
            if len(agent_data) > 0:
                agent_daily = (
                    agent_data.groupby(agent_data["Call Date"].dt.date)
                    .agg(Avg_QA_Score=("QA Score", "mean"))
                    .reset_index()
                )
                agent_daily.columns = ["Date", "My_Score"]

                # Get team average for same dates
                overall_daily = (
                    overall_df.groupby(overall_df["Call Date"].dt.date)
                    .agg(Avg_QA_Score=("QA Score", "mean"))
                    .reset_index()
                )
                overall_daily.columns = ["Date", "Team_Avg"]

                # Merge on date
                trend_comparison = pd.merge(
                    agent_daily, overall_daily, on="Date", how="outer"
                ).sort_values("Date")

                fig_agent, ax_agent = plt.subplots(figsize=(8, 5))
                ax_agent.plot(
                    trend_comparison["Date"],
                    trend_comparison["My_Score"],
                    marker="o",
                    linewidth=2,
                    label="My Score",
                    color="steelblue",
                )
                ax_agent.plot(
                    trend_comparison["Date"],
                    trend_comparison["Team_Avg"],
                    marker="s",
                    linewidth=2,
                    label="Team Average",
                    color="orange",
                    linestyle="--",
                )
                ax_agent.set_xlabel("Date")
                ax_agent.set_ylabel("Average QA Score (%)")
                ax_agent.set_title("My Performance Trend vs Team Average")
                ax_agent.grid(True, alpha=0.3)
                ax_agent.axhline(
                    y=alert_threshold,
                    color="r",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Threshold ({alert_threshold}%)",
                )
                ax_agent.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st_pyplot_safe(fig_agent)

        with agent_trends_col2:
            st.write("**My Pass Rate Trend vs Team**")
            if len(agent_data) > 0:
                agent_pass_daily = (
                    agent_data.groupby(agent_data["Call Date"].dt.date)
                    .agg(
                        Total_Pass=("Rubric Pass Count", "sum"),
                        Total_Fail=("Rubric Fail Count", "sum"),
                    )
                    .reset_index()
                )
                agent_pass_daily["Total"] = (
                    agent_pass_daily["Total_Pass"] + agent_pass_daily["Total_Fail"]
                )
                agent_pass_daily["My_Pass_Rate"] = (
                    agent_pass_daily["Total_Pass"] / agent_pass_daily["Total"] * 100
                ).fillna(0)

                team_pass_daily = (
                    overall_df.groupby(overall_df["Call Date"].dt.date)
                    .agg(
                        Total_Pass=("Rubric Pass Count", "sum"),
                        Total_Fail=("Rubric Fail Count", "sum"),
                    )
                    .reset_index()
                )
                team_pass_daily["Total"] = (
                    team_pass_daily["Total_Pass"] + team_pass_daily["Total_Fail"]
                )
                team_pass_daily["Team_Pass_Rate"] = (
                    team_pass_daily["Total_Pass"] / team_pass_daily["Total"] * 100
                ).fillna(0)

                pass_comparison = pd.merge(
                    agent_pass_daily[["Call Date", "My_Pass_Rate"]],
                    team_pass_daily[["Call Date", "Team_Pass_Rate"]],
                    on="Call Date",
                    how="outer",
                ).sort_values("Call Date")

                fig_pass_trend, ax_pass_trend = plt.subplots(figsize=(8, 5))
                ax_pass_trend.plot(
                    pass_comparison["Call Date"],
                    pass_comparison["My_Pass_Rate"],
                    marker="o",
                    linewidth=2,
                    label="My Pass Rate",
                    color="green",
                )
                ax_pass_trend.plot(
                    pass_comparison["Call Date"],
                    pass_comparison["Team_Pass_Rate"],
                    marker="s",
                    linewidth=2,
                    label="Team Pass Rate",
                    color="lightgreen",
                    linestyle="--",
                )
                ax_pass_trend.set_xlabel("Date")
                ax_pass_trend.set_ylabel("Pass Rate (%)")
                ax_pass_trend.set_title("My Pass Rate Trend vs Team Average")
                ax_pass_trend.grid(True, alpha=0.3)
                ax_pass_trend.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st_pyplot_safe(fig_pass_trend)

        # AHT Trend - in a new row with two columns (AHT in first column)
        aht_col1, aht_col2 = st.columns(2)
        with aht_col1:
            st.write("**My AHT Trend vs Team**")
            if len(agent_data) > 0 and "Call Duration (min)" in agent_data.columns:
                agent_aht_daily = (
                    agent_data.groupby(agent_data["Call Date"].dt.date)
                    .agg(Avg_AHT=("Call Duration (min)", "mean"))
                    .reset_index()
                )
                agent_aht_daily.columns = ["Call Date", "My_AHT"]

                team_aht_daily = (
                    overall_df.groupby(overall_df["Call Date"].dt.date)
                    .agg(Avg_AHT=("Call Duration (min)", "mean"))
                    .reset_index()
                )
                team_aht_daily.columns = ["Call Date", "Team_AHT"]

                aht_comparison = pd.merge(
                    agent_aht_daily[["Call Date", "My_AHT"]],
                    team_aht_daily[["Call Date", "Team_AHT"]],
                    on="Call Date",
                    how="outer",
                ).sort_values("Call Date")

                if len(aht_comparison) > 0 and aht_comparison["My_AHT"].notna().any():
                    fig_aht_trend, ax_aht_trend = plt.subplots(figsize=(8, 5))
                    ax_aht_trend.plot(
                        aht_comparison["Call Date"],
                        aht_comparison["My_AHT"],
                        marker="o",
                        linewidth=2,
                        label="My AHT",
                        color="purple",
                    )
                    ax_aht_trend.plot(
                        aht_comparison["Call Date"],
                        aht_comparison["Team_AHT"],
                        marker="s",
                        linewidth=2,
                        label="Team AHT",
                        color="lavender",
                        linestyle="--",
                    )
                    ax_aht_trend.set_xlabel("Date")
                    ax_aht_trend.set_ylabel("Average Handle Time (min)")
                    ax_aht_trend.set_title("My AHT Trend vs Team Average")
                    ax_aht_trend.grid(True, alpha=0.3)
                    ax_aht_trend.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st_pyplot_safe(fig_aht_trend)
                else:
                    st.info("AHT trend data not available")

else:
    # Admin view - agent selection and comparison
    with st.expander("Agent-Specific Performance Trends", expanded=False):
        st.subheader("Agent-Specific Performance Trends")
        if len(filtered_df) > 0 and len(selected_agents) > 0:
            agent_trends_col1, agent_trends_col2 = st.columns(2)

            with agent_trends_col1:
                selected_agent_for_trend = st.selectbox(
                    "Select Agent for Trend Analysis", selected_agents
                )

                agent_data = filtered_df[
                    filtered_df["Agent"] == selected_agent_for_trend
                ]
                if len(agent_data) > 0:
                    agent_daily = (
                        agent_data.groupby(agent_data["Call Date"].dt.date)
                        .agg(
                            Avg_QA_Score=("QA Score", "mean"),
                            Call_Count=("Call ID", "count"),
                            Avg_AHT=("Call Duration (min)", "mean"),
                        )
                        .reset_index()
                    )
                    agent_daily.columns = [
                        "Date",
                        "Avg_QA_Score",
                        "Call_Count",
                        "Avg_AHT",
                    ]

                    fig_agent, ax_agent = plt.subplots(figsize=(10, 5))
                    ax_agent.plot(
                        agent_daily["Date"],
                        agent_daily["Avg_QA_Score"],
                        marker="o",
                        linewidth=2,
                        label="QA Score",
                    )
                    ax_agent.set_xlabel("Date")
                    ax_agent.set_ylabel("Average QA Score (%)")
                    ax_agent.set_title(f"Performance Trend: {selected_agent_for_trend}")
                    ax_agent.grid(True, alpha=0.3)
                    ax_agent.axhline(
                        y=alert_threshold,
                        color="r",
                        linestyle="--",
                        alpha=0.5,
                        label="Threshold",
                    )
                    ax_agent.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st_pyplot_safe(fig_agent)

                    # AHT Trend for selected agent
                    if (
                        "Call Duration (min)" in agent_data.columns
                        and agent_daily["Avg_AHT"].notna().any()
                    ):
                        st.write("**AHT Trend**")
                        fig_aht_agent, ax_aht_agent = plt.subplots(figsize=(10, 5))
                        ax_aht_agent.plot(
                            agent_daily["Date"],
                            agent_daily["Avg_AHT"],
                            marker="o",
                            linewidth=2,
                            label="AHT",
                            color="purple",
                        )
                        ax_aht_agent.set_xlabel("Date")
                        ax_aht_agent.set_ylabel("Average Handle Time (min)")
                        ax_aht_agent.set_title(f"AHT Trend: {selected_agent_for_trend}")
                        ax_aht_agent.grid(True, alpha=0.3)
                        ax_aht_agent.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st_pyplot_safe(fig_aht_agent)

            with agent_trends_col2:
                # Agent Comparison
                st.write("**Agent Comparison**")
                compare_agents = st.multiselect(
                    "Select agents to compare",
                    selected_agents,
                    default=selected_agents[: min(3, len(selected_agents))],
                )

                if len(compare_agents) > 0:
                    compare_data = filtered_df[
                        filtered_df["Agent"].isin(compare_agents)
                    ]
                    agent_comparison = (
                        compare_data.groupby("Agent")
                        .agg(
                            Avg_QA_Score=("QA Score", "mean"),
                            Total_Calls=("Call ID", "count"),
                            Pass_Rate=(
                                "Rubric Pass Count",
                                lambda x: (
                                    x.sum()
                                    / (
                                        x.sum()
                                        + compare_data.loc[
                                            x.index, "Rubric Fail Count"
                                        ].sum()
                                    )
                                    * 100
                                )
                                if (
                                    x.sum()
                                    + compare_data.loc[
                                        x.index, "Rubric Fail Count"
                                    ].sum()
                                )
                                > 0
                                else 0,
                            ),
                        )
                        .reset_index()
                    )

                    fig_compare, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    # QA Score comparison
                    agent_comparison.plot(
                        x="Agent",
                        y="Avg_QA_Score",
                        kind="bar",
                        ax=ax1,
                        color="steelblue",
                    )
                    ax1.set_ylabel("Avg QA Score (%)")
                    ax1.set_title("Average QA Score Comparison")
                    ax1.axhline(y=alert_threshold, color="r", linestyle="--", alpha=0.5)
                    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

                    # Pass Rate comparison
                    agent_comparison.plot(
                        x="Agent", y="Pass_Rate", kind="bar", ax=ax2, color="green"
                    )
                    ax2.set_ylabel("Pass Rate (%)")
                    ax2.set_title("Pass Rate Comparison")
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

                    plt.tight_layout()
                    st_pyplot_safe(fig_compare)

                    # AHT Comparison
                    if "Call Duration (min)" in compare_data.columns:
                        agent_aht_comparison = (
                            compare_data.groupby("Agent")
                            .agg(Avg_AHT=("Call Duration (min)", "mean"))
                            .reset_index()
                        )
                        if (
                            len(agent_aht_comparison) > 0
                            and agent_aht_comparison["Avg_AHT"].notna().any()
                        ):
                            st.write("**AHT Comparison**")
                            fig_aht_compare, ax_aht_compare = plt.subplots(
                                figsize=(10, 5)
                            )
                            agent_aht_comparison.plot(
                                x="Agent",
                                y="Avg_AHT",
                                kind="bar",
                                ax=ax_aht_compare,
                                color="purple",
                            )
                            ax_aht_compare.set_ylabel("Average Handle Time (min)")
                            ax_aht_compare.set_title("Average Handle Time Comparison")
                            plt.setp(
                                ax_aht_compare.xaxis.get_majorticklabels(),
                                rotation=45,
                                ha="right",
                            )
                            plt.tight_layout()
                            st_pyplot_safe(fig_aht_compare)
                else:
                    st.info("No agent data available for trend analysis")

# --- QA Score Distribution and Label Distribution ---
with st.expander("Score & Label Distribution Analysis", expanded=False):
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("QA Score Distribution")
        if "QA Score" in filtered_df.columns:
            fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
            filtered_df["QA Score"].hist(
                bins=20, ax=ax_dist, edgecolor="black", color="steelblue"
            )
            ax_dist.set_xlabel("QA Score (%)")
            ax_dist.set_ylabel("Number of Calls")
            ax_dist.set_title("Distribution of QA Scores")
            ax_dist.axvline(
                x=alert_threshold,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Threshold ({alert_threshold}%)",
            )
            ax_dist.legend()
            plt.tight_layout()
            st_pyplot_safe(fig_dist)

    with col_right:
        st.subheader("Label Distribution")
        if "Label" in filtered_df.columns:
            label_counts = filtered_df["Label"].value_counts()
            fig_label, ax_label = plt.subplots(figsize=(8, 5))
            label_counts.plot(kind="bar", ax=ax_label, color="steelblue")
            ax_label.set_xlabel("Label")
            ax_label.set_ylabel("Number of Calls")
            ax_label.set_title("Call Labels Distribution")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st_pyplot_safe(fig_label)

# --- Coaching Insights Aggregation ---
with st.expander("Coaching Insights", expanded=False):
    st.subheader("Coaching Insights")

    if "Coaching Suggestions" in filtered_df.columns:
        # Collect all coaching suggestions
        all_coaching = []
        for idx, row in filtered_df.iterrows():
            coaching = row.get("Coaching Suggestions", [])
            if isinstance(coaching, list):
                all_coaching.extend([c for c in coaching if c and str(c).strip()])
            elif isinstance(coaching, str) and coaching:
                all_coaching.append(coaching)

        if all_coaching:
            from collections import Counter

            # Function to categorize coaching suggestions (matches the logic from AHT analysis)
            def categorize_coaching(text, text_type="coaching"):
                """Categorize coaching suggestions into themes using scoring system"""
                if not text or pd.isna(text):
                    return "Other"

                text_lower = str(text).lower()

                # Product knowledge keywords
                product_keywords = [
                    "product",
                    "item",
                    "inventory",
                    "stock",
                    "availability",
                    "specification",
                    "feature",
                    "model",
                    "brand",
                    "catalog",
                    "knowledge",
                    "unfamiliar",
                    "unaware",
                    "doesn't know",
                ]

                # Communication keywords
                communication_keywords = [
                    "communication",
                    "clarity",
                    "explain",
                    "understand",
                    "confusing",
                    "unclear",
                    "misunderstand",
                    "listening",
                    "tone",
                    "empathy",
                    "rapport",
                    "connection",
                    "interrupt",
                ]

                # Hold time keywords
                hold_keywords = [
                    "hold",
                    "wait",
                    "transfer",
                    "escalate",
                    "supervisor",
                    "on hold",
                    "put on hold",
                    "long hold",
                    "excessive hold",
                ]

                # System/Technical keywords
                system_keywords = [
                    "system",
                    "technical",
                    "error",
                    "bug",
                    "glitch",
                    "software",
                    "platform",
                    "tool",
                    "database",
                    "slow",
                    "loading",
                    "crash",
                    "down",
                    "issue",
                    "problem",
                ]

                # Process/Procedure keywords
                process_keywords = [
                    "process",
                    "procedure",
                    "policy",
                    "protocol",
                    "workflow",
                    "step",
                    "order",
                    "sequence",
                    "method",
                    "approach",
                    "guideline",
                    "standard",
                    "compliance",
                ]

                # Efficiency keywords
                efficiency_keywords = [
                    "efficient",
                    "quick",
                    "fast",
                    "speed",
                    "time",
                    "streamline",
                    "optimize",
                    "reduce time",
                    "faster",
                ]

                # Count matches for each category
                product_score = sum(1 for kw in product_keywords if kw in text_lower)
                comm_score = sum(1 for kw in communication_keywords if kw in text_lower)
                hold_score = sum(1 for kw in hold_keywords if kw in text_lower)
                system_score = sum(1 for kw in system_keywords if kw in text_lower)
                process_score = sum(1 for kw in process_keywords if kw in text_lower)
                efficiency_score = sum(
                    1 for kw in efficiency_keywords if kw in text_lower
                )

                # Return category with highest score
                scores = {
                    "Product Knowledge": product_score,
                    "Communication": comm_score,
                    "Hold Time": hold_score,
                    "System/Technical": system_score,
                    "Process/Procedure": process_score,
                    "Efficiency": efficiency_score,
                }

                max_score = max(scores.values())
                if max_score > 0:
                    return max(scores, key=scores.get)
                return "Other"

            # Create coaching insights options
            coaching_insights_options = [
                "Most Common Coaching Suggestions",
                "Coaching by Category",
                "Top 10 Coaching Suggestions (Chart)",
                "All Coaching Suggestions",
            ]

            # Dropdown to select coaching insights view
            selected_insight = st.selectbox(
                "Select Coaching Insights View:",
                coaching_insights_options,
                key="coaching_insights_select",
            )

            coaching_counts = Counter(all_coaching)

            if selected_insight == "Most Common Coaching Suggestions":
                top_coaching = pd.DataFrame(
                    coaching_counts.most_common(10),
                    columns=["Coaching Suggestion", "Frequency"],
                )
                st.write("**Most Common Coaching Suggestions**")
                st.dataframe(top_coaching, width="stretch")

            elif selected_insight == "Coaching by Category":
                # Categorize all coaching suggestions
                categorized_coaching = [categorize_coaching(c) for c in all_coaching]
                category_counts = Counter(categorized_coaching)

                category_df = pd.DataFrame(
                    category_counts.most_common(), columns=["Category", "Frequency"]
                )
                category_df["Percentage"] = (
                    category_df["Frequency"] / len(all_coaching) * 100
                ).round(1)

                col_cat1, col_cat2 = st.columns(2)
                with col_cat1:
                    st.write("**Coaching Suggestions by Category**")
                    st.dataframe(category_df, width="stretch")

                with col_cat2:
                    fig_cat, ax_cat = plt.subplots(figsize=(8, 6))
                    category_df.plot(
                        x="Category",
                        y="Frequency",
                        kind="barh",
                        ax=ax_cat,
                        color="steelblue",
                    )
                    ax_cat.set_xlabel("Frequency")
                    ax_cat.set_title("Coaching Suggestions by Category")
                    plt.tight_layout()
                    st_pyplot_safe(fig_cat)

            elif selected_insight == "Top 10 Coaching Suggestions (Chart)":
                top_coaching = pd.DataFrame(
                    coaching_counts.most_common(10),
                    columns=["Coaching Suggestion", "Frequency"],
                )
                fig_coach, ax_coach = plt.subplots(figsize=(10, 6))
                top_coaching.plot(
                    x="Coaching Suggestion",
                    y="Frequency",
                    kind="barh",
                    ax=ax_coach,
                    color="orange",
                )
                ax_coach.set_xlabel("Frequency")
                ax_coach.set_title("Top 10 Coaching Suggestions")
                plt.tight_layout()
                st_pyplot_safe(fig_coach)

            elif selected_insight == "All Coaching Suggestions":
                all_coaching_df = pd.DataFrame(
                    coaching_counts.most_common(),
                    columns=["Coaching Suggestion", "Frequency"],
                )
                all_coaching_df["Percentage"] = (
                    all_coaching_df["Frequency"] / len(all_coaching) * 100
                ).round(1)
                st.write(
                    f"**All Coaching Suggestions ({len(all_coaching_df)} unique suggestions)**"
                )
                st.dataframe(all_coaching_df, width="stretch")
        else:
            st.info("No coaching suggestions found in the filtered data.")
    else:
        st.info("Coaching Suggestions column not found in the data.")

# --- Full Rubric Reference ---
with st.expander("QA Rubric Reference", expanded=False):
    st.subheader("QA Rubric Reference")
    if rubric_data:
        col_rubric_header1, col_rubric_header2 = st.columns([3, 1])
        with col_rubric_header1:
            st.info(
                f" Complete rubric with {len(rubric_data)} items. Use the tabs below to browse by section or search all items."
            )
        with col_rubric_header2:
            # Load and serve the pre-formatted Excel rubric file
            try:
                import os

                rubric_excel_path = os.path.join(
                    os.path.dirname(__file__), "Separatetab-rubric33.xlsx"
                )
                if os.path.exists(rubric_excel_path):
                    with open(rubric_excel_path, "rb") as f:
                        rubric_excel_bytes = f.read()

                    st.download_button(
                        label=" Download Rubric (Excel)",
                        data=rubric_excel_bytes,
                        file_name="QA_Rubric_v33.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                else:
                    st.warning(" Rubric Excel file not found")
                    st.info(
                        "Place 'Separatetab-rubric33.xlsx' in the dashboard directory"
                    )
            except Exception as e:
                st.error(f"Error loading rubric Excel: {e}")

        rubric_tab1, rubric_tab2 = st.tabs([" Search All Items", " Browse by Section"])

        with rubric_tab1:
            # Search interface
            col_search1, col_search2 = st.columns([3, 1])
            with col_search1:
                rubric_search = st.text_input(
                    " Search rubric",
                    placeholder="Enter code (e.g., 1.1.0), section, item, or criterion...",
                    key="full_rubric_search",
                )
            with col_search2:
                show_all = st.checkbox(
                    "Show all", value=not bool(rubric_search), key="show_all_rubric"
                )

            if rubric_search and not show_all:
                search_lower = rubric_search.lower()
                filtered_items = [
                    item
                    for item in rubric_data
                    if (
                        search_lower in item.get("code", "").lower()
                        or search_lower in item.get("section", "").lower()
                        or search_lower in item.get("item", "").lower()
                        or search_lower in item.get("criterion", "").lower()
                    )
                ]
                st.write(f"**Found {len(filtered_items)} matching items**")
            else:
                filtered_items = rubric_data
                st.write(f"**All {len(rubric_data)} rubric items**")

            # Display filtered items with pagination
            items_per_page = 20
            if len(filtered_items) > items_per_page:
                total_pages = (len(filtered_items) - 1) // items_per_page + 1
                page_num = st.number_input(
                    f"Page (1-{total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key="rubric_page",
                )
                start_idx = (page_num - 1) * items_per_page
                end_idx = start_idx + items_per_page
                display_items = filtered_items[start_idx:end_idx]
                st.caption(
                    f"Showing items {start_idx + 1}-{min(end_idx, len(filtered_items))} of {len(filtered_items)}"
                )
            else:
                display_items = filtered_items

            # Display items
            for item in display_items:
                with st.expander(
                    f"{item.get('code', 'N/A')} - {item.get('item', 'N/A')} | {item.get('section', 'N/A')} | Weight: {item.get('weight', 'N/A')}",
                    expanded=False,
                ):
                    st.write(f"**Section:** {item.get('section', 'N/A')}")
                    st.write(f"**Item:** {item.get('item', 'N/A')}")
                    st.write(f"**Criterion:** {item.get('criterion', 'N/A')}")
                    st.write(f"**Weight:** {item.get('weight', 'N/A')}")

                    # Full width display for criteria
                    st.markdown("** Pass Criteria:**")
                    st.info(item.get("pass", "N/A"))
                    st.markdown("** Fail Criteria:**")
                    st.error(item.get("fail", "N/A"))
                    st.markdown("**N/A Criteria:**")
                    st.warning(item.get("na", "N/A"))

                    if item.get("agent_script_example"):
                        st.markdown("**Agent Script Example:**")
                        st.code(item.get("agent_script_example"), language=None)

        with rubric_tab2:
            # Group by section
            sections = {}
            for item in rubric_data:
                section = item.get("section", "Other")
                if section not in sections:
                    sections[section] = []
                sections[section].append(item)

            selected_section = st.selectbox(
                "Select Section", sorted(sections.keys()), key="rubric_section"
            )

            if selected_section:
                section_items = sections[selected_section]
                st.write(f"**{len(section_items)} items in {selected_section}**")

                # Pagination for section items too
                if len(section_items) > items_per_page:
                    section_total_pages = (len(section_items) - 1) // items_per_page + 1
                    section_page_num = st.number_input(
                        f"Page (1-{section_total_pages})",
                        min_value=1,
                        max_value=section_total_pages,
                        value=1,
                        key="section_page",
                    )
                    section_start_idx = (section_page_num - 1) * items_per_page
                    section_end_idx = section_start_idx + items_per_page
                    display_section_items = section_items[
                        section_start_idx:section_end_idx
                    ]
                    st.caption(
                        f"Showing items {section_start_idx + 1}-{min(section_end_idx, len(section_items))} of {len(section_items)}"
                    )
                else:
                    display_section_items = section_items

                for item in display_section_items:
                    with st.expander(
                        f"{item.get('code', 'N/A')} - {item.get('item', 'N/A')} | Weight: {item.get('weight', 'N/A')}",
                        expanded=False,
                    ):
                        st.write(f"**Section:** {item.get('section', 'N/A')}")
                        st.write(f"**Item:** {item.get('item', 'N/A')}")
                        st.write(f"**Criterion:** {item.get('criterion', 'N/A')}")
                        st.write(f"**Weight:** {item.get('weight', 'N/A')}")

                        # Full width display for criteria
                        st.markdown("** Pass Criteria:**")
                        st.info(item.get("pass", "N/A"))
                        st.markdown("** Fail Criteria:**")
                        st.error(item.get("fail", "N/A"))
                        st.markdown("**N/A Criteria:**")
                        st.warning(item.get("na", "N/A"))

                        if item.get("agent_script_example"):
                            st.markdown("**Agent Script Example:**")
                            st.code(item.get("agent_script_example"), language=None)
    else:
        st.warning(
            " Rubric file not found. Please ensure 'Rubric_v33.json' is in the dashboard directory."
        )

st.markdown("---")

# --- Individual Call Details ---
with st.expander("Individual Call Details", expanded=False):
    st.subheader("Individual Call Details")
    if len(filtered_df) > 0:
        call_options = filtered_df["Call ID"].tolist()
        if call_options:
            st.markdown("### View Call Details")
            selected_call_id = st.selectbox(
                "Select a call to view details:",
                options=call_options,
                format_func=lambda x: f"{x[:50]}... - {filtered_df[filtered_df['Call ID'] == x]['QA Score'].iloc[0] if len(filtered_df[filtered_df['Call ID'] == x]) > 0 and 'QA Score' in filtered_df.columns and not pd.isna(filtered_df[filtered_df['Call ID'] == x]['QA Score'].iloc[0]) else 'N/A'}%",
            )

            if selected_call_id:
                call_details = filtered_df[
                    filtered_df["Call ID"] == selected_call_id
                ].iloc[0]

                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.write("**Call Information**")
                    st.write(f"**Call ID:** {call_details.get('Call ID', 'N/A')}")
                    st.write(f"**Agent:** {call_details.get('Agent', 'N/A')}")
                    st.write(f"**Date:** {call_details.get('Call Date', 'N/A')}")
                    st.write(f"**Time:** {call_details.get('Call Time', 'N/A')}")
                    qa_score = call_details.get("QA Score", "N/A")
                    st.write(
                        f"**QA Score:** {qa_score}%"
                        if isinstance(qa_score, (int, float))
                        else f"**QA Score:** {qa_score}"
                    )
                    st.write(f"**Label:** {call_details.get('Label', 'N/A')}")
                    call_dur = call_details.get("Call Duration (min)", "N/A")
                    if isinstance(call_dur, (int, float)):
                        st.write(f"**Call Length:** {call_dur:.2f} min")
                    else:
                        st.write(f"**Call Length:** {call_dur}")

                    st.write("**Reason:**")
                    st.write(call_details.get("Reason", "N/A"))

                    st.write("**Outcome:**")
                    st.write(call_details.get("Outcome", "N/A"))

                with detail_col2:
                    st.write("**Summary**")
                    st.write(call_details.get("Summary", "N/A"))

                    st.write("**Strengths**")
                    st.write(call_details.get("Strengths", "N/A"))

                    st.write("**Challenges**")
                    st.write(call_details.get("Challenges", "N/A"))

                    st.write("**Coaching Suggestions**")
                    coaching = call_details.get("Coaching Suggestions", [])
                    if isinstance(coaching, list):
                        for suggestion in coaching:
                            st.write(f"- {suggestion}")
                    else:
                        st.write(coaching if coaching else "N/A")

                # Rubric Details
                st.write("**Rubric Details**")
                rubric_details = call_details.get("Rubric Details", {})
                if isinstance(rubric_details, dict) and rubric_details:
                    rubric_df = pd.DataFrame(
                        [
                            {
                                "Code": code,
                                "Status": details.get("status", "N/A")
                                if isinstance(details, dict)
                                else "N/A",
                                "Note": details.get("note", "")
                                if isinstance(details, dict)
                                else "",
                            }
                            for code, details in rubric_details.items()
                        ]
                    )
                    st.dataframe(rubric_df)

                    # Export individual call report
                    st.markdown("---")
                    call_dur_export = call_details.get("Call Duration (min)", "N/A")
                    if isinstance(call_dur_export, (int, float)):
                        call_dur_formatted = f"{call_dur_export:.2f}"
                    else:
                        call_dur_formatted = call_dur_export

                    report_text = f"""
# QA Call Report

## Call Information
- **Call ID:** {call_details.get("Call ID", "N/A")}
- **Agent:** {call_details.get("Agent", "N/A")}
- **Date:** {call_details.get("Call Date", "N/A")}
- **Time:** {call_details.get("Call Time", "N/A")}
- **QA Score:** {call_details.get("QA Score", "N/A")}%
- **Label:** {call_details.get("Label", "N/A")}
- **Call Length:** {call_dur_formatted} min

## Call Details
**Reason:** {call_details.get("Reason", "N/A")}

**Outcome:** {call_details.get("Outcome", "N/A")}

**Summary:**
{call_details.get("Summary", "N/A")}

**Strengths:**
{call_details.get("Strengths", "N/A")}

**Challenges:**
{call_details.get("Challenges", "N/A")}

**Coaching Suggestions:**
{chr(10).join(["- " + s for s in call_details.get("Coaching Suggestions", [])]) if isinstance(call_details.get("Coaching Suggestions"), list) else call_details.get("Coaching Suggestions", "N/A")}

## Rubric Details
{chr(10).join([f"- {code}: {details.get('status', 'N/A')} - {details.get('note', '')}" for code, details in rubric_details.items() if isinstance(details, dict)])}

---
Generated by QA Dashboard • {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
                    st.download_button(
                        label=" Export Call Report (TXT)",
                        data=report_text,
                        file_name=f"call_report_{call_details.get('Call ID', 'unknown')}.txt",
                        mime="text/plain",
                    )
                else:
                    st.write("No rubric details available")

# --- Call Volume Analysis ---
with st.expander("Call Volume Analysis", expanded=False):
    st.subheader("Call Volume Analysis")
    if len(filtered_df) > 0:
        vol_col1, vol_col2 = st.columns(2)

        with vol_col1:
            st.write("**Call Volume by Agent**")
            agent_volume = (
                filtered_df.groupby("Agent")
                .agg(
                    Total_Calls=("Call ID", "count"), Avg_QA_Score=("QA Score", "mean")
                )
                .reset_index()
                .sort_values("Total_Calls", ascending=False)
            )

            fig_vol, ax_vol = plt.subplots(figsize=(10, 6))
            agent_volume.plot(
                x="Agent", y="Total_Calls", kind="bar", ax=ax_vol, color="steelblue"
            )
            ax_vol.set_ylabel("Number of Calls")
            ax_vol.set_xlabel("Agent")
            ax_vol.set_title("Call Volume by Agent")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st_pyplot_safe(fig_vol)

        with vol_col2:
            st.write("**Call Volume Over Time**")
            daily_volume = (
                filtered_df.groupby(filtered_df["Call Date"].dt.date)
                .size()
                .reset_index()
            )
            daily_volume.columns = ["Date", "Call Count"]

            fig_vol_time, ax_vol_time = plt.subplots(figsize=(10, 5))
            ax_vol_time.plot(
                daily_volume["Date"],
                daily_volume["Call Count"],
                marker="o",
                linewidth=2,
                color="purple",
            )
            ax_vol_time.set_xlabel("Date")
            ax_vol_time.set_ylabel("Number of Calls")
            ax_vol_time.set_title("Call Volume Trend Over Time")
            ax_vol_time.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st_pyplot_safe(fig_vol_time)

# --- Time of Day Analysis ---
with st.expander("Time of Day Analysis", expanded=False):
    st.subheader("Time of Day Analysis")
    if "Call Time" in filtered_df.columns and len(filtered_df) > 0:
        time_col1, time_col2 = st.columns(2)

        with time_col1:
            st.write("**QA Score by Time of Day**")
            # Extract hour from time
            filtered_df["Hour"] = pd.to_datetime(
                filtered_df["Call Time"], format="%H:%M:%S", errors="coerce"
            ).dt.hour
            time_scores = filtered_df.groupby("Hour")["QA Score"].mean().reset_index()
            time_scores = time_scores.dropna()

            if len(time_scores) > 0:
                fig_time, ax_time = plt.subplots(figsize=(10, 5))
                ax_time.plot(
                    time_scores["Hour"],
                    time_scores["QA Score"],
                    marker="o",
                    linewidth=2,
                    color="teal",
                )
                ax_time.set_xlabel("Hour of Day")
                ax_time.set_ylabel("Average QA Score (%)")
                ax_time.set_title("QA Score by Time of Day")
                ax_time.grid(True, alpha=0.3)
                ax_time.set_xticks(range(0, 24, 2))
                plt.tight_layout()
                st_pyplot_safe(fig_time)

        with time_col2:
            st.write("**Call Volume by Time of Day**")
            time_volume = filtered_df.groupby("Hour").size().reset_index()
            time_volume.columns = ["Hour", "Call Count"]
            time_volume = time_volume.dropna()

            if len(time_volume) > 0:
                fig_time_vol, ax_time_vol = plt.subplots(figsize=(10, 5))
                ax_time_vol.bar(
                    time_volume["Hour"],
                    time_volume["Call Count"],
                    color="orange",
                    alpha=0.7,
                )
                ax_time_vol.set_xlabel("Hour of Day")
                ax_time_vol.set_ylabel("Number of Calls")
                ax_time_vol.set_title("Call Volume by Time of Day")
                ax_time_vol.set_xticks(range(0, 24, 2))
                plt.tight_layout()
                st_pyplot_safe(fig_time_vol)

# --- Advanced Analytics ---
with st.expander("Advanced Analytics", expanded=False):
    st.subheader("Advanced Analytics")
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(
        [
            "Week-over-Week Comparison",
            " Agent Improvement Trends",
            " Failure Analysis",
        ]
    )

with analytics_tab1:
    st.markdown("### Week-over-Week Performance Comparison")

    if "Call Date" in filtered_df.columns and "QA Score" in filtered_df.columns:
        # Group by week
        filtered_df["Week"] = (
            pd.to_datetime(filtered_df["Call Date"]).dt.to_period("W").astype(str)
        )
        weekly_stats = (
            filtered_df.groupby("Week")
            .agg(
                {
                    "QA Score": ["mean", "count"],
                    "Rubric Pass Count": "sum",
                    "Rubric Fail Count": "sum",
                }
            )
            .reset_index()
        )

        weekly_stats.columns = [
            "Week",
            "Avg_QA_Score",
            "Call_Count",
            "Total_Pass",
            "Total_Fail",
        ]
        weekly_stats["Pass_Rate"] = (
            weekly_stats["Total_Pass"]
            / (weekly_stats["Total_Pass"] + weekly_stats["Total_Fail"])
            * 100
        ).fillna(0)
        weekly_stats = weekly_stats.sort_values("Week")

        if len(weekly_stats) > 1:
            # Calculate week-over-week change
            weekly_stats["WoW_Score_Change"] = weekly_stats["Avg_QA_Score"].diff()
            weekly_stats["WoW_PassRate_Change"] = weekly_stats["Pass_Rate"].diff()
            weekly_stats["WoW_CallCount_Change"] = weekly_stats["Call_Count"].diff()

            wow_col1, wow_col2 = st.columns(2)

            with wow_col1:
                st.write("**QA Score Week-over-Week**")
                fig_wow_score, ax_wow_score = plt.subplots(figsize=(12, 6))
                ax_wow_score.plot(
                    weekly_stats["Week"],
                    weekly_stats["Avg_QA_Score"],
                    marker="o",
                    linewidth=2,
                    label="Avg QA Score",
                )
                ax_wow_score.set_xlabel("Week")
                ax_wow_score.set_ylabel("Average QA Score (%)")
                ax_wow_score.set_title("Week-over-Week QA Score Trend")
                ax_wow_score.grid(True, alpha=0.3)
                ax_wow_score.legend()
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st_pyplot_safe(fig_wow_score)

                # Show WoW changes
                st.write("**Week-over-Week Changes**")
                wow_display = weekly_stats[
                    [
                        "Week",
                        "Avg_QA_Score",
                        "WoW_Score_Change",
                        "Call_Count",
                        "WoW_CallCount_Change",
                    ]
                ].copy()
                wow_display["WoW_Score_Change"] = wow_display["WoW_Score_Change"].apply(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
                )
                wow_display["WoW_CallCount_Change"] = wow_display[
                    "WoW_CallCount_Change"
                ].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "N/A")
                wow_display.columns = [
                    "Week",
                    "Avg QA Score",
                    "WoW Change",
                    "Call Count",
                    "WoW Count Change",
                ]
                st.dataframe(wow_display, hide_index=True)

            with wow_col2:
                st.write("**Pass Rate Week-over-Week**")
                fig_wow_pass, ax_wow_pass = plt.subplots(figsize=(12, 6))
                ax_wow_pass.plot(
                    weekly_stats["Week"],
                    weekly_stats["Pass_Rate"],
                    marker="s",
                    linewidth=2,
                    color="green",
                    label="Pass Rate",
                )
                ax_wow_pass.set_xlabel("Week")
                ax_wow_pass.set_ylabel("Pass Rate (%)")
                ax_wow_pass.set_title("Week-over-Week Pass Rate Trend")
                ax_wow_pass.grid(True, alpha=0.3)
                ax_wow_pass.legend()
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st_pyplot_safe(fig_wow_pass)
        else:
            st.info(" Need at least 2 weeks of data for week-over-week comparison")

with analytics_tab2:
    st.markdown("### Agent Improvement Trends")

    if (
        "Agent" in filtered_df.columns
        and "Call Date" in filtered_df.columns
        and "QA Score" in filtered_df.columns
    ):
        # Group by agent and week
        filtered_df["Week"] = (
            pd.to_datetime(filtered_df["Call Date"]).dt.to_period("W").astype(str)
        )
        agent_weekly = (
            filtered_df.groupby(["Agent", "Week"])
            .agg({"QA Score": "mean", "Call ID": "count"})
            .reset_index()
        )
        agent_weekly.columns = ["Agent", "Week", "Avg_QA_Score", "Call_Count"]

        # Calculate improvement (first week vs last week for each agent)
        agent_improvement = []
        for agent in agent_weekly["Agent"].unique():
            agent_data = agent_weekly[agent_weekly["Agent"] == agent].sort_values(
                "Week"
            )
            if len(agent_data) > 1:
                first_score = agent_data.iloc[0]["Avg_QA_Score"]
                last_score = agent_data.iloc[-1]["Avg_QA_Score"]
                improvement = last_score - first_score
                agent_improvement.append(
                    {
                        "Agent": agent,
                        "First Week Score": f"{first_score:.1f}%",
                        "Last Week Score": f"{last_score:.1f}%",
                        "Improvement": f"{improvement:+.1f}%",
                        "Trend": " Improving"
                        if improvement > 0
                        else " Declining"
                        if improvement < 0
                        else " Stable",
                    }
                )

        if agent_improvement:
            improvement_df = pd.DataFrame(agent_improvement)
            improvement_df = improvement_df.sort_values(
                "Improvement",
                key=lambda x: x.str.replace("%", "").str.replace("+", "").astype(float),
                ascending=False,
            )
            st.dataframe(improvement_df, hide_index=True)

            # Show trend chart for selected agents
            selected_agents_trend = st.multiselect(
                "Select agents to view trend:",
                options=filtered_df["Agent"].unique().tolist(),
                default=filtered_df["Agent"].unique().tolist()[:5]
                if len(filtered_df["Agent"].unique()) > 5
                else filtered_df["Agent"].unique().tolist(),
            )

            if selected_agents_trend:
                fig_agent_trend, ax_agent_trend = plt.subplots(figsize=(14, 6))
                for agent in selected_agents_trend:
                    agent_data = agent_weekly[
                        agent_weekly["Agent"] == agent
                    ].sort_values("Week")
                    ax_agent_trend.plot(
                        agent_data["Week"],
                        agent_data["Avg_QA_Score"],
                        marker="o",
                        label=agent,
                        linewidth=2,
                    )

                ax_agent_trend.set_xlabel("Week")
                ax_agent_trend.set_ylabel("Average QA Score (%)")
                ax_agent_trend.set_title("Agent Performance Trends Over Time")
                ax_agent_trend.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                ax_agent_trend.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st_pyplot_safe(fig_agent_trend)
        else:
            st.info(" Need multiple weeks of data per agent to show improvement trends")
    else:
        st.warning(" Missing required columns for agent improvement analysis")

with analytics_tab3:
    st.markdown("### Most Common Failure Reasons")

    if "Rubric Details" in filtered_df.columns:
        # Collect all failed rubric codes with their frequencies
        failure_reasons = {}
        for idx, row in filtered_df.iterrows():
            rubric_details = row.get("Rubric Details", {})
            if isinstance(rubric_details, dict):
                for code, details in rubric_details.items():
                    if (
                        isinstance(details, dict)
                        and details.get("status", "").lower() == "fail"
                    ):
                        if code not in failure_reasons:
                            failure_reasons[code] = {
                                "count": 0,
                                "calls": set(),
                                "notes": [],
                            }
                        failure_reasons[code]["count"] += 1
                        failure_reasons[code]["calls"].add(row.get("Call ID", ""))
                        note = details.get("note", "")
                        if note and note not in failure_reasons[code]["notes"]:
                            failure_reasons[code]["notes"].append(note)

        if failure_reasons:
            # Sort by frequency
            sorted_failures = sorted(
                failure_reasons.items(), key=lambda x: x[1]["count"], reverse=True
            )

            failure_col1, failure_col2 = st.columns([2, 1])

            with failure_col1:
                st.write("**Top Failure Reasons**")
                failure_data = []
                for code, data in sorted_failures[:20]:  # Top 20
                    failure_data.append(
                        {
                            "Rubric Code": code,
                            "Failure Count": data["count"],
                            "Affected Calls": len(data["calls"]),
                            "Sample Notes": data["notes"][0][:50] + "..."
                            if data["notes"]
                            else "N/A",
                        }
                    )

                failure_df = pd.DataFrame(failure_data)
                st.dataframe(failure_df, hide_index=True)

            with failure_col2:
                st.write("**Failure Distribution**")
                top_10_failures = sorted_failures[:10]
                codes = [item[0] for item in top_10_failures]
                counts = [item[1]["count"] for item in top_10_failures]

                fig_fail, ax_fail = plt.subplots(figsize=(8, 6))
                ax_fail.barh(range(len(codes)), counts, color="red", alpha=0.7)
                ax_fail.set_yticks(range(len(codes)))
                ax_fail.set_yticklabels(codes)
                ax_fail.set_xlabel("Failure Count")
                ax_fail.set_title("Top 10 Failure Reasons")
                plt.tight_layout()
                st_pyplot_safe(fig_fail)

            # Show detailed view for selected failure code
            selected_failure_code = st.selectbox(
                "View details for failure code:",
                options=[code for code, _ in sorted_failures],
                help="Select a failure code to see detailed information",
            )

            if selected_failure_code:
                failure_info = failure_reasons[selected_failure_code]
                st.markdown(f"### Failure Code: {selected_failure_code}")
                st.metric("Total Failures", failure_info["count"])
                st.metric("Affected Calls", len(failure_info["calls"]))

                if failure_info["notes"]:
                    st.write("**Sample Failure Notes:**")
                    for note in failure_info["notes"][:5]:  # Show first 5 notes
                        st.text_area(
                            "Note",
                            value=note,
                            height=68,
                            disabled=True,
                            key=f"note_{hash(note)}",
                            label_visibility="collapsed",
                        )
        else:
            st.info(" No failed rubric items found in the filtered data")
    else:
        st.warning(" Rubric Details column not found")

# --- Export Options ---
st.markdown("---")
st.subheader(" Export Data")

# Export Templates
if "export_templates" not in st.session_state:
    st.session_state.export_templates = {
        "Default (All Columns)": {"columns": "all", "format": "excel"},
        "Summary Only": {
            "columns": ["Call ID", "Agent", "Call Date", "QA Score", "Label"],
            "format": "excel",
        },
        "Detailed Report": {
            "columns": [
                "Call ID",
                "Agent",
                "Call Date",
                "QA Score",
                "Label",
                "Reason",
                "Outcome",
                "Summary",
            ],
            "format": "excel",
        },
    }

with st.expander(" Export Templates", expanded=False):
    template_col1, template_col2 = st.columns(2)

    with template_col1:
        st.write("**Saved Templates:**")
        selected_template = st.selectbox(
            "Choose template:",
            options=list(st.session_state.export_templates.keys()),
            help="Select a template to customize or use as-is",
        )

    with template_col2:
        if st.button("➕ Save Current as Template"):
            template_name = st.text_input("Template name:", key="new_template_name")
            if template_name:
                # Get selected columns (will be set below)
                st.session_state.export_templates[template_name] = {
                    "columns": "all",  # Will be customized
                    "format": "excel",
                }
                st.success(f"Template '{template_name}' saved!")
                st.rerun()

    # Template customization
    if selected_template:
        template = st.session_state.export_templates[selected_template]
        available_columns = filtered_df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to export:",
            options=available_columns,
            default=template.get("columns", available_columns)
            if isinstance(template.get("columns"), list)
            else available_columns,
            help="Choose which columns to include in export",
        )
        export_format = st.radio(
            "Export format:",
            ["Excel", "CSV"],
            index=0 if template.get("format") == "excel" else 1,
        )

        # Update template
        st.session_state.export_templates[selected_template]["columns"] = (
            selected_columns
        )
        st.session_state.export_templates[selected_template]["format"] = (
            export_format.lower()
        )

# Get selected template columns
selected_template_name = st.selectbox(
    "Use template:",
    ["None"] + list(st.session_state.export_templates.keys()),
    key="export_template_select",
)
if selected_template_name != "None":
    template = st.session_state.export_templates[selected_template_name]
    template_columns = template.get("columns", "all")
    if template_columns == "all":
        export_df = filtered_df.copy()
    else:
        # Only include columns that exist
        available_template_cols = [
            col for col in template_columns if col in filtered_df.columns
        ]
        export_df = filtered_df[available_template_cols].copy()
else:
    export_df = filtered_df.copy()

# Create Excel export
excel_buffer = io.BytesIO()
with ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    export_df_template = export_df.copy()

    # Clean data for export
    from datetime import datetime, timezone

    def _clean(val):
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        if isinstance(val, datetime) and val.tzinfo is not None:
            val = val.astimezone(timezone.utc).replace(tzinfo=None)
            return val
        return val

    for col in export_df_template.columns:
        export_df_template[col] = export_df_template[col].map(_clean)

    export_df_template.to_excel(writer, sheet_name="QA Data", index=False)

    # Add Agent Leaderboard sheet for admin view
    if not user_agent_id:
        # Ensure Agent column is normalized for export (safety check)
        export_filtered_df = filtered_df.copy()
        if "Agent" in export_filtered_df.columns:
            export_filtered_df["Agent"] = export_filtered_df["Agent"].apply(
                normalize_agent_id
            )

        # Recalculate agent performance for export
        agent_perf_export = (
            export_filtered_df.groupby("Agent")
            .agg(
                Total_Calls=("Call ID", "count"),
                Avg_QA_Score=("QA Score", "mean"),
                Total_Pass=("Rubric Pass Count", "sum"),
                Total_Fail=("Rubric Fail Count", "sum"),
                Avg_Call_Duration=("Call Duration (min)", "mean"),
            )
            .reset_index()
        )
        agent_perf_export["Pass_Rate"] = (
            agent_perf_export["Total_Pass"]
            / (agent_perf_export["Total_Pass"] + agent_perf_export["Total_Fail"])
            * 100
        ).fillna(0)
        agent_perf_export = agent_perf_export.sort_values(
            "Avg_QA_Score", ascending=False
        )
        agent_perf_export.to_excel(writer, sheet_name="Agent Leaderboard", index=False)

# Export buttons
export_col1, export_col2 = st.columns(2)

with export_col1:
    # Track export generation for audit (download buttons don't have callbacks)
    export_key = f"export_excel_{start_date}_{end_date}_{len(export_df)}"
    if export_key not in st.session_state:
        if current_username:
            log_audit_event(
                current_username,
                "export_data",
                f"Generated Excel export: {start_date} to {end_date}, {len(export_df)} rows",
            )
        st.session_state[export_key] = True

    st.download_button(
        label=" Download QA Data (Excel)",
        data=excel_buffer.getvalue(),
        file_name=f"qa_report_{start_date}_to_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with export_col2:
    # CSV export
    csv_buffer = io.StringIO()
    export_df_csv = filtered_df.copy()

    # Clean data for CSV export
    from datetime import datetime, timezone

    def _clean_csv(val):
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        if isinstance(val, datetime) and val.tzinfo is not None:
            val = val.astimezone(timezone.utc).replace(tzinfo=None)
            return val
        return val

    for col in export_df_csv.columns:
        export_df_csv[col] = export_df_csv[col].map(_clean_csv)

    export_df_csv.to_csv(csv_buffer, index=False)

    # Track export generation for audit
    export_csv_key = f"export_csv_{start_date}_{end_date}_{len(export_df_csv)}"
    if export_csv_key not in st.session_state:
        if current_username:
            log_audit_event(
                current_username,
                "export_data",
                f"Generated CSV export: {start_date} to {end_date}, {len(export_df_csv)} rows",
            )
        st.session_state[export_csv_key] = True

    st.download_button(
        label=" Download QA Data (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"qa_report_{start_date}_to_{end_date}.csv",
        mime="text/csv",
    )

# --- Select Calls for Export ---
st.markdown("---")
st.markdown("### Select Calls for Export")
if len(filtered_df) > 0:
    call_options = filtered_df["Call ID"].tolist()
    if call_options:
        select_all_col1, select_all_col2 = st.columns([1, 4])
        with select_all_col1:
            if st.button("Select All", key="select_all_bottom"):
                st.session_state.selected_call_ids = call_options.copy()
                st.rerun()
        with select_all_col2:
            if st.button("Clear Selection", key="clear_selection_bottom"):
                st.session_state.selected_call_ids = []
                st.rerun()

        # Multi-select for calls
        if "selected_call_ids" not in st.session_state:
            st.session_state.selected_call_ids = []

        # Filter out invalid default values (calls that no longer exist in options)
        # This prevents StreamlitAPIException when old selections are not in current options
        valid_defaults = [
            call_id
            for call_id in st.session_state.selected_call_ids
            if call_id in call_options
        ]
        if len(valid_defaults) != len(st.session_state.selected_call_ids):
            removed_count = len(st.session_state.selected_call_ids) - len(
                valid_defaults
            )
            logger.debug(
                f"Removed {removed_count} invalid call IDs from selection defaults"
            )
            st.session_state.selected_call_ids = valid_defaults

        selected_for_export = st.multiselect(
            "Choose calls to export (you can select multiple):",
            options=call_options,
            default=valid_defaults,
            key="select_calls_export_bottom",
            format_func=lambda x: f"{x[:50]}... - {filtered_df[filtered_df['Call ID'] == x]['QA Score'].iloc[0] if len(filtered_df[filtered_df['Call ID'] == x]) > 0 and 'QA Score' in filtered_df.columns and not pd.isna(filtered_df[filtered_df['Call ID'] == x]['QA Score'].iloc[0]) else 'N/A'}%",
        )
        st.session_state.selected_call_ids = selected_for_export

        if selected_for_export:
            st.info(f" {len(selected_for_export)} call(s) selected for export")

# Export selected individual calls (if any are selected)
if "selected_call_ids" not in st.session_state:
    st.session_state.selected_call_ids = []

if len(filtered_df) > 0:
    st.caption("Export the calls selected above")

    # Show selected calls count
    if st.session_state.selected_call_ids:
        selected_calls_df = filtered_df[
            filtered_df["Call ID"].isin(st.session_state.selected_call_ids)
        ]
        st.info(f" {len(selected_calls_df)} call(s) selected for export")

        export_selected_col1, export_selected_col2 = st.columns(2)

        with export_selected_col1:
            # Excel export for selected calls
            selected_excel_buffer = io.BytesIO()
            with ExcelWriter(selected_excel_buffer, engine="xlsxwriter") as writer:
                selected_export_df = selected_calls_df.copy()
                for col in selected_export_df.columns:
                    selected_export_df[col] = selected_export_df[col].map(_clean)
                selected_export_df.to_excel(
                    writer, sheet_name="Selected Calls", index=False
                )

            st.download_button(
                label=" Export Selected (Excel)",
                data=selected_excel_buffer.getvalue(),
                file_name=f"selected_calls_{start_date}_to_{end_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with export_selected_col2:
            # CSV export for selected calls
            selected_csv_buffer = io.StringIO()
            selected_export_df_csv = selected_calls_df.copy()
            for col in selected_export_df_csv.columns:
                selected_export_df_csv[col] = selected_export_df_csv[col].map(
                    _clean_csv
                )
            selected_export_df_csv.to_csv(selected_csv_buffer, index=False)

            st.download_button(
                label=" Export Selected (CSV)",
                data=selected_csv_buffer.getvalue(),
                file_name=f"selected_calls_{start_date}_to_{end_date}.csv",
                mime="text/csv",
            )

        if st.button(" Clear Selection"):
            st.session_state.selected_call_ids = []
            st.rerun()
    else:
        st.caption(" No calls selected. Select calls above.")

st.markdown("---")
st.markdown(
    "Built with ❤️ by [Valence](https://www.getvalenceai.com) | QA Dashboard © 2026"
)
