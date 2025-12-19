import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime, timedelta, date
from pandas import ExcelWriter
from matplotlib.backends.backend_pdf import PdfPages
import streamlit_authenticator as stauth
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import json
from xlsxwriter.utility import xl_rowcol_to_cell
import time
import re
import os
import sys
import logging
import shutil
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from pdf_parser import parse_pdf_from_bytes
from utils import log_audit_event, check_session_timeout, load_metrics, track_feature_usage
import warnings

# File locking imports (platform-specific)
try:
    if sys.platform == 'win32':
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
except Exception as e:
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.debug(f"Logging initialized. Log file: {log_file}")

# Suppress noisy matplotlib categorical-unit warnings about string data that look like numbers/dates
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.category')

# Suppress matplotlib.category INFO level logging messages (categorical units warnings)
logging.getLogger('matplotlib.category').setLevel(logging.WARNING)

# --- File Locking (must be defined before load_metrics/save_metrics) ---
class LockTimeoutError(Exception):
    """Raised when file lock acquisition times out."""
    pass

@contextmanager
def cache_file_lock(filepath, timeout=30):
    """Acquire file lock for cache operations. Prevents concurrent reads/writes.
    
    Args:
        filepath: Path to the file to lock
        timeout: Maximum time to wait for lock (seconds). None = wait indefinitely.
    
    Yields:
        Locked file handle (or None if locking unavailable)
    
    Raises:
        LockTimeoutError: If lock cannot be acquired within timeout period (only if timeout is not None)
    """
    lock_path = filepath.with_suffix(filepath.suffix + '.lock')
    lock_file = None
    lock_acquired = False
    
    if not HAS_FILE_LOCKING:
        # Fallback: no locking available, just yield
        yield None
        return
    
    try:
        # Create lock file
        lock_file = open(lock_path, 'w')
        start_time = time.time()
        
        # Try to acquire lock - wait indefinitely if timeout is None
        if timeout is None:
            # Wait indefinitely until lock is acquired
            while True:
                try:
                    if sys.platform == 'win32':
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
        else:
            # Wait with timeout
            while time.time() - start_time < timeout:
                try:
                    if sys.platform == 'win32':
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
            raise LockTimeoutError(f"Could not acquire lock for {filepath} within {timeout}s. Another process may be accessing the cache file.")
        
        yield lock_file
        
    except LockTimeoutError:
        # Re-raise LockTimeoutError - don't catch it, let it propagate to callers
        raise
    except Exception as e:
        # Only catch other unexpected errors, not LockTimeoutError
        logger.warning(f"⚠️ Error acquiring file lock: {e}, proceeding without lock")
        yield None
    finally:
        # Release lock and close file
        if lock_file:
            # Only unlock if lock was actually acquired
            if lock_acquired:
                try:
                    if sys.platform == 'win32':
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
    
    temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
    
    # Clean up any existing temp file from previous failed attempts
    if temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors
    
    for attempt in range(max_retries):
        try:
            # Write to temporary file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=str, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic rename (replaces existing file atomically on most systems)
            os.replace(temp_path, filepath)
            return
            
        except (IOError, OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Atomic write attempt {attempt + 1} failed: {e}, retrying...")
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
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    if not isinstance(metrics_data, dict):
                        logger.warning(
                            f"⚠️ Metrics file contains invalid data structure: "
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
        logger.warning(f"⚠️ Could not acquire lock for metrics file, skipping save")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")

def track_feature_usage(feature_name):
    """Track feature usage."""
    metrics = load_metrics()
    if "features_used" not in metrics:
        metrics["features_used"] = {}
    metrics["features_used"][feature_name] = metrics["features_used"].get(feature_name, 0) + 1
    save_metrics(metrics)

def track_error(error_type, error_message):
    """Track errors for alerting on repeated failures."""
    metrics = load_metrics()
    if "errors" not in metrics:
        metrics["errors"] = {}
    
    error_key = f"{error_type}:{error_message[:50]}"
    if error_key not in metrics["errors"]:
        metrics["errors"][error_key] = {"count": 0, "first_seen": datetime.now().isoformat(), "last_seen": None}
    
    metrics["errors"][error_key]["count"] += 1
    metrics["errors"][error_key]["last_seen"] = datetime.now().isoformat()
    save_metrics(metrics)
    
    # Log error
    logger.error(f"{error_type}: {error_message}")
    
    # Alert on repeated failures (5+ occurrences)
    if metrics["errors"][error_key]["count"] >= 5:
        logger.warning(f"ALERT: Repeated failure detected - {error_key} (count: {metrics['errors'][error_key]['count']})")

# Initialize session metrics
if 'session_started' not in st.session_state:
    metrics = load_metrics()
    metrics["sessions"] = metrics.get("sessions", 0) + 1
    save_metrics(metrics)
    st.session_state.session_started = True
    logger.info(f"New session started. Total sessions: {metrics['sessions']}")

st.set_page_config(page_title="Emotion Dashboard", layout="wide")
logger.debug("Page config set, starting app initialization...")

# Show immediate feedback - app is loading
initial_status = st.empty()
initial_status.info("🔄 **Initializing dashboard...** Please wait.")

# Initialize S3 client from secrets (but don't test connection yet - do that after login)
logger.debug("Initializing S3 client from secrets...")
try:
    # Check if secrets are available
    if "s3" not in st.secrets:
        raise KeyError("s3 section not found in secrets")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
        region_name=st.secrets["s3"].get("region_name", "us-east-1")
    )
    s3_bucket_name = st.secrets["s3"]["bucket_name"]
    s3_prefix = st.secrets["s3"].get("prefix", "")  # Optional prefix/folder path
    logger.debug(f"S3 client initialized. Bucket: {s3_bucket_name}, Prefix: {s3_prefix}")
    initial_status.empty()  # Clear initial status once S3 client is ready
except KeyError as e:
    logger.error(f"Missing S3 configuration in secrets: {e}")
    initial_status.empty()
    st.error(f"❌ Missing S3 configuration in secrets: {e}")
    st.error("Please check your `.streamlit/secrets.toml` file and ensure all S3 fields are set.")
    st.error(f"**Current working directory:** `{os.getcwd()}`")
    st.error(f"**Expected secrets path:** `.streamlit/secrets.toml` in the project directory")
    st.error(f"**Make sure you're running Streamlit from the project root directory:** `{PROJECT_ROOT}`")
    st.stop()
except Exception as e:
    logger.exception(f"Error initializing S3 client: {e}")
    initial_status.empty()
    st.error(f"❌ Error initializing S3 client: {e}")
    st.error("Please check your AWS credentials and try again.")
    st.stop()

def load_all_calls_internal(max_files=None):
    """
    Internal function to load PDF files from S3 bucket.
    Returns tuple: (call_data_list, error_message)
    
    Args:
        max_files: Maximum number of PDFs to load (None = load all)
    """
    try:
        all_calls = []
        
        # Configure S3 client with timeout
        import botocore.config
        config = botocore.config.Config(
            connect_timeout=10,
            read_timeout=30,
            retries={'max_attempts': 2}
        )
        s3_client_with_timeout = boto3.client(
            's3',
            aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
            region_name=st.secrets["s3"].get("region_name", "us-east-1"),
            config=config
        )
        
        # List all PDF files in the S3 bucket
        try:
            paginator = s3_client_with_timeout.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=s3_bucket_name,
                Prefix=s3_prefix,
                MaxKeys=1000  # Limit to prevent huge lists
            )
            
            # Collect all PDF file keys with their modification dates
            pdf_keys_with_dates = []
            for page in pages:
                if isinstance(page, dict) and 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj.get('Key')
                        if not key or not key.lower().endswith('.pdf'):
                            continue
                        # Store key and last modified date for sorting
                        last_modified = obj.get('LastModified', datetime.min)
                        # Ensure LastModified is a datetime object for safe sorting
                        if not isinstance(last_modified, datetime):
                            last_modified = datetime.min
                        pdf_keys_with_dates.append({
                            'key': key,
                            'last_modified': last_modified
                        })
        except Exception as e:
            return [], f"Error listing S3 objects: {e}"
        
        if not pdf_keys_with_dates:
            return [], "No PDF files found in S3 bucket"
        
        # Sort by modification date (most recent first)
        pdf_keys_with_dates.sort(key=lambda x: x['last_modified'], reverse=True)
        
        # Store total count before limiting
        total_pdfs = len(pdf_keys_with_dates)
        
        # Limit the number of files if specified (most recent files first)
        if max_files and max_files > 0:
            pdf_keys_with_dates = pdf_keys_with_dates[:max_files]
        
        # Extract just the keys for processing
        pdf_keys = [item['key'] for item in pdf_keys_with_dates]
        
        # Download and parse PDFs in parallel for faster processing
        errors = []
        total = len(pdf_keys)
        
        def process_pdf_with_retry(key, max_retries=3):
            """Process a single PDF with retry logic for transient errors."""
            import time
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Download PDF from S3 (with timeout per file)
                    response = s3_client_with_timeout.get_object(Bucket=s3_bucket_name, Key=key)
                    pdf_bytes = response['Body'].read()
                    
                    # Extract filename from key
                    filename = key.split('/')[-1]
                    
                    # Parse PDF
                    parsed_data = parse_pdf_from_bytes(pdf_bytes, filename)
                    
                    if parsed_data:
                        # Add S3 metadata - normalize key for consistent comparison
                        # Store normalized key but keep original for S3 operations
                        normalized_key = key.strip('/')
                        parsed_data['_id'] = normalized_key
                        parsed_data['_s3_key'] = normalized_key
                        return parsed_data, None
                    else:
                        return None, f"Failed to parse {filename}"
                        
                except Exception as e:
                    last_error = e
                    # Retry on transient errors (network issues, timeouts)
                    if attempt < max_retries - 1:
                        # Exponential backoff: wait 0.5s, 1s, 2s
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    else:
                        return None, f"{key}: {str(e)}"
            
            return None, f"{key}: {str(last_error)}"
        
        # Use parallel processing for faster parsing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Initialize progress tracking in session state
        # Try to restore progress from disk cache if it exists (for partial cache continuation)
        restored_progress = False
        if CACHE_FILE.exists():
            try:
                # Use file locking for consistency (even though this is just metadata)
                with cache_file_lock(CACHE_FILE, timeout=1):
                    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    if cached_data.get('partial', False):
                        partial_processed = cached_data.get('processed', 0)
                        partial_total = cached_data.get('total', 0)
                        if partial_total > 0 and partial_processed > 0:
                            # Restore progress from cache
                            if 'pdf_processing_progress' not in st.session_state:
                                st.session_state.pdf_processing_progress = {
                                    'processed': partial_processed,
                                    'total': partial_total,
                                    'errors': 0,
                                    'processing_start_time': None
                                }
                                restored_progress = True
                                logger.info(f"📂 Restored progress from cache: {partial_processed}/{partial_total} files")
            except (LockTimeoutError, json.JSONDecodeError, FileNotFoundError, Exception):
                # Silently fail - progress restoration is optional
                pass
        
        if not restored_progress:
            if 'pdf_processing_progress' not in st.session_state:
                st.session_state.pdf_processing_progress = {'processed': 0, 'total': total, 'errors': 0, 'processing_start_time': None}
        else:
            # Progress was restored from cache - update total and errors, but keep processed count
            st.session_state.pdf_processing_progress['total'] = total
            st.session_state.pdf_processing_progress['errors'] = 0
        # Track actual processing start time
        processing_start_time = time.time()
        st.session_state.pdf_processing_progress['processing_start_time'] = processing_start_time
        
        # Process PDFs in smaller batches to reduce lock contention with large cache files
        BATCH_SIZE = 50  # Reduced from 100 to reduce lock contention
        DASHBOARD_UPDATE_INTERVAL = 500
        
        logger.info(f"📥 Starting to process {total} PDF files in batches of {BATCH_SIZE} (this will take 10-20 minutes)")
        
        last_dashboard_update = 0
        
        # Process in batches
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch_keys = pdf_keys[batch_start:batch_end]
            
            batch_num = batch_start//BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1)//BATCH_SIZE
            if batch_num % 5 == 0 or batch_num == 1 or batch_num == total_batches:
                logger.info(f"📦 Processing batch {batch_num}/{total_batches}: files {batch_start+1}-{batch_end} of {total}")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_key = {executor.submit(process_pdf_with_retry, key): key for key in batch_keys}
                
                for future in as_completed(future_to_key):
                    try:
                        parsed_data, error = future.result(timeout=60)  # 60 second timeout per file
                    except Exception as e:
                        # Handle both TimeoutError and concurrent.futures.TimeoutError
                        from concurrent.futures import TimeoutError as FuturesTimeoutError
                        if isinstance(e, (TimeoutError, FuturesTimeoutError)):
                            key = future_to_key.get(future, 'Unknown')
                            logger.error(f"⏱️ Timeout processing {key}: {e}")
                            errors.append(f"{key}: Processing timeout (60s)")
                            st.session_state.pdf_processing_progress['errors'] += 1
                            st.session_state.pdf_processing_progress['processed'] += 1
                            continue
                        else:
                            # Unexpected error in future execution
                            logger.error(f"❌ Unexpected error in future: {e}")
                            errors.append(f"Unknown: {str(e)}")
                            st.session_state.pdf_processing_progress['errors'] += 1
                            st.session_state.pdf_processing_progress['processed'] += 1
                            continue
                    
                    if parsed_data:
                        all_calls.append(parsed_data)
                    elif error:
                        errors.append(error)
                        st.session_state.pdf_processing_progress['errors'] += 1
                    
                    st.session_state.pdf_processing_progress['processed'] += 1
                        
                    processed = st.session_state.pdf_processing_progress['processed']
                    if processed % 100 == 0:
                        current_time = time.time()
                        elapsed = current_time - processing_start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = (total - processed) / rate if rate > 0 else 0
                        logger.info(f"📊 Progress: {processed}/{total} files processed ({processed*100//total if total > 0 else 0}%), {len(all_calls)} successful, {len(errors)} errors. Rate: {rate:.1f} files/sec, ETA: {remaining/60:.1f} min")
                        
                        # INCREMENTAL SAVE: Save to disk cache every 100 files
                        # This prevents losing all progress if app restarts during long loads
                        # Try to get last save time from disk cache metadata first (survives restarts)
                        last_save_time = 0
                        try:
                            # Read last_save_time with locking to prevent concurrent access
                            if CACHE_FILE.exists():
                                with cache_file_lock(CACHE_FILE, timeout=2):
                                    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                                        cached_data = json.load(f)
                                        last_save_time = cached_data.get('last_save_time', 0)
                        except LockTimeoutError:
                            # Lock timeout - skip reading last_save_time, will use session state fallback
                            pass
                        except Exception:
                            pass
                        # Fallback to session state if not in cache
                        if not last_save_time:
                            last_save_time = getattr(st.session_state, '_last_incremental_save_time', 0)
                        time_since_last_save = time.time() - last_save_time
                        # Save every 100 files OR every 1 minute (more frequent saves to prevent data loss on restarts)
                        should_save = (processed % 100 == 0) or (time_since_last_save > 60)
                        if should_save:
                            try:
                                # Deduplicate before incremental save to prevent cache bloat
                                calls_to_save = deduplicate_calls(all_calls.copy())
                                if len(calls_to_save) < len(all_calls):
                                    logger.info(f"🔍 Deduplicated before incremental save: {len(all_calls)} → {len(calls_to_save)} unique calls")
                                
                                # Use atomic write with locking via save_cached_data_to_disk
                                save_cached_data_to_disk(
                                    calls_to_save,
                                    errors.copy(),
                                    partial=True,
                                    processed=processed,
                                    total=total
                                )
                                st.session_state._last_incremental_save_time = time.time()
                                logger.info(f"💾 Incremental save: Saved {len(calls_to_save)} calls to disk cache ({processed}/{total} = {processed*100//total if total > 0 else 0}% complete - progress protected)")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed incremental save: {e}")
                    
                    # Update dashboard progress in session state (without rerun to avoid cache corruption)
                    processed = st.session_state.pdf_processing_progress['processed']
                    if processed >= last_dashboard_update + DASHBOARD_UPDATE_INTERVAL:
                        logger.info(f"🔄 Dashboard update: {processed} calls processed, updating progress...")
                        last_dashboard_update = processed
                        
                        # Update session state with progress (no rerun to prevent cache corruption)
                        # Progress will be visible on next natural rerun (user interaction or completion)
                        st.session_state['last_progress_update'] = {
                            'processed': processed,
                            'total': total,
                            'timestamp': time.time()
                        }
                        
                        # Note: Removed st.rerun() to prevent cache corruption from concurrent reads/writes
                        # Progress updates will be visible when processing completes or user interacts
        
        elapsed_total = time.time() - processing_start_time
        logger.info(f"✅ Completed processing {total} files in {elapsed_total/60:.1f} minutes. Success: {len(all_calls)}, Errors: {len(errors)}")
        
        # Deduplicate before returning (in case of any duplicates from S3 or processing)
        original_count = len(all_calls)
        all_calls = deduplicate_calls(all_calls)
        if len(all_calls) < original_count:
            logger.info(f"🔍 Deduplication after S3 load: {original_count} calls → {len(all_calls)} unique calls (removed {original_count - len(all_calls)} duplicates)")
        
        # Sort by call_date if available (already sorted by S3 date, but this ensures call_date order)
        try:
            all_calls.sort(key=lambda x: x.get('call_date', datetime.min) if isinstance(x.get('call_date'), datetime) else datetime.min, reverse=True)
        except:
            pass  # If sorting fails, just return unsorted
        
        # Track processed S3 keys in session state (for smart refresh)
        if 'processed_s3_keys' not in st.session_state:
            st.session_state['processed_s3_keys'] = set()
        processed_keys = {call.get('_s3_key') for call in all_calls if call.get('_s3_key')}
        st.session_state['processed_s3_keys'].update(processed_keys)
        
        # Store actual processing time in session state
        if 'processing_start_time' in st.session_state.pdf_processing_progress:
            actual_processing_time = time.time() - st.session_state.pdf_processing_progress['processing_start_time']
            st.session_state['last_actual_processing_time'] = actual_processing_time
            st.session_state['last_processing_file_count'] = len(all_calls)
        
        # Return with info about total vs loaded
        return all_calls, errors
        
    except NoCredentialsError as e:
        error_msg = "AWS credentials not found. Please configure S3 credentials in secrets."
        track_error("S3_NoCredentials", str(e))
        return [], error_msg
    except ClientError as e:
        error_code = 'Unknown'
        if hasattr(e, 'response') and isinstance(e.response, dict):
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchBucket':
            error_msg = f"S3 bucket '{s3_bucket_name}' not found."
            track_error(f"S3_{error_code}", error_msg)
            return [], error_msg
        elif error_code == 'AccessDenied':
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
            's3',
            aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
            region_name=st.secrets["s3"].get("region_name", "us-east-1")
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
            logger.warning(f"⚠️ Corrupted cache file is very large ({file_size / 1024 / 1024:.1f}MB), recovery may be slow")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to find call_data array in the content
        # Look for patterns like "call_data": [ ... ]
        
        # Try to extract call_data array
        call_data_match = re.search(r'"call_data"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        if call_data_match:
            # Try to parse just the array portion
            array_content = '[' + call_data_match.group(1) + ']'
            try:
                # Try to parse as JSON array
                recovered_calls = json.loads(array_content)
                if isinstance(recovered_calls, list):
                    # Validate that list contains dicts (call objects)
                    if len(recovered_calls) > 0 and not isinstance(recovered_calls[0], dict):
                        logger.warning("⚠️ Recovered data is not in expected format")
                        return None, None
                    logger.info(f"✅ Recovered {len(recovered_calls)} calls from corrupted cache (recovery complete)")
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
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1
                        if brace_depth == 0:
                            end_pos = i + 1
                            break
            
            if brace_depth == 0 and end_pos > start_pos:
                # Extract the complete object
                try:
                    obj_str = content[start_pos:end_pos]
                    call_obj = json.loads(obj_str)
                    if isinstance(call_obj, dict) and call_obj.get('_s3_key'):
                        recovered_calls.append(call_obj)
                except json.JSONDecodeError:
                    continue
        
        if recovered_calls:
            logger.info(f"✅ Recovered {len(recovered_calls)} calls from corrupted cache (partial recovery complete)")
            return recovered_calls, []
        
        return None, None
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to recover partial data: {e}")
        return None, None

def deduplicate_calls(call_data):
    """Remove duplicate calls based on _s3_key or _id. Keeps the first occurrence."""
    if not call_data:
        return []
    
    seen_keys = set()
    deduplicated = []
    duplicates_count = 0
    
    for call in call_data:
        # Skip non-dict items (defensive programming)
        if not isinstance(call, dict):
            logger.warning(f"⚠️ Skipping non-dict item in call_data: {type(call).__name__}")
            continue
        
        # Use _s3_key as primary identifier, fall back to _id
        key = call.get('_s3_key') or call.get('_id') or call.get('Call ID')
        if key and key not in seen_keys:
            seen_keys.add(key)
            deduplicated.append(call)
        elif key:
            duplicates_count += 1
    
    if duplicates_count > 0:
        logger.warning(f"🔍 Removed {duplicates_count} duplicate calls (kept {len(deduplicated)} unique calls)")
    else:
        logger.debug(f"🔍 Deduplication check: No duplicates found, all {len(deduplicated)} calls are unique")
    
    return deduplicated

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
    refresh_in_progress = st.session_state.get('refresh_in_progress', False)
    if refresh_in_progress:
        cache_key = '_disk_cache_during_refresh'
        if cache_key in st.session_state:
            cached_result, cached_timestamp = st.session_state[cache_key]
            # Use cached result if less than 5 seconds old (refresh updates frequently)
            if time.time() - cached_timestamp < 5:
                logger.debug("📦 Using session-cached disk cache result during refresh")
                return cached_result
    
    # CRITICAL: Check session state for S3 cache first (to avoid duplicate S3 loads)
    # If load_all_calls_cached() already loaded from S3, reuse that result
    s3_cache_key = '_s3_cache_result'
    if s3_cache_key in st.session_state:
        cached_result = st.session_state[s3_cache_key]
        logger.debug("📦 Using session-cached S3 result in load_cached_data_from_disk() - avoiding duplicate S3 load")
        
        # Try to sync to local disk if file doesn't exist or is older (non-blocking, don't reload from S3)
        # Just check if disk cache exists and is recent, if not, we'll sync on next S3 load
        # This avoids another S3 load just for syncing
        
        # Cache result in session state during refresh
        result = cached_result
        if refresh_in_progress:
            cache_key = '_disk_cache_during_refresh'
            st.session_state[cache_key] = (result, time.time())
        
        return result
    
    # Try loading from S3 first (only if not already in session state)
    s3_client, s3_bucket = get_s3_client_and_bucket()
    if s3_client and s3_bucket:
        try:
            response = s3_client.get_object(Bucket=s3_bucket, Key=S3_CACHE_KEY)
            cached_data = json.loads(response['Body'].read().decode('utf-8'))
            if not isinstance(cached_data, dict):
                logger.warning(f"⚠️ S3 cache contains invalid data structure: {type(cached_data).__name__}, expected dict")
            else:
                call_data = cached_data.get('call_data', [])
                errors = cached_data.get('errors', [])
                cache_timestamp = cached_data.get('timestamp', None)
                cache_count = len(call_data)
                is_partial = cached_data.get('partial', False)
                
                if is_partial:
                    processed = cached_data.get('processed', 0)
                    total = cached_data.get('total', 0)
                    logger.info(
                        f"☁️ Found PARTIAL cache in S3 with {cache_count} calls "
                        f"({processed}/{total} = {processed*100//total if total > 0 else 0}% complete, "
                        f"saved at {cache_timestamp})"
                    )
                else:
                    logger.info(
                        f"☁️ Found COMPLETE cache in S3 with {cache_count} calls (saved at {cache_timestamp})"
                    )
                
                # Also sync to local disk for faster subsequent loads
                try:
                    with cache_file_lock(CACHE_FILE, timeout=5):
                        atomic_write_json(CACHE_FILE, cached_data)
                        logger.debug(f"💾 Synced S3 cache to local disk: {CACHE_FILE}")
                except Exception as sync_error:
                    logger.debug(f"Could not sync S3 cache to local disk: {sync_error}")
                
                # Cache result in session state during refresh to reduce repeated S3 loads
                result = (call_data, errors)
                if refresh_in_progress:
                    cache_key = '_disk_cache_during_refresh'
                    st.session_state[cache_key] = (result, time.time())
                
                return result
        except ClientError as s3_error:
            error_code = s3_error.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchKey':
                logger.debug(f"📂 No cache found in S3: s3://{s3_bucket}/{S3_CACHE_KEY}")
            else:
                logger.warning(f"⚠️ Failed to load cache from S3 (will try local disk): {s3_error}")
        except Exception as s3_error:
            logger.warning(f"⚠️ Failed to load cache from S3 (will try local disk): {s3_error}")
    
    # Fall back to local disk if S3 unavailable or not found
    if not CACHE_FILE.exists():
        return None, None
    
    # Retry logic for transient errors
    for attempt in range(max_retries):
        try:
            # Use file locking to prevent concurrent reads during writes
            # CRITICAL FIX: Remove timeout - wait indefinitely for lock
            # In single-process Streamlit app, locks are released when operations complete
            # No risk of permanent deadlock, so we can wait as long as needed
            try:
                with cache_file_lock(CACHE_FILE, timeout=None):
                    logger.debug(f"📂 Checking persistent cache file: {CACHE_FILE}")
                    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        if not isinstance(cached_data, dict):
                            logger.error(
                                f"⚠️ Cache file contains invalid data structure: "
                                f"{type(cached_data).__name__}, expected dict"
                            )
                            return None, None
                        call_data = cached_data.get('call_data', [])
                        errors = cached_data.get('errors', [])
                        cache_timestamp = cached_data.get('timestamp', None)
                        cache_count = len(call_data)
                        
                        # OPTIMIZATION: Cache should already be deduplicated (done in save_cached_data_to_disk)
                        # No need to deduplicate here - reduces redundant processing
                        is_partial = cached_data.get('partial', False)
                        
                        if is_partial:
                            processed = cached_data.get('processed', 0)
                            total = cached_data.get('total', 0)
                            logger.info(
                                f"📦 Found PARTIAL cache with {cache_count} calls "
                                f"({processed}/{total} = {processed*100//total if total > 0 else 0}% complete, "
                                f"saved at {cache_timestamp})"
                            )
                            logger.debug("💡 This partial cache will be used if it's more complete than a fresh load")
                        else:
                            logger.info(
                                f"✅ Found COMPLETE persistent cache with {cache_count} calls (saved at {cache_timestamp})"
                            )
                        
                        # Cache result in session state during refresh to reduce repeated disk loads
                        result = (call_data, errors)
                        if refresh_in_progress:
                            cache_key = '_disk_cache_during_refresh'
                            st.session_state[cache_key] = (result, time.time())
                        
                        return result
            except LockTimeoutError as e:
                # Lock timeout - retry with exponential backoff
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Lock timeout on cache read attempt {attempt + 1}: {e}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"❌ Failed to acquire lock for cache read after {max_retries} attempts: {e}")
                    return None, None
                    
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to load persistent cache: Corrupted JSON file - {e}")
            
            # Try to recover partial data before deleting
            # Note: Recovery operations need lock protection to prevent concurrent access
            # Read corrupted file with lock protection
            recovered_calls, recovered_errors = None, None
            try:
                with cache_file_lock(CACHE_FILE, timeout=10):
                    recovered_calls, recovered_errors = recover_partial_json(CACHE_FILE)
            except LockTimeoutError as e:
                logger.error(f"❌ Lock timeout while reading corrupted cache for recovery: {e}")
            except Exception as recovery_read_error:
                logger.warning(f"⚠️ Failed to read corrupted cache for recovery: {recovery_read_error}")
            
            if recovered_calls:
                # Save recovered data to a new cache file (with lock protection)
                try:
                    try:
                        with cache_file_lock(CACHE_FILE, timeout=10):
                            backup_path = CACHE_FILE.with_suffix('.json.corrupted')
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            backup_path = CACHE_FILE.parent / f"cached_calls_data.json.corrupted.{timestamp}"
                            shutil.copy2(CACHE_FILE, backup_path)
                            logger.info(f"💾 Backed up corrupted cache to {backup_path}")
                            
                            # Save recovered data
                            recovered_data = {
                                'call_data': recovered_calls,
                                'errors': recovered_errors,
                                'timestamp': datetime.now().isoformat(),
                                'count': len(recovered_calls),
                                'partial': True,  # Mark as partial since we lost metadata
                                'recovered_from_corruption': True
                            }
                            atomic_write_json(CACHE_FILE, recovered_data)
                            logger.info(
                                f"✅ Recovered and saved {len(recovered_calls)} calls from corrupted cache "
                                f"(recovery and save complete)"
                            )
                            return recovered_calls, recovered_errors
                    except LockTimeoutError as e:
                        logger.error(f"❌ Lock timeout while saving recovered data: {e}")
                        raise  # Re-raise to trigger outer exception handler
                    
                except Exception as recovery_error:
                    logger.error(f"❌ Failed to save recovered data: {recovery_error}")
            
            # If recovery failed, backup and delete corrupted cache (with lock protection)
            try:
                try:
                    with cache_file_lock(CACHE_FILE, timeout=10):
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        backup_path = CACHE_FILE.parent / f"cached_calls_data.json.corrupted.{timestamp}"
                        shutil.copy2(CACHE_FILE, backup_path)
                        CACHE_FILE.unlink()
                        logger.info(
                            f"💾 Backed up corrupted cache to {backup_path} and deleted original"
                        )
                except LockTimeoutError as e:
                    logger.error(f"❌ Lock timeout while backing up corrupted cache: {e}")
            except Exception as backup_error:
                logger.error(f"❌ Failed to backup corrupted cache: {backup_error}")
            
            # Don't retry on JSON decode errors - file is corrupted
            return None, None
            
        except (IOError, OSError, PermissionError) as e:
            # Transient errors - retry
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Cache read attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"❌ Failed to load persistent cache after {max_retries} attempts: {e}")
                return None, None
                
        except Exception as e:
            # Other errors - log and return None
            logger.warning(f"⚠️ Failed to load persistent cache: {e}")
    
    # Cache result in session state during refresh to reduce repeated S3/disk loads
    result = None, None
    if refresh_in_progress:
        cache_key = '_disk_cache_during_refresh'
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
        
        # Deduplicate before saving to prevent cache bloat
        if call_data:
            original_count = len(call_data)
            call_data = deduplicate_calls(call_data)
            if len(call_data) < original_count:
                logger.info(f"🔍 Deduplicated before save: {original_count} → {len(call_data)} unique calls")
        
        cache_data = {
            'call_data': call_data,
            'errors': errors,
            'timestamp': datetime.now().isoformat(),
            'last_save_time': time.time(),  # Track last save time
            'count': len(call_data),
            'partial': partial  # Mark as partial or complete
        }
        
        # Add progress info for partial saves
        if partial:
            cache_data['processed'] = processed
            cache_data['total'] = total
        
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
                    logger.warning(f"⚠️ Lock timeout on save attempt {lock_attempt + 1}/{max_lock_retries}: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to acquire lock for cache save after {max_lock_retries} attempts: {e}")
                    raise  # Re-raise on final failure
        
        status = "PARTIAL" if partial else "COMPLETE"
        logger.info(f"💾 Successfully saved {len(call_data)} calls to persistent cache ({status}): {CACHE_FILE}")
        
        # Also save to S3 for persistence across deployments
        s3_client, s3_bucket = get_s3_client_and_bucket()
        if s3_client and s3_bucket:
            try:
                cache_json = json.dumps(cache_data, default=str, ensure_ascii=False)
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=S3_CACHE_KEY,
                    Body=cache_json.encode('utf-8'),
                    ContentType='application/json'
                )
                logger.info(f"☁️ Successfully saved {len(call_data)} calls to S3 cache ({status}): s3://{s3_bucket}/{S3_CACHE_KEY}")
            except Exception as s3_error:
                # Don't fail the entire save if S3 upload fails - local cache is still saved
                logger.warning(f"⚠️ Failed to save cache to S3 (local cache saved successfully): {s3_error}")
    except LockTimeoutError:
        # Re-raise LockTimeoutError - callers need to know about this
        raise
    except Exception as e:
        # CRITICAL FIX: Re-raise all exceptions after logging so callers can handle failures
        # This prevents silent failures that cause data loss
        logger.error(f"❌ Failed to save persistent cache: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise  # Re-raise so callers know save failed

# Cached wrapper - uses both Streamlit cache (fast) and disk cache (persistent, survives restarts)
# First load will take time, subsequent loads will be instant
# Use "Refresh New Data" button when new PDFs are added to S3 - it only loads new files
# Note: Using max_entries=1 to prevent cache from growing, and no TTL so it never auto-expires
@st.cache_data(ttl=None, max_entries=1, show_spinner=True)
def load_all_calls_cached(cache_version=0):
    """Cached wrapper - loads ALL data once, then serves from cache indefinitely until manually refreshed.
    
    cache_version parameter forces cache refresh when incremented (used after refresh completes).
    
    Strategy:
        1. Always check S3 cache first (source of truth, shared across all users)
        2. Invalidate Streamlit cache if S3 cache is newer
        3. Use Streamlit cache only if it matches S3 cache timestamp
        4. Fall back to disk cache only if S3 unavailable (backup only)
    
    For incremental updates, use the "Refresh New Data" button which calls load_new_calls_only().
    """
    import time
    from datetime import datetime
    start_time = time.time()
    
    # Check if user explicitly requested full reload
    reload_all_triggered = st.session_state.get('reload_all_triggered', False)
    
    # CRITICAL: Always check S3 cache first (source of truth, shared across all users)
    # This ensures all users get the same up-to-date data
    # Use session state caching to prevent duplicate S3 loads
    s3_cache_result = None
    s3_cache_timestamp = None
    
    # Check session state first to avoid duplicate S3 loads
    s3_cache_key = '_s3_cache_result'
    s3_timestamp_key = '_s3_cache_timestamp'
    if s3_cache_key in st.session_state and s3_timestamp_key in st.session_state:
        cached_timestamp = st.session_state[s3_timestamp_key]
        # Use cached result if timestamp matches (cache is still valid)
        # CRITICAL: Validate cached result before accessing to prevent crashes from corrupted session state
        cached_result = st.session_state[s3_cache_key]
        if cached_result is not None and isinstance(cached_result, tuple) and len(cached_result) >= 1 and cached_result[0] is not None:
            s3_cache_result = cached_result
            s3_cache_timestamp = cached_timestamp
            logger.debug(f"📦 Using session-cached S3 result: {len(s3_cache_result[0])} calls (timestamp: {s3_cache_timestamp})")
        else:
            # Session state contains invalid data - clear it and reload from S3
            logger.warning(f"⚠️ Session state contains invalid S3 cache data, clearing and reloading from S3")
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
                response = s3_client.get_object(Bucket=s3_bucket, Key=S3_CACHE_KEY)
                s3_cached_data = json.loads(response['Body'].read().decode('utf-8'))
                if isinstance(s3_cached_data, dict):
                    s3_cache_result = (s3_cached_data.get('call_data', []), s3_cached_data.get('errors', []))
                    s3_cache_timestamp = s3_cached_data.get('timestamp', None)
                    logger.info(f"☁️ Loaded from S3 cache (source of truth): {len(s3_cache_result[0])} calls (timestamp: {s3_cache_timestamp})")
                    
                    # Cache in session state to avoid duplicate loads
                    st.session_state[s3_cache_key] = s3_cache_result
                    if s3_cache_timestamp:
                        st.session_state[s3_timestamp_key] = s3_cache_timestamp
            except ClientError as s3_error:
                error_code = s3_error.response.get('Error', {}).get('Code', '')
                if error_code != 'NoSuchKey':
                    logger.warning(f"⚠️ Failed to load from S3 cache: {s3_error}")
            except Exception as s3_error:
                logger.warning(f"⚠️ Failed to load from S3 cache: {s3_error}")
    
    # CRITICAL: If S3 cache exists, check if Streamlit cache is stale
    # Invalidate Streamlit cache if S3 cache is newer
    if s3_cache_timestamp:
        streamlit_cache_timestamp = st.session_state.get('_s3_cache_timestamp', None)
        if streamlit_cache_timestamp and streamlit_cache_timestamp != s3_cache_timestamp:
            # S3 cache has been updated - clear Streamlit cache AND session state cache
            logger.info(f"🔄 S3 cache updated ({s3_cache_timestamp} != {streamlit_cache_timestamp}) - invalidating caches")
            try:
                load_all_calls_cached.clear()
                # Clear session state cache so we reload from S3
                if s3_cache_key in st.session_state:
                    del st.session_state[s3_cache_key]
                logger.info("✅ Cleared stale Streamlit cache and session state cache - will reload from S3")
            except Exception:
                pass  # Ignore clear errors
        # Store S3 cache timestamp for future comparison
        st.session_state['_s3_cache_timestamp'] = s3_cache_timestamp
    
    # CRITICAL: If refresh is in progress, use S3 cache directly (most up-to-date)
    refresh_in_progress = st.session_state.get('refresh_in_progress', False)
    if refresh_in_progress:
        if s3_cache_result and s3_cache_result[0]:
            logger.info(f"🔄 Refresh in progress - using S3 cache directly: {len(s3_cache_result[0])} calls")
            return s3_cache_result
        else:
            logger.info("⚠️ Refresh in progress but no S3 cache found - continuing with normal load")
    
    # Check if there's merged cache data from refresh operation
    if '_merged_cache_data' in st.session_state:
        merged_data = st.session_state['_merged_cache_data']
        merged_errors = st.session_state.get('_merged_cache_errors', [])
        logger.info(f"✅ Using merged cache data from refresh: {len(merged_data)} calls")
        # Store S3 timestamp if available
        if s3_cache_timestamp:
            st.session_state['_s3_cache_timestamp'] = s3_cache_timestamp
        return merged_data, merged_errors
    
    # CRITICAL: Use S3 cache if available (source of truth)
    if s3_cache_result and s3_cache_result[0]:
        logger.info(f"✅ Using S3 cache (source of truth): {len(s3_cache_result[0])} calls")
        # Store timestamp for future comparison
        if s3_cache_timestamp:
            st.session_state['_s3_cache_timestamp'] = s3_cache_timestamp
        return s3_cache_result
    
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
    disk_cache_timestamp = None
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
        cache_count = len(disk_call_data)
        
        # Get cache metadata
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    disk_cache_timestamp = cached_data.get('timestamp', None)
                    is_partial = cached_data.get('partial', False)
                    partial_processed = cached_data.get('processed', 0)
                    partial_total = cached_data.get('total', 0)
            except Exception as e:
                logger.warning(f"⚠️ Failed to read cache metadata: {e}")
        
        # Log cache comparison immediately after loading disk cache
        logger.info(f"📊 Cache Comparison: Disk cache = {cache_count} files (partial={is_partial}, {partial_processed}/{partial_total if partial_total > 0 else '?'} processed)")
        
        # CRITICAL: Only return early if cache is COMPLETE (not partial)
        # If partial, continue loading to complete it with incremental saves
        if not reload_all_triggered:
            if cache_count >= 100:  # Use cache if we have 100+ calls
                if is_partial:
                    progress_pct = (partial_processed * 100 // partial_total) if partial_total > 0 else 0
                    logger.info(f"📦 Found PARTIAL cache: {cache_count} calls ({progress_pct}% complete)")
                    logger.info(f"🔄 Will continue loading remaining files from S3 with incremental saves")
                    # Don't return early - continue to load remaining files
                else:
                    logger.info(f"✅ USING COMPLETE DISK CACHE: {cache_count} calls - prevents restart loss")
                    
                    # CRITICAL FIX: Clean up _merged_cache_data if Streamlit cache has correct count
                    # This prevents keeping stale _merged_cache_data when Streamlit cache is already updated
                    if '_merged_cache_data' in st.session_state:
                        expected_count = len(st.session_state['_merged_cache_data'])
                        if cache_count >= expected_count:
                            # Streamlit cache has correct data (or more), safe to delete _merged_cache_data
                            logger.info(f"✅ Streamlit cache confirmed updated ({cache_count} >= {expected_count} calls), cleaning up _merged_cache_data")
                            del st.session_state['_merged_cache_data']
                            if '_merged_cache_errors' in st.session_state:
                                del st.session_state['_merged_cache_errors']
                    
                    # After loading from cache, check for new files in background
                    # Set flag to trigger background refresh check
                    if 'auto_refresh_checked' not in st.session_state:
                        st.session_state.auto_refresh_checked = False
                    
                    if not st.session_state.auto_refresh_checked:
                        # Check for new files (lightweight check)
                        logger.info("🔍 Checking for new files in background...")
                        try:
                            new_count, check_error = check_for_new_pdfs_lightweight()
                            if check_error:
                                logger.warning(f"⚠️ Background check error: {check_error}")
                            elif new_count > 0:
                                logger.info(f"🆕 Found {new_count} new PDF(s) - will load in background")
                                # Set flag to trigger background load
                                st.session_state.auto_refresh_pending = new_count
                                st.session_state.new_pdfs_notification_count = new_count
                            else:
                                logger.info("✅ No new files found - cache is up to date")
                                # Clear notification count since there are no new files
                                st.session_state.new_pdfs_notification_count = 0
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to check for new files: {e}")
                        
                        st.session_state.auto_refresh_checked = True
                    
                    # Return disk cache (will update Streamlit cache with this value)
                    logger.info(f"✅ USING COMPLETE DISK CACHE: {cache_count} calls - prevents restart loss")
                    return disk_call_data, disk_errors if disk_errors else []
            else:
                logger.info(f"⚠️ Cache has only {cache_count} calls (< 100), will load from S3")
        else:
            logger.info(f"🔄 Reload ALL Data triggered - ignoring cache with {cache_count} calls, will load fresh from S3")
    
    # Only reach here if:
    # 1. No cache found, OR
    # 2. Cache has < 100 calls, OR  
    # 3. User explicitly requested reload (reload_all_triggered = True)
    # 4. Cache is partial (will continue loading remaining files)
    
    # Determine what to load
    if reload_all_triggered:
        # User explicitly requested full dataset - load ALL files (may take 10-20 min)
        logger.info(f"🔍 Reload ALL Data triggered - loading ALL files from S3 (this will take 10-20 minutes)")
        max_files = None  # Load all files
        st.session_state['reload_all_triggered'] = False  # Clear flag after use
    elif is_partial and partial_total > 0 and partial_processed < partial_total:
        # Partial cache exists - but limit auto-continuation to prevent crashes
        remaining_files = partial_total - partial_processed
        
        # Only auto-continue if remaining files is small (<= 1000) to prevent long blocking operations
        # This prevents crashes from trying to load thousands of files synchronously during page load
        if remaining_files <= 1000:
            logger.info(f"🔄 Continuing partial cache: {partial_processed}/{partial_total} files loaded, {remaining_files} remaining")
            logger.info(f"📥 Loading remaining {remaining_files} files from S3 with incremental saves (skipping already-processed files)")
            
            try:
                load_start = time.time()
                new_calls, new_errors, actual_new_count = load_new_calls_only()
                load_duration = time.time() - load_start
                elapsed = time.time() - start_time
                
                if isinstance(new_errors, str):
                    logger.error(f"❌ Error loading new files: {new_errors}")
                    # Return existing cache if new load fails
                    return disk_call_data, [new_errors] if disk_errors else []
                
                # Merge new calls with existing disk cache
                all_calls_merged = disk_call_data + new_calls
                all_calls_merged = deduplicate_calls(all_calls_merged)
                
                # Save merged data to disk immediately
                save_cached_data_to_disk(all_calls_merged, new_errors if new_errors else [])
                
                logger.info(f"✅ Loaded {actual_new_count} new files. Total: {len(all_calls_merged)} calls (merged with {len(disk_call_data)} from cache)")
                
                # Return merged data
                return all_calls_merged, new_errors if new_errors else []
            except Exception as e:
                logger.error(f"❌ Error in load_new_calls_only: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fallback to disk cache if new load fails
                return disk_call_data, disk_errors if disk_errors else []
        else:
            # Too many files remaining - return partial cache and let user manually refresh
            # This prevents crashes from long-running operations during page load
            logger.info(f"📦 Found PARTIAL cache: {partial_processed}/{partial_total} files ({remaining_files} remaining)")
            logger.info(f"✅ Returning partial cache immediately - use 'Refresh New Data' button to load remaining {remaining_files} files")
            logger.info(f"💡 Auto-continuation skipped to prevent crashes (>{1000} files remaining)")
            return disk_call_data, disk_errors if disk_errors else []
    else:
        # No substantial cache - load ALL files from S3
        logger.info(f"🔍 No substantial cache found - loading ALL files from S3")
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
        
        # Log cache counts for debugging (this happens after S3 load, so Streamlit cache may have data now)
        streamlit_cache_count = len(streamlit_call_data) if streamlit_call_data else 0
        disk_cache_count = len(disk_call_data) if disk_call_data else 0
        logger.info(f"📊 Cache Comparison (after load): Streamlit cache = {streamlit_cache_count} files, Disk cache = {disk_cache_count} files")
        
        # Determine which cache is better (more recent or more complete)
        use_streamlit_cache = False
        use_disk_cache = False
        
        if streamlit_call_data and len(streamlit_call_data) > 0:
            # Streamlit cache has data - check if it's better than disk cache
            if load_duration < 2.0:
                # Fast load = likely from Streamlit's in-memory cache
                logger.info(f"⚡ Detected Streamlit in-memory cache ({len(streamlit_call_data)} calls loaded in {load_duration:.2f}s)")
                
                if disk_call_data and len(disk_call_data) > 0:
                    # CRITICAL: Always prefer disk cache if it has more data, regardless of load speed
                    # This ensures we never show stale Streamlit cache when disk cache is more complete
                    if len(disk_call_data) > len(streamlit_call_data):
                        logger.info(f"✅ Disk cache is more complete ({len(disk_call_data)} vs {len(streamlit_call_data)} calls) - ALWAYS using disk cache")
                        use_disk_cache = True
                    elif len(streamlit_call_data) > len(disk_call_data):
                        logger.info(f"✅ Streamlit cache is more complete ({len(streamlit_call_data)} vs {len(disk_call_data)} calls) - using it")
                        use_streamlit_cache = True
                    elif len(streamlit_call_data) == len(disk_call_data):
                        # Same size - prefer disk cache as source of truth (it persists across restarts)
                        logger.info(f"✅ Caches match - using disk cache as source of truth (persists across restarts)")
                        use_disk_cache = True
                else:
                    # No disk cache - use Streamlit cache
                    use_streamlit_cache = True
            else:
                # Slow load = loaded from S3, not from cache
                # This is new data, use it and ALWAYS save to disk (critical for persistence)
                logger.info(f"📥 Loaded {len(streamlit_call_data)} calls from S3 (took {load_duration:.1f}s)")
                logger.info(f"💾 CRITICAL: Saving to disk cache to prevent loss on restart")
                use_streamlit_cache = True
                # Force save to disk immediately for slow loads (they're fresh data from S3)
                if streamlit_call_data:
                    # CRITICAL FIX: Wrap save in try/except so save failures don't cause function to return empty data
                    try:
                        save_cached_data_to_disk(streamlit_call_data, streamlit_errors)
                    except Exception as save_error:
                        logger.error(f"❌ Failed to save S3-loaded data to disk: {save_error}")
                        logger.warning(f"⚠️ Data was successfully loaded but not saved - will be lost on restart")
                        # Continue anyway - return the successfully loaded data
                elif disk_call_data and len(disk_call_data) > 0:
                    # Only disk cache has data
                    logger.info(f"✅ Using disk cache ({len(disk_call_data)} calls)")
                    use_disk_cache = True
        
        # Use the best cache
        if use_streamlit_cache:
            final_call_data, final_errors = streamlit_call_data, streamlit_errors
            # Deduplicate Streamlit cache data
            if final_call_data:
                original_count = len(final_call_data)
                final_call_data = deduplicate_calls(final_call_data)
                if len(final_call_data) < original_count:
                    logger.info(f"🔍 Deduplicated Streamlit cache: {original_count} → {len(final_call_data)} unique calls")
            # ALWAYS update disk cache with Streamlit cache data (it's more recent)
            # This ensures Streamlit cache is preserved to disk for future restarts
            if final_call_data:
                # Only skip save if we already saved it above (for slow S3 loads)
                if load_duration >= 2.0:
                    # Already saved above for slow loads, just log
                    logger.info(f"💾 Streamlit cache data already saved to disk (from S3 load)")
                else:
                    # Fast load = from Streamlit cache, save it to disk now
                    # CRITICAL FIX: Wrap save in try/except so save failures don't cause function to return empty data
                    try:
                        save_cached_data_to_disk(final_call_data, final_errors)
                        logger.info(f"💾 Saved Streamlit cache to disk cache ({len(final_call_data)} calls) - preserved for future restarts")
                    except Exception as save_error:
                        logger.error(f"❌ Failed to save Streamlit cache to disk: {save_error}")
                        logger.warning(f"⚠️ Data is available but not saved - will be lost on restart")
                        # Continue anyway - return the successfully loaded data
        elif use_disk_cache:
            final_call_data, final_errors = disk_call_data, disk_errors
            # Deduplicate disk cache data (should already be deduplicated, but double-check)
            if final_call_data:
                original_count = len(final_call_data)
                final_call_data = deduplicate_calls(final_call_data)
                if len(final_call_data) < original_count:
                    logger.info(f"🔍 Deduplicated disk cache: {original_count} → {len(final_call_data)} unique calls")
            logger.info(f"💾 Using disk cache - populating Streamlit's in-memory cache for faster access")
        else:
            # No cache available - use what we loaded
            final_call_data, final_errors = streamlit_call_data, streamlit_errors
            # Deduplicate before saving
            if final_call_data:
                original_count = len(final_call_data)
                final_call_data = deduplicate_calls(final_call_data)
                if len(final_call_data) < original_count:
                    logger.info(f"🔍 Deduplicated fresh load: {original_count} → {len(final_call_data)} unique calls")
                # CRITICAL FIX: Wrap save in try/except so save failures don't cause function to return empty data
                try:
                    save_cached_data_to_disk(final_call_data, final_errors)
                except Exception as save_error:
                    logger.error(f"❌ Failed to save fresh load to disk: {save_error}")
                    logger.warning(f"⚠️ Data was successfully loaded but not saved - will be lost on restart")
                    # Continue anyway - return the successfully loaded data
        
        logger.info(f"⏱️ Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        if reload_all_triggered:
            logger.info(f"✅ Returning {len(final_call_data) if final_call_data else 0} calls from FULL dataset")
        else:
            logger.info(f"✅ Returning {len(final_call_data) if final_call_data else 0} calls")
        
        # Return the data - Streamlit's @st.cache_data automatically caches this return value
        # This ensures both caches are in sync with the most recent data
        return final_call_data, final_errors
    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"❌ Error in load_all_calls_cached after {elapsed:.1f} seconds: {e}")
        # Return empty data with error message
        return [], [f"Failed to load data: {str(e)}"]

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

def create_filter_cache_key(date_range, agent_filter, score_filter, label_filter, search_text=None):
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
        str(search_text) if search_text else ""
    ]
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()

# Enhanced chart caching with filter-based cache keys
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_chart_with_filters(chart_id: str, cache_key: str, chart_func, *args, **kwargs):
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
    return hashlib.md5(data_str.encode()).hexdigest()

def create_filter_cache_key(date_range, agent_filter, score_filter, label_filter, search_text=None):
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
        str(search_text) if search_text else ""
    ]
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()

# Enhanced chart caching with filter-based cache keys
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_chart_with_filters(chart_id: str, cache_key: str, chart_func, *args, **kwargs):
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
    
    # Last 30 days baseline
    last_30_start = current_end_date - timedelta(days=30)
    last_30_data = df[(df["Call Date"] >= last_30_start) & (df["Call Date"] < current_start_date)]
    if not last_30_data.empty:
        baselines['last_30_days'] = {
            'avg_score': last_30_data["QA Score"].mean() if "QA Score" in last_30_data.columns else None,
            'pass_rate': calculate_pass_rate(last_30_data),
            'total_calls': len(last_30_data),
            'period': (last_30_start, current_start_date)
        }
    
    # Last 90 days baseline
    last_90_start = current_end_date - timedelta(days=90)
    last_90_data = df[(df["Call Date"] >= last_90_start) & (df["Call Date"] < current_start_date)]
    if not last_90_data.empty:
        baselines['last_90_days'] = {
            'avg_score': last_90_data["QA Score"].mean() if "QA Score" in last_90_data.columns else None,
            'pass_rate': calculate_pass_rate(last_90_data),
            'total_calls': len(last_90_data),
            'period': (last_90_start, current_start_date)
        }
    
    # Year-over-year (if data available)
    if current_start_date.year > df["Call Date"].min().year:
        yoy_start = current_start_date - timedelta(days=365)
        yoy_end = current_end_date - timedelta(days=365)
        yoy_data = df[(df["Call Date"] >= yoy_start) & (df["Call Date"] <= yoy_end)]
        if not yoy_data.empty:
            baselines['year_over_year'] = {
                'avg_score': yoy_data["QA Score"].mean() if "QA Score" in yoy_data.columns else None,
                'pass_rate': calculate_pass_rate(yoy_data),
                'total_calls': len(yoy_data),
                'period': (yoy_start, yoy_end)
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
    agent_perf['percentile'] = agent_perf[metric_col].rank(pct=True) * 100
    agent_perf = agent_perf.sort_values('percentile', ascending=False)
    
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
    daily_scores.columns = ['ds', 'y']
    daily_scores = daily_scores.sort_values('ds')
    
    if len(daily_scores) < 7:  # Need at least 7 days of data
        return predict_future_scores_simple(df, days_ahead)
    
    try:
        model = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True)
        model.fit(daily_scores)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        
        # Extract forecast for future dates only
        forecast_dates = forecast.tail(days_ahead)
        
        return {
            'dates': forecast_dates['ds'].dt.date.tolist(),
            'forecast': forecast_dates['yhat'].tolist(),
            'lower_bound': forecast_dates['yhat_lower'].tolist(),
            'upper_bound': forecast_dates['yhat_upper'].tolist(),
            'method': 'prophet'
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
    daily_scores = daily_scores.sort_values('Call Date')
    
    if len(daily_scores) < 2:
        return None
    
    # Simple linear regression
    from datetime import timedelta
    import numpy as np
    
    dates = daily_scores["Call Date"]
    scores = daily_scores["QA Score"]
    
    # Convert dates to numeric for regression
    date_nums = [(d - dates.min()).days for d in dates]
    
    # Linear regression
    coeffs = np.polyfit(date_nums, scores, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Predict future dates
    last_date = dates.max()
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    forecast_nums = [(d - dates.min()).days for d in forecast_dates]
    forecast_scores = [slope * n + intercept for n in forecast_nums]
    
    # Calculate confidence interval (simple std-based)
    residuals = scores - (slope * np.array(date_nums) + intercept)
    std_error = np.std(residuals)
    
    return {
        'dates': [d.date() for d in forecast_dates],
        'forecast': forecast_scores,
        'lower_bound': [f - 1.96 * std_error for f in forecast_scores],
        'upper_bound': [f + 1.96 * std_error for f in forecast_scores],
        'method': 'linear'
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
    
    if "Call Date" not in df.columns or "Agent" not in df.columns or "QA Score" not in df.columns:
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
        trend_slope = calculate_trend_slope(agent_data["Call Date"], agent_data["QA Score"])
        volatility = agent_data["QA Score"].std()
        proximity_to_threshold = threshold - recent_avg
        
        # Calculate risk score (0-100)
        risk_score = 0
        
        # Trend component (0-40 points)
        if trend_slope < -1:  # Declining trend
            risk_score += 40
        elif trend_slope < -0.5:
            risk_score += 20
        
        # Volatility component (0-30 points)
        if volatility > 15:
            risk_score += 30
        elif volatility > 10:
            risk_score += 15
        
        # Proximity component (0-30 points)
        if proximity_to_threshold <= 5:  # Very close to threshold
            risk_score += 30
        elif proximity_to_threshold <= 10:
            risk_score += 15
        
        if risk_score >= 50:  # High risk threshold
            at_risk.append({
                'agent': agent,
                'risk_score': risk_score,
                'recent_avg': recent_avg,
                'trend_slope': trend_slope,
                'volatility': volatility,
                'proximity_to_threshold': proximity_to_threshold,
                'recent_calls': len(agent_data)
            })
    
    # Sort by risk score
    at_risk.sort(key=lambda x: x['risk_score'], reverse=True)
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
    from datetime import timedelta
    
    if len(dates) < 2:
        return 0
    
    date_nums = [(d - dates.min()).days for d in dates]
    coeffs = np.polyfit(date_nums, scores, 1)
    return coeffs[0]

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
        return {'trajectory': 'insufficient_data', 'slope': 0, 'volatility': 0}
    
    df_sorted = df.sort_values("Call Date")
    dates = df_sorted["Call Date"]
    scores = df_sorted["QA Score"]
    
    slope = calculate_trend_slope(dates, scores)
    volatility = scores.std()
    
    # Classify trajectory
    if volatility > 15:
        trajectory = 'volatile'
    elif slope > 0.5:
        trajectory = 'improving'
    elif slope < -0.5:
        trajectory = 'declining'
    else:
        trajectory = 'stable'
    
    # Projected score if trend continues
    last_score = scores.iloc[-1]
    projected_score = last_score + (slope * 7)  # Project 7 days ahead
    
    return {
        'trajectory': trajectory,
        'slope': slope,
        'volatility': volatility,
        'current_score': last_score,
        'projected_score': projected_score
    }

def load_new_calls_only():
    """
    Smart refresh: Only loads PDFs that haven't been processed yet.
    Returns tuple: (new_call_data_list, error_message, count_of_new_files)
    """
    try:
        # OPTIMIZATION: Load cache ONCE at start and reuse throughout refresh
        # This prevents repeated cache reads (3-4 times per batch) that cause lock contention
        logger.debug("📂 Loading cache once at start of refresh (will reuse throughout)")
        disk_result = load_cached_data_from_disk()
        # CRITICAL FIX: Check if disk_result is None before accessing its elements
        existing_calls = disk_result[0] if (disk_result and disk_result[0] is not None) else []
        existing_errors = disk_result[1] if (disk_result and disk_result[1] is not None) else []
        
        # Extract existing_cache_keys once for duplicate checking
        existing_cache_keys = set()
        if existing_calls:
            existing_cache_keys = {call.get('_s3_key') or call.get('_id') for call in existing_calls if call.get('_s3_key') or call.get('_id')}
        
        # Get last_save_time from cache metadata (if available)
        last_save_time = 0
        if existing_calls and CACHE_FILE.exists():
            try:
                    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    last_save_time = cached_data.get('last_save_time', 0)
            except:
                pass
        
        # Also check session state as fallback for last_save_time
        if not last_save_time:
            last_save_time = getattr(st.session_state, '_last_incremental_save_time', 0)
        
        # Get already processed keys from disk cache (survives restarts)
        # Also check session state as a fallback
        processed_keys = set()
        
        if existing_calls:
            cached_calls = existing_calls
            logger.debug(f"📂 Loaded {len(cached_calls)} calls from disk cache - extracting S3 keys...")
            
            # Extract all S3 keys from cached calls
            keys_found = 0
            keys_missing = 0
            keys_from_s3_key = 0
            keys_from_id = 0
            keys_from_filename = 0
            sample_keys = []
            for call in cached_calls:
                # Try multiple ways to get the S3 key
                s3_key = call.get('_s3_key')
                if s3_key:
                    keys_from_s3_key += 1
                else:
                    s3_key = call.get('_id')
                    if s3_key:
                        keys_from_id += 1
                
                # If still no key, try to get from filename
                if not s3_key:
                    filename = call.get('Filename') or call.get('Call ID')
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
                        # Ensure it ends with .pdf
                        if not s3_key.lower().endswith('.pdf'):
                            s3_key = f"{s3_key}.pdf"
                        keys_from_filename += 1
                
                # Normalize the key (remove leading/trailing slashes for comparison)
                if s3_key:
                    s3_key = s3_key.strip('/')
                    processed_keys.add(s3_key)
                    keys_found += 1
                    if len(sample_keys) < 5:
                        sample_keys.append(s3_key)
                else:
                    keys_missing += 1
            
            logger.debug(f"📊 Key Extraction Summary: {keys_found} found ({keys_from_s3_key} from _s3_key, {keys_from_id} from _id, {keys_from_filename} from filename), {keys_missing} missing")
            if sample_keys:
                logger.debug(f"📋 Sample cached keys (first 5): {sample_keys[:5]}")
            if keys_missing > 0:
                logger.warning(f"⚠️ {keys_missing} cached calls are missing _s3_key - they may be reprocessed")
            else:
                logger.info("📂 No disk cache found in count_new_pdfs - all files will be treated as new")
        
        # Also check session state (for files processed in current session)
        session_keys = st.session_state.get('processed_s3_keys', set())
        if session_keys:
            processed_keys.update(session_keys)
            logger.info(f"📋 Found {len(session_keys)} additional files in session state")
        
        logger.info(f"🔍 Total {len(processed_keys)} files already processed - will skip these")
        
        # Configure S3 client
        import botocore.config
        config = botocore.config.Config(
            connect_timeout=10,
            read_timeout=30,
            retries={'max_attempts': 2}
        )
        s3_client_with_timeout = boto3.client(
            's3',
            aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
            region_name=st.secrets["s3"].get("region_name", "us-east-1"),
            config=config
        )
        
        # List all PDF files in S3
        paginator = s3_client_with_timeout.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=s3_bucket_name,
            Prefix=s3_prefix,
            MaxKeys=1000
        )
        
        # Find new PDFs (not in processed_keys)
        new_pdf_keys = []
        processed_keys_normalized = {key.strip('/') for key in processed_keys}
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
            
            if isinstance(page, dict) and 'Contents' in page:
                for obj in page['Contents']:
                    key_raw = obj.get('Key')
                    if not key_raw:
                        continue
                    key = key_raw.strip('/')  # Normalize S3 key (consistent with cache normalization)
                    if key.lower().endswith('.pdf'):
                        total_s3_files += 1
                        page_file_count += 1
                        all_s3_keys.append(key)
                        if len(sample_s3_keys) < 10:
                            sample_s3_keys.append(key)
                        if key not in processed_keys_normalized:
                            last_modified = obj.get('LastModified', datetime.min)
                            # Ensure LastModified is a datetime object for safe sorting
                            if not isinstance(last_modified, datetime):
                                last_modified = datetime.min
                            new_pdf_keys.append({
                                'key': key,  # Store normalized key
                                'last_modified': last_modified
                            })
        
            files_per_page.append(page_file_count)
            # Check if pagination is truncated (though paginator should handle this automatically)
            if 'IsTruncated' in page:
                is_truncated = page['IsTruncated']
                # Log IsTruncated status every 10 pages or if truncated
                if page_count % 10 == 0 or is_truncated:
                    logger.info(f"📄 Page {page_count}: IsTruncated={is_truncated}, files_in_page={page_file_count}")
            
            # Log pagination progress every 10 pages
            if page_count % 10 == 0:
                logger.info(f"📄 Processing page {page_count}, found {total_s3_files} PDF files so far...")
        
        # Log pagination completion with final IsTruncated status (combined)
        logger.info(f"📄 Pagination complete: {page_count} pages, {total_s3_files} files, IsTruncated={is_truncated} (False=complete, True=may be incomplete)")
        if files_per_page:
            logger.debug(f"📄 Files per page: min={min(files_per_page)}, max={max(files_per_page)}, last_page={files_per_page[-1]}")
        
        # Verify pagination completed (warn if suspicious)
        if total_s3_files > 0 and total_s3_files % 1000 == 0:
            logger.warning(f"⚠️ Total files ({total_s3_files}) is exactly divisible by 1000 - pagination might be incomplete!")
        if is_truncated:
            logger.error(f"❌ CRITICAL: Last page had IsTruncated=True! Pagination may be incomplete - expected more pages!")
        
        # Additional check: If cache has files and S3 listing found same count, but user expects more files,
        # this might indicate files are in a different location or being filtered
        # Check if last page was full (1000 files) - if so, there might be more pages even if IsTruncated=False
        if files_per_page and len(files_per_page) > 0:
            last_page_count = files_per_page[-1]
            if last_page_count == 1000 and not is_truncated:
                logger.warning(f"⚠️ WARNING: Last page had exactly 1000 files but IsTruncated=False!")
                logger.warning(f"⚠️ This is unusual - typically a full page (1000 files) means more pages exist.")
                logger.warning(f"⚠️ If you know there are more files in S3, they may be in a different prefix/folder.")
        
        # Exhaustive key comparison - compare ALL cache keys against ALL S3 keys
        # Count actual matches (exhaustive comparison)
        actual_matches = len([k for k in all_s3_keys if k in processed_keys_normalized])
        
        # Count S3 keys NOT in cache (should equal new files)
        s3_keys_not_in_cache = len([k for k in all_s3_keys if k not in processed_keys_normalized])
        
        # Count cache keys NOT in S3 (orphaned cache entries)
        cache_keys_not_in_s3 = len([k for k in processed_keys_normalized if k not in all_s3_keys])
        
        match_rate = (actual_matches / total_s3_files * 100) if total_s3_files > 0 else 0
        cache_hit_rate = (actual_matches / len(processed_keys_normalized) * 100) if len(processed_keys_normalized) > 0 else 0
        
        # Condensed key comparison results
        logger.info(f"📊 Key Comparison: S3={total_s3_files}, Cache={len(processed_keys_normalized)}, Matches={actual_matches}, New={s3_keys_not_in_cache}, Orphaned={cache_keys_not_in_s3}, MatchRate={match_rate:.1f}%, HitRate={cache_hit_rate:.1f}%")
        logger.info(f"   - New files to process: {len(new_pdf_keys)}")
        
        # Validate that new file count matches expected
        if s3_keys_not_in_cache != len(new_pdf_keys):
            logger.warning(f"⚠️ WARNING: Mismatch! S3 keys not in cache ({s3_keys_not_in_cache}) != new_pdf_keys count ({len(new_pdf_keys)})")
        
        if sample_s3_keys:
            logger.info(f"📋 Sample S3 keys (first 10): {sample_s3_keys[:10]}")
        
        # Diagnostic warnings for key matching issues
        if len(processed_keys_normalized) > 0 and cache_hit_rate < 90:
            logger.warning(f"⚠️ Cache hit rate is only {cache_hit_rate:.1f}% - some cached keys not found in S3")
            logger.warning(f"⚠️ This could indicate key normalization issues or orphaned cache entries")
            if cache_keys_not_in_s3 > 0:
                logger.warning(f"⚠️ Found {cache_keys_not_in_s3} orphaned cache keys (not in S3)")
        
        # Warn if all files are being treated as new when cache exists
        if len(new_pdf_keys) == total_s3_files and len(processed_keys_normalized) > 0 and total_s3_files > 0:
            logger.error(f"❌ ERROR: All {total_s3_files} files are being treated as new, but {len(processed_keys_normalized)} are cached!")
            logger.error(f"❌ Cache hit rate: {cache_hit_rate:.1f}% - Key extraction or normalization may have failed!")
            logger.error(f"❌ Sample cached keys: {list(processed_keys_normalized)[:5]}")
            logger.error(f"❌ Sample S3 keys: {all_s3_keys[:5]}")
        
        # Validate S3 listing completeness before early exit
        # IMPORTANT: If cache count matches S3 count exactly, but user knows there are more files in S3,
        # this indicates S3 listing is incomplete (files might be in different prefix/folder or filtered out)
        if len(processed_keys_normalized) > total_s3_files:
            logger.error(f"❌ CRITICAL: Cache has MORE keys ({len(processed_keys_normalized)}) than S3 ({total_s3_files})!")
            logger.error(f"❌ This suggests S3 listing is incomplete or cache has orphaned entries.")
            logger.error(f"❌ Difference: {len(processed_keys_normalized) - total_s3_files} keys")
            logger.error(f"❌ S3 prefix used: '{s3_prefix}' (empty means root of bucket)")
            logger.error(f"❌ If you know there are more files, they may be in a different prefix/folder.")
            # Don't exit early - S3 listing appears incomplete, continue to process
        elif len(processed_keys_normalized) == total_s3_files and total_s3_files > 0:
            # Cache count matches S3 count - verify all are actually matched
            # WARNING: This doesn't mean S3 listing is complete - files might be in different locations
            if actual_matches == total_s3_files and not new_pdf_keys:
                logger.info(f"✅ All {total_s3_files} S3 files (prefix '{s3_prefix}') are in cache and verified. Note: More files may exist in other prefixes.")
                return [], None, 0  # No new files - verified complete match for current prefix
            elif actual_matches == total_s3_files and len(new_pdf_keys) > 0:
                logger.warning(f"⚠️ WARNING: All keys match, but {len(new_pdf_keys)} files marked as new!")
                logger.warning(f"⚠️ This indicates a logic error - proceeding to process new files")
            else:
                logger.warning(f"⚠️ Cache count ({len(processed_keys_normalized)}) matches S3 count ({total_s3_files}), but only {actual_matches} keys match!")
                logger.warning(f"⚠️ This suggests key normalization issues - proceeding to process new files")
        elif not new_pdf_keys and len(processed_keys_normalized) < total_s3_files:
            # S3 has more files than cache, but no new files found - this shouldn't happen
            logger.warning(f"⚠️ WARNING: S3 has {total_s3_files} files, cache has {len(processed_keys_normalized)}, but no new files found!")
            logger.warning(f"⚠️ This suggests a comparison issue - proceeding anyway")
        
        # Final early exit check (only if validation passed above)
        if not new_pdf_keys and len(processed_keys_normalized) <= total_s3_files and actual_matches == total_s3_files:
            logger.info("✅ No new files found - all files are already processed and verified")
            return [], None, 0  # No new files
        
        # Additional check: if processed count matches total, all should be cached
        if len(processed_keys_normalized) >= total_s3_files and total_s3_files > 0:
            logger.warning(f"⚠️ WARNING: Cached keys ({len(processed_keys_normalized)}) >= Total S3 files ({total_s3_files})")
            logger.warning(f"⚠️ But {len(new_pdf_keys)} files are marked as new. This indicates a key format mismatch.")
            logger.warning(f"⚠️ Match rate: {match_rate:.1f}% - Keys are not matching correctly.")
        
        # Sort by modification date (most recent first)
        new_pdf_keys.sort(key=lambda x: x['last_modified'], reverse=True)
        
        # Process new PDFs
        new_calls = []
        errors = []
        
        def process_pdf(key_item):
            """Process a single PDF: download, parse, and return result.
            
            Args:
                key_item: Either a string (normalized key) or dict with 'key' and 'last_modified'
            """
            try:
                # Handle both string and dict formats
                if isinstance(key_item, dict):
                    normalized_key = key_item['key']
                else:
                    normalized_key = key_item
                
                # Use normalized key for S3 (S3 keys typically don't have leading slashes anyway)
                # But ensure it works - if download fails, try with prefix
                s3_key_to_use = normalized_key
                if s3_prefix and not normalized_key.startswith(s3_prefix):
                    # Try adding prefix if not present
                    s3_key_to_use = f"{s3_prefix.rstrip('/')}/{normalized_key}" if not normalized_key.startswith(s3_prefix) else normalized_key
                
                response = s3_client_with_timeout.get_object(Bucket=s3_bucket_name, Key=s3_key_to_use)
                pdf_bytes = response['Body'].read()
                filename = normalized_key.split('/')[-1]
                parsed_data = parse_pdf_from_bytes(pdf_bytes, filename)
                
                if parsed_data:
                    # Store normalized key for consistent comparison
                    parsed_data['_id'] = normalized_key
                    parsed_data['_s3_key'] = normalized_key
                    return parsed_data, None
                else:
                    return None, f"Failed to parse {filename}"
            except Exception as e:
                # If download failed with normalized key, log the error
                logger.warning(f"⚠️ Failed to download with key '{normalized_key}': {e}")
                return None, f"{normalized_key}: {str(e)}"
        
        # Process in smaller batches to reduce lock contention with large cache files
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        BATCH_SIZE = 50  # Reduced from 100 to reduce lock contention
        DASHBOARD_UPDATE_INTERVAL = 500
        MAX_FILES_PER_REFRESH = 1000  # Limit to 1000 files per refresh to avoid timeouts
        
        # Limit the number of files processed per refresh
        total_new_unlimited = len(new_pdf_keys)
        new_pdf_keys = new_pdf_keys[:MAX_FILES_PER_REFRESH]
        total_new = len(new_pdf_keys)
        
        if total_new_unlimited > MAX_FILES_PER_REFRESH:
            remaining = total_new_unlimited - MAX_FILES_PER_REFRESH
            logger.info(f"🔄 Refresh New Data: Found {total_new_unlimited} new PDF files total, processing {total_new} this refresh (limit: {MAX_FILES_PER_REFRESH}), {remaining} remaining")
        else:
            logger.info(f"🔄 Refresh New Data: Found {total_new} new PDF files to process")
        
        logger.info(f"📥 Starting to process {total_new} new PDF files in batches of {BATCH_SIZE} (out of {total_s3_files} total in S3)")
        
        processing_start_time = time.time()
        processed_count = 0
        last_dashboard_update = 0
        last_incremental_save_time = time.time()
        batches_since_save = 0
        SAVE_INTERVAL_BATCHES = 3  # Save every 3 batches instead of every batch
        SAVE_INTERVAL_SECONDS = 120  # Or every 2 minutes, whichever comes first
        
        # Note: last_save_time already loaded at start of function (reused here)
        
        # Process in batches
        for batch_start in range(0, total_new, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_new)
            batch_keys = new_pdf_keys[batch_start:batch_end]
            batch_num = batch_start//BATCH_SIZE + 1
            total_batches = (total_new + BATCH_SIZE - 1)//BATCH_SIZE
            
            if batch_num % 5 == 0 or batch_num == 1 or batch_num == total_batches:
                logger.info(f"📦 Processing batch {batch_num}/{total_batches}: files {batch_start+1}-{batch_end} of {total_new}")
            
            # Track calls from this batch only (for incremental save)
            batch_calls = []
            
            # OPTIMIZATION: Reuse existing_cache_keys loaded at start (no need to reload from disk)
            # existing_cache_keys already loaded at function start
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_key = {executor.submit(process_pdf, item): item['key'] for item in batch_keys}
                
                for future in as_completed(future_to_key):
                    try:
                        parsed_data, error = future.result(timeout=60)  # 60 second timeout per file
                    except Exception as e:
                        # Handle both TimeoutError and concurrent.futures.TimeoutError
                        from concurrent.futures import TimeoutError as FuturesTimeoutError
                        if isinstance(e, (TimeoutError, FuturesTimeoutError)):
                            key = future_to_key.get(future, 'Unknown')
                            logger.error(f"⏱️ Timeout processing {key}: {e}")
                            errors.append(f"{key}: Processing timeout (60s)")
                            st.session_state.pdf_processing_progress['errors'] += 1
                            processed_count += 1
                            continue
                        else:
                            # Unexpected error in future execution
                            logger.error(f"❌ Unexpected error in future: {e}")
                            errors.append(f"Unknown: {str(e)}")
                            st.session_state.pdf_processing_progress['errors'] += 1
                            processed_count += 1
                            continue
                    
                    processed_count += 1
                    
                    if parsed_data:
                        new_calls.append(parsed_data)
                        batch_calls.append(parsed_data)  # Track for this batch
                        
                        # Check if this call is already in cache
                        call_key = parsed_data.get('_s3_key') or parsed_data.get('_id')
                        if call_key and call_key in existing_cache_keys:
                            # Already in cache - skip (already added to new_calls above)
                            pass
                    elif error:
                        errors.append(error)
                    
                    # Log progress every 100 files (unconditionally for each processed file)
                    if processed_count % 100 == 0:
                        elapsed = time.time() - processing_start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        remaining = (total_new - processed_count) / rate if rate > 0 else 0
                        logger.info(f"📊 Refresh Progress: {processed_count}/{total_new} files processed ({processed_count*100//total_new if total_new > 0 else 0}%), {len(new_calls)} successful, {len(errors)} errors. Rate: {rate:.1f} files/sec, ETA: {remaining/60:.1f} min")
            
            # Update existing_calls with batch data (for next batch)
            # OPTIMIZATION: Use extend() instead of concatenation for better performance
            # This modifies the list in-place (O(k)) instead of creating a new list (O(n))
            existing_calls.extend(batch_calls)
            # Update existing_cache_keys to include new batch keys
            batch_keys_set = {call.get('_s3_key') or call.get('_id') for call in batch_calls if call.get('_s3_key') or call.get('_id')}
            existing_cache_keys.update(batch_keys_set)
            
            batches_since_save += 1
            time_since_save = time.time() - last_incremental_save_time
            should_save = (batches_since_save >= SAVE_INTERVAL_BATCHES) or (time_since_save >= SAVE_INTERVAL_SECONDS) or (batch_num == total_batches)
            
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
                                total=total_new
                            )
                            save_success = True
                            break  # Success, exit retry loop
                        except Exception as save_e:
                            save_error = save_e
                            if save_attempt < max_save_retries - 1:
                                wait_time = 0.5 * (save_attempt + 1)  # Exponential backoff: 0.5s, 1s, 1.5s
                                logger.warning(f"⚠️ Incremental save attempt {save_attempt + 1}/{max_save_retries} failed: {save_e}, retrying in {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"❌ CRITICAL: Failed to save incremental cache after {max_save_retries} attempts: {save_e}")
                                logger.error(f"❌ Data loss risk: {len(calls_to_save)} calls not saved to disk")
                    
                    if save_success:
                        last_incremental_save_time = time.time()
                        st.session_state._last_incremental_save_time = last_incremental_save_time
                        batches_since_save = 0  # Reset counter
                        logger.info(f"💾 Incremental save: Saved {len(calls_to_save)} calls to disk cache ({processed_count}/{total_new} = {processed_count*100//total_new if total_new > 0 else 0}% complete)")
                    else:
                        # All retries failed - log critical error but continue processing
                        # The next save attempt might succeed, and we don't want to lose all progress
                        logger.error(f"❌ CRITICAL: All incremental save retries failed. Last error: {save_error}")
                        logger.error(f"❌ Processing will continue, but data may be lost if app crashes")
                except Exception as e:
                    logger.error(f"❌ Unexpected error during incremental save: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Update dashboard every 500 calls
            if processed_count >= last_dashboard_update + DASHBOARD_UPDATE_INTERVAL:
                logger.info(f"🔄 Dashboard update: {processed_count} calls processed, updating progress...")
                last_dashboard_update = processed_count
                
                # Update session state with progress (no rerun to prevent cache corruption)
                # Progress will be visible on next natural rerun (user interaction or completion)
                st.session_state['last_refresh_progress_update'] = {
                    'processed': processed_count,
                    'total': total_new,
                    'timestamp': time.time()
                }
                
                # Note: Removed st.rerun() to prevent cache corruption from concurrent reads/writes
                # Progress updates will be visible when refresh completes or user interacts
        
        elapsed_total = time.time() - processing_start_time
        logger.info(f"✅ Refresh completed: Processed {total_new} new files in {elapsed_total/60:.1f} minutes. Success: {len(new_calls)}, Errors: {len(errors)}. Cache updated with {len(new_calls)} new calls.")
        
        return new_calls, errors if errors else None, len(new_calls)
        
    except Exception as e:
        return [], f"Error loading new calls: {e}", 0

credentials = st.secrets["credentials"].to_dict()
cookie = st.secrets["cookie"]
auto_hash = st.secrets.get("auto_hash", False)

authenticator = stauth.Authenticate(
    credentials,
    cookie["name"],
    cookie["key"],
    cookie["expiry_days"],
    auto_hash=auto_hash,
)

# --- LOGIN GUARD ---
auth_status = st.session_state.get("authentication_status")

# If they've never submitted the form, show it
if auth_status is None:
    try:
        authenticator.login("main", "Login")
    except Exception as e:
        # Handle CookieManager component loading issues gracefully
        error_msg = str(e).lower()
        if "cookiemanager" in error_msg or "component" in error_msg or "frontend" in error_msg or "extra_streamlit_components" in error_msg:
            st.error("⚠️ **Authentication Component Loading Issue**")
            st.warning("The authentication component is having trouble loading. This is usually a temporary network or CDN issue.")
            st.markdown("### 🔧 **Quick Fixes (try in order):**")
            st.markdown("1. **Wait 10-15 seconds** and refresh the page (F5 or Cmd+R)")
            st.markdown("2. **Hard refresh** the page:")
            st.code("Windows/Linux: Ctrl+Shift+R\nMac: Cmd+Shift+R", language=None)
            st.markdown("3. **Clear browser cache** and cookies for this site")
            st.markdown("4. **Try a different browser** or incognito/private mode")
            st.markdown("5. **Check your network connection** - ensure you can access external CDNs")
            st.markdown("6. **Verify installation** - ensure `extra-streamlit-components` is installed:")
            st.code("pip install --upgrade extra-streamlit-components", language="bash")
            st.markdown("---")
            st.info("💡 **If the issue persists:** This may be a network/proxy/CDN issue. Contact your administrator or check if your deployment environment allows access to Streamlit component CDNs.")
            logger.warning(f"CookieManager component loading issue: {e}")
            # Don't stop immediately - let user try to refresh
            st.markdown("---")
            if st.button("🔄 **Try Again After Refreshing**", type="primary"):
                st.rerun()
        else:
            st.error(f"❌ **Authentication Error**")
            st.error(f"Error: {str(e)}")
            logger.exception("Authentication error")
        st.stop()
    st.stop()

# If they submitted bad creds, show error and stay on login
if auth_status is False:
    st.error("❌ Username or password is incorrect")
    st.stop()

# Get current user info
current_username = st.session_state.get("username")
current_name = st.session_state.get("name")

# Check if user is anonymous (needed early for view mode display)
is_anonymous_user = current_username and current_username.lower() == "anonymous"

# Session management - track activity and timeout
if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time.time()
else:
    st.session_state.last_activity = time.time()  # Update on each interaction

# Check for session timeout (30 minutes of inactivity)
SESSION_TIMEOUT_MINUTES = 30
if check_session_timeout(st.session_state.last_activity, SESSION_TIMEOUT_MINUTES):
    st.warning("⏰ Your session has expired due to inactivity. Please log in again.")
    st.session_state.authentication_status = None
    st.session_state.last_activity = 0
    st.rerun()

# Show session timeout warning (5 minutes before timeout)
time_remaining = SESSION_TIMEOUT_MINUTES - ((time.time() - st.session_state.last_activity) / 60)
if 0 < time_remaining <= 5:
    st.sidebar.warning(f"⏰ Session expires in {int(time_remaining)} minute(s)")

# Audit logging (admin only - Shannon and Chloe)
if current_username and current_username.lower() in ["chloe", "shannon"]:
    log_audit_event(current_username, "page_access", f"Accessed dashboard at {datetime.now().isoformat()}")

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
    - Access background refresh settings
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
    - Background Refresh Settings
    - System Monitoring & Metrics
    - Data Quality Validation
    
    Currently only: Chloe, Shannon, and Jerson
    """
    if not current_username:
        return False
    allowed_users = ['chloe', 'shannon', 'jerson']
    return current_username.lower() in allowed_users

# Set user_agent_id and is_admin for backward compatibility
try:
    user_mapping = st.secrets.get("user_mapping", {})
    if current_username and current_username in user_mapping:
        agent_id_value = user_mapping[current_username].get("agent_id", "")
        if agent_id_value:
            user_agent_id = agent_id_value
except Exception:
    pass

# Set is_admin for backward compatibility (but prefer using is_regular_admin() function)
is_admin = is_regular_admin()

st.sidebar.success(f"Welcome, {current_name} 👋")

# Show view mode
if is_anonymous_user:
    st.sidebar.info("🔒 Anonymous View: De-identified Data")
elif user_agent_id:
    st.sidebar.info(f"👤 Agent View: {user_agent_id}")
elif is_regular_admin():
    st.sidebar.info("👑 Admin View: All Data")
else:
    st.sidebar.info("👤 User View: All Data")

# Logout button
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout", help="Log out of your account", type="secondary"):
    try:
        authenticator.logout("Logout", "sidebar")
        st.session_state.authentication_status = None
        st.session_state.username = None
        st.session_state.name = None
        st.rerun()
    except Exception as e:
        logger.error(f"Logout error: {e}")
        st.sidebar.error("Error logging out. Please refresh the page.")

# --- Background Refresh: Check for new PDFs periodically ---
@st.cache_data(ttl=60, max_entries=1, show_spinner=False)
def check_for_new_pdfs_lightweight():
    """
    Lightweight check: Just counts new PDFs without downloading.
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
            logger.info(f"📂 Checking disk cache for processed keys: {len(cached_calls)} calls")
            
            # Extract all S3 keys from cached calls
            keys_found = 0
            keys_missing = 0
            for call in cached_calls:
                # Try multiple ways to get the S3 key
                s3_key = call.get('_s3_key') or call.get('_id')
                
                # If still no key, try to get from filename
                if not s3_key:
                    filename = call.get('Filename') or call.get('Call ID')
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
                        # Ensure it ends with .pdf
                        if not s3_key.lower().endswith('.pdf'):
                            s3_key = f"{s3_key}.pdf"
                
                # Normalize the key (remove leading/trailing slashes for comparison)
                if s3_key:
                    s3_key = s3_key.strip('/')
                    processed_keys.add(s3_key)
                    keys_found += 1
                else:
                    keys_missing += 1
            
            logger.info(f"📂 Found {keys_found} processed S3 keys in disk cache ({keys_missing} calls missing keys)")
        else:
            logger.info("📂 No disk cache found in count_new_pdfs - all files will be treated as new")
        
        # Also check session state (for files processed in current session)
        session_keys = st.session_state.get('processed_s3_keys', set())
        if session_keys:
            processed_keys.update(session_keys)
            logger.info(f"📋 Found {len(session_keys)} additional files in session state")
        
        logger.info(f"🔍 Total {len(processed_keys)} files already processed")
        
        # Configure S3 client with short timeout for quick checks
        import botocore.config
        config = botocore.config.Config(
            connect_timeout=5,
            read_timeout=10,
            retries={'max_attempts': 1}
        )
        s3_client_quick = boto3.client(
            's3',
            aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
            region_name=st.secrets["s3"].get("region_name", "us-east-1"),
            config=config
        )
        
        # List all PDF files in S3 (quick check - just count)
        paginator = s3_client_quick.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=s3_bucket_name,
            Prefix=s3_prefix,
            MaxKeys=1000
        )
        
        # Count new PDFs (not in processed_keys)
        new_count = 0
        total_pdfs = 0
        processed_keys_normalized = {key.strip('/') for key in processed_keys}  # Consistent normalization
        
        # Track pagination progress
        page_count = 0
        is_truncated = False
        
        for page in pages:
            page_count += 1
            
            if isinstance(page, dict) and 'Contents' in page:
                for obj in page['Contents']:
                    key_raw = obj.get('Key')
                    if not key_raw:
                        continue
                    key = key_raw.strip('/')  # Normalize S3 key (consistent with cache normalization)
                    if key.lower().endswith('.pdf'):
                        total_pdfs += 1
                        if key not in processed_keys_normalized:
                            new_count += 1
            # Check if pagination is truncated
            if 'IsTruncated' in page:
                is_truncated = page['IsTruncated']
                # Log IsTruncated status every 10 pages or if truncated
                if page_count % 10 == 0 or is_truncated:
                    logger.info(f"📄 Page {page_count}: IsTruncated={is_truncated}")
        
        # Log pagination completion with final IsTruncated status (combined)
        logger.info(f"📄 Lightweight check: {page_count} pages, {total_pdfs} files, IsTruncated={is_truncated} (False=complete, True=may be incomplete)")
        
        # Verify pagination completed (warn if suspicious)
        if total_pdfs > 0 and total_pdfs % 1000 == 0:
            logger.warning(f"⚠️ Total files ({total_pdfs}) is exactly divisible by 1000 - pagination might be incomplete!")
        if is_truncated:
            logger.error(f"❌ CRITICAL: Last page had IsTruncated=True! Pagination may be incomplete - expected more pages!")
        
        # Validate S3 listing completeness
        if len(processed_keys_normalized) > total_pdfs:
            logger.error(f"❌ CRITICAL: Cache has MORE keys ({len(processed_keys_normalized)}) than S3 ({total_pdfs})!")
            logger.error(f"❌ This suggests S3 listing is incomplete or cache has orphaned entries.")
        
        logger.info(f"📊 PDF Count: {total_pdfs} total in S3, {len(processed_keys_normalized)} processed, {new_count} new")
        
        # Note: For lightweight check, we don't do full exhaustive comparison to save time
        # The full exhaustive comparison happens in load_new_calls_only()
        
        return new_count, None
        
    except Exception as e:
        return 0, f"Error checking for new PDFs: {e}"

# Initialize background refresh state
if 'bg_refresh_enabled' not in st.session_state:
    st.session_state.bg_refresh_enabled = True  # Enable by default
if 'last_bg_check_time' not in st.session_state:
    st.session_state.last_bg_check_time = 0
if 'bg_check_interval_minutes' not in st.session_state:
    st.session_state.bg_check_interval_minutes = 60  # Check every hour (very low AWS cost)
if 'new_pdfs_notification_count' not in st.session_state:
    st.session_state.new_pdfs_notification_count = 0
if 'bg_check_error' not in st.session_state:
    st.session_state.bg_check_error = None

# Background refresh: Check for new PDFs if enough time has passed
current_time = time.time()
time_since_last_check = (current_time - st.session_state.last_bg_check_time) / 60  # minutes

if st.session_state.bg_refresh_enabled and time_since_last_check >= st.session_state.bg_check_interval_minutes:
    # Perform lightweight check
    new_count, check_error = check_for_new_pdfs_lightweight()
    st.session_state.last_bg_check_time = current_time
    
    if check_error:
        st.session_state.bg_check_error = check_error
    else:
        st.session_state.bg_check_error = None
        if new_count > 0:
            st.session_state.new_pdfs_notification_count = new_count
        else:
            # Clear notification count since there are no new files
            st.session_state.new_pdfs_notification_count = 0
else:
    # Even if not time for background check, verify count is correct
    # If we have a notification count but haven't checked recently, verify it's still valid
    if st.session_state.new_pdfs_notification_count > 0:
        # Quick verification - if we have a notification but haven't checked recently, verify it
        new_count, _ = check_for_new_pdfs_lightweight()
        if new_count == 0:
            # Clear stale notification count
            st.session_state.new_pdfs_notification_count = 0

# Show notification if new PDFs are available
if st.session_state.new_pdfs_notification_count > 0:
    st.sidebar.markdown("---")
    st.sidebar.success(f"🆕 **{st.session_state.new_pdfs_notification_count} new PDF(s) available!**")
    st.sidebar.caption("Click 'Refresh New Data' below to load them")

# Show background check error if any
if st.session_state.bg_check_error:
    st.sidebar.markdown("---")
    st.sidebar.warning(f"⚠️ Background check: {st.session_state.bg_check_error}")

# Prominent refresh button for when new data is added (Chloe, Shannon, and Jerson only)
if is_super_admin():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Refresh Data")
    st.sidebar.info("💡 **When to refresh:** Click 'Refresh New Data' after new PDFs are added to S3")
    st.sidebar.caption("ℹ️ **Cache never expires** - Data stays cached until you manually refresh")

# Background refresh settings (Chloe, Shannon, and Jerson only)
if is_super_admin():
    with st.sidebar.expander("⚙️ Background Refresh Settings"):
        bg_enabled = st.checkbox("Enable background refresh", value=st.session_state.bg_refresh_enabled)
        if bg_enabled != st.session_state.bg_refresh_enabled:
            st.session_state.bg_refresh_enabled = bg_enabled
            st.rerun()
        
        if bg_enabled:
            interval = st.number_input(
                "Check interval (minutes)",
                min_value=1,
                max_value=60,
                value=st.session_state.bg_check_interval_minutes,
                step=1,
                help="How often to check for new PDFs in the background"
            )
            if interval != st.session_state.bg_check_interval_minutes:
                st.session_state.bg_check_interval_minutes = interval
                st.rerun()
            
            if st.button("🔄 Check Now"):
                new_count, check_error = check_for_new_pdfs_lightweight()
                st.session_state.last_bg_check_time = time.time()
                if check_error:
                    st.error(f"❌ {check_error}")
                elif new_count > 0:
                    st.success(f"✅ Found {new_count} new PDF(s)!")
                    st.session_state.new_pdfs_notification_count = new_count
                else:
                    st.info("ℹ️ No new PDFs found")
                st.rerun()

# Smart refresh button (Chloe, Shannon, and Jerson only) - only loads new PDFs
# Note: files_to_load will be defined later, but we'll use None here to get all cached data
if is_super_admin():
    if st.sidebar.button('🔄 Refresh New Data', help='Only processes new PDFs added since last refresh. Fast and efficient!', type='primary'):
        log_audit_event(current_username, 'refresh_data', 'Refreshed new data from S3')
        
        # Set flag to prevent main data loading during refresh (prevents conflicts and crashes)
        st.session_state['refresh_in_progress'] = True
        
        # IMPORTANT: Preserve Streamlit cache BEFORE refresh to avoid calling load_all_calls_cached() during refresh
        # This prevents crashes from triggering a full S3 reload during the refresh operation
        previous_streamlit_cache = None
        previous_streamlit_errors = []
        try:
            # Get Streamlit cache BEFORE refresh (safe to call here, won't trigger reload)
            # Use cache_version when preserving Streamlit cache
            cache_version = st.session_state.get('_cache_version', 0)
            streamlit_result = load_all_calls_cached(cache_version=cache_version)
            previous_streamlit_cache = streamlit_result[0] if streamlit_result[0] else []
            previous_streamlit_errors = streamlit_result[1] if streamlit_result[1] else []
            logger.info(f"💾 Preserved Streamlit cache: {len(previous_streamlit_cache)} calls before refresh")
        except Exception as e:
            logger.warning(f"⚠️ Could not preserve Streamlit cache: {e} - will use disk cache only")
        
        # CRITICAL FIX: Wrap load_new_calls_only() in try/except to ensure refresh_in_progress flag is always cleared
        refresh_failed = False
        try:
            with st.spinner('🔄 Checking for new PDFs...'):
                new_calls, new_errors, new_count = load_new_calls_only()
        except Exception as e:
            # If load_new_calls_only() crashes, clear flag and show error
            refresh_failed = True
            st.session_state['refresh_in_progress'] = False
            logger.exception(f"❌ CRITICAL: load_new_calls_only() crashed: {e}")
            st.error(f"❌ Refresh failed with unexpected error: {e}")
            st.info("💡 Please try again or use 'Reload ALL Data' button if the issue persists")
            # Return early to prevent further processing
            new_calls, new_errors, new_count = [], f"Unexpected error: {e}", 0
        
        # Check if there was an overall error (returns string instead of list)
        if isinstance(new_errors, str):
            # Overall error occurred (e.g., network timeout, S3 access issue)
            refresh_failed = True
            st.session_state['refresh_in_progress'] = False  # Clear flag on error
            st.error(f"❌ Error refreshing data: {new_errors}")
            st.info("💡 Try using 'Reload ALL Data' button if the issue persists")
            # Rerun to update UI and prevent further processing
            st.rerun()
        elif new_count > 0 and not refresh_failed:
            # Successfully found and processed new files
            # CRITICAL FIX: load_new_calls_only() already merged new calls with existing_calls incrementally
            # and saved to disk during processing. Don't merge again - just load from disk and verify.
            
            # Load disk cache once (should already contain merged data from incremental saves)
            disk_result = load_cached_data_from_disk()
            # CRITICAL FIX: Check if disk_result is None before accessing its elements
            disk_cached_calls = disk_result[0] if (disk_result and disk_result[0] is not None) else []
            disk_cached_errors = disk_result[1] if (disk_result and disk_result[1] is not None) else []
            
            # The disk cache should already contain all merged data from incremental saves
            # Verify that incremental saves worked correctly by checking if new_calls keys are in disk cache
            disk_cache_keys = {call.get('_s3_key') or call.get('_id') for call in disk_cached_calls if call.get('_s3_key') or call.get('_id')}
            new_calls_keys = {call.get('_s3_key') or call.get('_id') for call in new_calls if call.get('_s3_key') or call.get('_id')}
            
            # Check if incremental saves worked: all new_calls keys should be in disk cache
            missing_keys = new_calls_keys - disk_cache_keys
            if missing_keys:
                # Incremental saves didn't work properly - merge as fallback
                logger.warning(f"⚠️ Disk cache missing {len(missing_keys)} keys from new_calls - incremental saves may have failed, merging as fallback")
                logger.warning(f"⚠️ Sample missing keys: {list(missing_keys)[:5]}")
                all_calls_merged = disk_cached_calls + new_calls
                all_calls_merged = deduplicate_calls(all_calls_merged)
                logger.info(f"🔍 Fallback merge: {len(disk_cached_calls)} + {len(new_calls)} = {len(all_calls_merged)} unique calls")
            elif not disk_cached_calls and new_calls:
                # Disk cache is empty but we have new calls - merge as fallback
                logger.warning(f"⚠️ Disk cache is empty but {len(new_calls)} new calls were processed - merging as fallback")
                # CRITICAL FIX: Deduplicate before assignment to match what save_cached_data_to_disk() will do
                # This ensures verification check compares post-dedup counts correctly
                all_calls_merged = deduplicate_calls(new_calls)
                logger.info(f"🔍 Fallback merge: Using {len(all_calls_merged)} unique calls from {len(new_calls)} new calls (disk cache was empty)")
            else:
                # Use disk cache directly (already merged from incremental saves)
                all_calls_merged = disk_cached_calls
                if new_calls_keys and disk_cache_keys:
                    overlap_count = len(new_calls_keys & disk_cache_keys)
                    logger.info(f"✅ Using disk cache directly: {len(all_calls_merged)} calls (verified: {overlap_count}/{len(new_calls_keys)} new keys in cache)")
                else:
                    logger.info(f"✅ Using disk cache directly: {len(all_calls_merged)} calls (already merged from incremental saves)")
            
            # Mark cache as complete (non-partial) now that refresh is done
            try:
                save_cached_data_to_disk(
                    all_calls_merged,
                    new_errors if new_errors else disk_cached_errors,
                    partial=False
                )
                logger.info(f"💾 Marked cache as complete: {len(all_calls_merged)} calls")
            except Exception as save_error:
                logger.error(f"❌ CRITICAL: Failed to mark cache as complete: {save_error}")
                logger.error(f"❌ Cache may still be marked as partial - this could cause issues on next refresh")
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
                    logger.warning(f"⚠️ Verification load returned None - checking previous disk_result as fallback")
                    # CRITICAL FIX: Check if disk_result is None before using as fallback
                    if disk_result is not None and disk_result[0] is not None:
                        disk_result_verify = disk_result
                        disk_cache_count = len(disk_result[0])
                    else:
                        logger.error(f"❌ Both verification load and disk_result are None - using empty lists as fallback")
                        disk_result_verify = ([], [])
                        disk_cache_count = 0
                        verification_failed = True
            except Exception as verify_error:
                logger.error(f"❌ CRITICAL: Verification load failed: {verify_error}")
                logger.error(f"❌ Checking previous disk_result as fallback for verification")
                verification_failed = True
                # CRITICAL FIX: Check if disk_result is None before using as fallback
                if disk_result is not None and disk_result[0] is not None:
                    disk_result_verify = disk_result
                    disk_cache_count = len(disk_result[0])
                    verification_failed = False  # We have valid fallback data
                else:
                    logger.error(f"❌ Both verification load and disk_result are None - using empty lists as fallback")
                    disk_result_verify = ([], [])
                    disk_cache_count = 0
            
            if disk_cache_count >= len(all_calls_merged):
                # Disk cache was saved successfully
                # CRITICAL FIX: Check if disk_result_verify and its elements are valid before using
                # Also check if verification failed and we're using empty fallback lists
                if (disk_result_verify is None or disk_result_verify[0] is None or 
                    (verification_failed and disk_result_verify == ([], []))):
                    logger.error(f"❌ CRITICAL: disk_result_verify is None or has None data - cannot proceed with merge")
                    logger.error(f"❌ Using all_calls_merged as fallback")
                    # Use all_calls_merged as fallback since verification failed
                    st.session_state['_merged_cache_data'] = all_calls_merged
                    st.session_state['_merged_cache_errors'] = new_errors if new_errors else []
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
                            key = call.get('_s3_key') or call.get('_id') or call.get('Call ID')
                            if key:
                                previous_keys.add(key)
                        
                        disk_keys = set()
                        for call in disk_result_verify[0]:
                            key = call.get('_s3_key') or call.get('_id') or call.get('Call ID')
                            if key:
                                disk_keys.add(key)
                        
                        if previous_keys == disk_keys:
                            # Caches are identical - no merge needed, use all_calls_merged directly
                            logger.info(f"✅ Streamlit cache and disk cache have identical keys ({len(previous_keys)} calls) - skipping merge to avoid crash")
                            logger.info(f"✅ Using all_calls_merged ({len(all_calls_merged)} calls) directly - no data loss")
                            st.session_state['_merged_cache_data'] = all_calls_merged
                            verify_errors = disk_result_verify[1] if (disk_result_verify and len(disk_result_verify) > 1 and disk_result_verify[1] is not None) else []
                            st.session_state['_merged_cache_errors'] = new_errors if new_errors else verify_errors
                        else:
                            # Caches are different - merge to preserve all data
                            # Calculate difference for logging
                            only_in_streamlit = previous_keys - disk_keys
                            only_in_disk = disk_keys - previous_keys
                            
                            logger.info(f"🔄 Caches differ: {len(only_in_streamlit)} calls only in Streamlit, {len(only_in_disk)} calls only in disk")
                            logger.info(f"🔄 Merging preserved Streamlit cache ({len(previous_streamlit_cache)} calls) with disk cache ({disk_cache_count} calls)")
                            
                            # Merge preserved Streamlit cache with disk cache
                            merged_data = previous_streamlit_cache + disk_result_verify[0]
                            merged_data = deduplicate_calls(merged_data)
                            merged_count = len(merged_data)
                            
                            logger.info(f"✅ Merged result: {merged_count} unique calls (removed {len(previous_streamlit_cache) + len(disk_result_verify[0]) - merged_count} duplicates)")
                            
                            # Save merged result to disk
                            merge_save_succeeded = False
                            try:
                                # CRITICAL FIX: Check if disk_result_verify[1] exists before using
                                verify_errors = disk_result_verify[1] if (disk_result_verify and len(disk_result_verify) > 1 and disk_result_verify[1] is not None) else []
                                save_cached_data_to_disk(
                                    merged_data,
                                    previous_streamlit_errors if previous_streamlit_errors else verify_errors
                                )
                                logger.info(f"✅ Saved merged cache to disk: {merged_count} calls")
                                merge_save_succeeded = True
                            except Exception as merge_save_error:
                                logger.error(f"❌ CRITICAL: Failed to save merged cache: {merge_save_error}")
                                logger.error(f"❌ Using disk cache without Streamlit cache merge - some data may be lost")
                            
                            # CRITICAL FIX: Only store merged_data in session state if save succeeded
                            # If save failed, use disk cache (fallback) to prevent data loss
                            if merge_save_succeeded:
                                # Store merged data in session state so it's used after cache clear
                                st.session_state['_merged_cache_data'] = merged_data
                                verify_errors = disk_result_verify[1] if (disk_result_verify and len(disk_result_verify) > 1 and disk_result_verify[1] is not None) else []
                                st.session_state['_merged_cache_errors'] = previous_streamlit_errors if previous_streamlit_errors else verify_errors
                            else:
                                # Fallback: use disk cache without merge (save failed, merged_data not on disk)
                                # CRITICAL FIX: Skip using disk_result_verify entirely if verification failed
                                # Don't use potentially corrupted cache data when verification fails
                                if not verification_failed and disk_result_verify and disk_result_verify[0] is not None:
                                    st.session_state['_merged_cache_data'] = disk_result_verify[0]
                                    verify_errors = disk_result_verify[1] if (disk_result_verify and len(disk_result_verify) > 1 and disk_result_verify[1] is not None) else []
                                    st.session_state['_merged_cache_errors'] = verify_errors
                                else:
                                    logger.error(f"❌ CRITICAL: Cannot use disk cache (verification failed or invalid) - using all_calls_merged as fallback")
                                    st.session_state['_merged_cache_data'] = all_calls_merged
                                    st.session_state['_merged_cache_errors'] = new_errors if new_errors else []
                    else:
                        # disk_result_verify[0] is None - use previous_streamlit_cache only
                        logger.warning(f"⚠️ disk_result_verify[0] is None - using previous_streamlit_cache only")
                        st.session_state['_merged_cache_data'] = previous_streamlit_cache
                        st.session_state['_merged_cache_errors'] = previous_streamlit_errors if previous_streamlit_errors else []
                else:
                    # No previous Streamlit cache - just use disk cache
                    logger.info(f"ℹ️ No previous Streamlit cache to merge - using disk cache ({disk_cache_count} calls)")
                    # CRITICAL FIX: Skip using disk_result_verify entirely if verification failed
                    # Don't use potentially corrupted cache data when verification fails
                    if not verification_failed and disk_result_verify and disk_result_verify[0] is not None:
                        st.session_state['_merged_cache_data'] = disk_result_verify[0]
                        # CRITICAL FIX: Check if disk_result_verify is None before calling len()
                        verify_errors = disk_result_verify[1] if (disk_result_verify and len(disk_result_verify) > 1 and disk_result_verify[1] is not None) else []
                        st.session_state['_merged_cache_errors'] = verify_errors
                    else:
                        logger.error(f"❌ CRITICAL: disk_result_verify is invalid (None or failed verification) - using all_calls_merged as fallback")
                        st.session_state['_merged_cache_data'] = all_calls_merged
                        st.session_state['_merged_cache_errors'] = new_errors if new_errors else []
                
                # CRITICAL FIX: Don't clear Streamlit cache - let it update naturally
                # The merged data is stored in session state (_merged_cache_data)
                # When load_all_calls_cached() is called next, it will return that data
                # and Streamlit's @st.cache_data will automatically cache it
                # This ensures disk cache is backed up in Streamlit cache without risk of data loss
                # Clearing cache here could cause data loss if something goes wrong before next call
                logger.info(f"✅ Disk cache saved ({disk_cache_count} calls) - Streamlit cache will update with latest data on next access")
                logger.info(f"💾 Merged data stored in session state - will be cached automatically when load_all_calls_cached() is called")
                
                # REMOVED: load_all_calls_cached.clear() - causes crashes during refresh/rerun
                # Instead, rely on _merged_cache_data in session state which load_all_calls_cached() 
                # will use on next call (line 1118-1126), and Streamlit will cache that result automatically.
                # The disk cache is already persisted to logs/cached_calls_data.json, so data is safe.
            else:
                logger.warning(f"⚠️ Disk cache verification failed: expected {len(all_calls_merged)} calls, found {disk_cache_count} - NOT clearing Streamlit cache")
                # CRITICAL FIX: When verification fails, use verified disk cache data if available
                # Don't store unverified all_calls_merged that may not have been persisted to disk
                if not verification_failed and disk_result_verify and disk_result_verify[0] is not None:
                    logger.info(f"⚠️ Using verified disk cache data ({disk_cache_count} calls) instead of unverified merged data ({len(all_calls_merged)} calls)")
                    all_calls_merged = disk_result_verify[0]
                    # Use verified errors if available
                    if disk_result_verify and len(disk_result_verify) > 1 and disk_result_verify[1] is not None:
                        new_errors = disk_result_verify[1]
                else:
                    logger.error(f"❌ CRITICAL: Verification failed and no valid disk cache available - data may not be persisted")
                    # Still use all_calls_merged as last resort, but log the risk
                    logger.warning(f"⚠️ Using unverified all_calls_merged as fallback - data may be lost on restart")
            
            # We need to manually update the cache - store in session state temporarily
            st.session_state['merged_calls'] = all_calls_merged
            st.session_state['merged_errors'] = new_errors if new_errors else []
            # Update processed keys tracking
            if 'processed_s3_keys' not in st.session_state:
                st.session_state['processed_s3_keys'] = set()
            new_keys = {call.get('_s3_key') for call in new_calls if call.get('_s3_key')}
            st.session_state['processed_s3_keys'].update(new_keys)
            st.success(f'✅ Added {new_count} new call(s)! Total: {len(all_calls_merged)} calls')
            if new_errors:
                st.warning(f'⚠️ {len(new_errors)} file(s) had errors')
            # Clear notification count after successful refresh
            st.session_state.new_pdfs_notification_count = 0
            # Reset auto_refresh_checked so startup check runs again to verify 0 new files
            st.session_state.auto_refresh_checked = False
            
            # CRITICAL FIX: Increment cache version and clear Streamlit cache before rerun
            # This forces Streamlit cache to reload from S3 cache (source of truth) on next call
            try:
                # Increment cache version to force cache refresh
                current_version = st.session_state.get('_cache_version', 0)
                st.session_state['_cache_version'] = current_version + 1
                
                # Store S3 cache timestamp after saving (so we can detect when it's updated)
                s3_client, s3_bucket = get_s3_client_and_bucket()
                if s3_client and s3_bucket:
                    try:
                        response = s3_client.get_object(Bucket=s3_bucket, Key=S3_CACHE_KEY)
                        s3_data = json.loads(response['Body'].read().decode('utf-8'))
                        s3_timestamp = s3_data.get('timestamp', None)
                        if s3_timestamp:
                            st.session_state['_s3_cache_timestamp'] = s3_timestamp
                            logger.info(f"✅ Stored S3 cache timestamp: {s3_timestamp}")
                    except Exception as s3_read_error:
                        logger.debug(f"Could not read S3 cache timestamp: {s3_read_error}")
                
                # Clear Streamlit cache to force reload from S3 cache
                load_all_calls_cached.clear()
                logger.info(f"✅ Cleared Streamlit cache - will reload {len(all_calls_merged)} calls from S3 cache (source of truth)")
            except Exception as clear_error:
                logger.warning(f"⚠️ Could not clear Streamlit cache: {clear_error} - will rely on cache_version parameter")
                # Continue anyway - cache_version parameter will force refresh
            
            # Clear refresh flag before rerun
            st.session_state['refresh_in_progress'] = False
            # Rerun to show updated data (Streamlit cache will reload from _merged_cache_data)
            st.rerun()
        else:
            # No new files found and no errors
            st.session_state['refresh_in_progress'] = False  # Clear flag
            st.info("ℹ️ No new PDFs found. All data is up to date!")
# Admin-only: Full reload button (Chloe and Shannon only)
if current_username and current_username.lower() in ["chloe", "shannon"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👑 Admin: Full Reload")
    if st.sidebar.button("🔄 Reload ALL Data (Admin Only)", help="⚠️ Clears cache and reloads ALL PDFs from S3. This may take 10-20 minutes.", type="secondary"):
        if current_username and current_username.lower() in ["chloe", "shannon"]:
            log_audit_event(current_username, "reload_all_data", "Cleared cache and reloaded all data from S3")
        st.cache_data.clear()
        # Clear persistent disk cache
        if CACHE_FILE.exists():
            try:
                CACHE_FILE.unlink()
                logger.info(f"🗑️ Cleared persistent disk cache: {CACHE_FILE}")
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {e}")
        if 'processed_s3_keys' in st.session_state:
            del st.session_state['processed_s3_keys']
        # Mark that full dataset should be cached after this reload
        st.session_state['full_dataset_cached'] = False  # Will be set to True after load completes
        st.session_state['reload_all_triggered'] = True  # Flag to trigger full load
        st.success("🔄 Cache cleared! Reloading all data from S3...")
        st.rerun()

# --- Load Rubric Reference ---
@st.cache_data
def load_rubric():
    """Load the rubric JSON file."""
    try:
        import json
        import os
        rubric_path = os.path.join(os.path.dirname(__file__), "Rubric_v33.json")
        if os.path.exists(rubric_path):
            with open(rubric_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        return None

rubric_data = load_rubric()

# --- Rubric Reference Link in Sidebar ---
st.sidebar.markdown("---")
if rubric_data:
    st.sidebar.success(f"📚 Rubric loaded ({len(rubric_data)} items)")
    st.sidebar.info("💡 View full rubric reference in the main dashboard below")
else:
    st.sidebar.warning("⚠️ Rubric file not found")

# --- Fetch Call Metadata ---
# Only test S3 connection AFTER user is logged in (moved from before login)
status_text = st.empty()
status_text.text("📋 Preparing to load data...")
logger.info("Starting data load process (user is authenticated)")

try:
    # Quick connection test with aggressive timeouts (only after login)
    import botocore.config
    
    config = botocore.config.Config(
        connect_timeout=5,  # 5 seconds max
        read_timeout=10,    # 10 seconds max
        retries={'max_attempts': 1}  # No retries for faster failure
    )
    test_client = boto3.client(
        's3',
        aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
        region_name=st.secrets["s3"].get("region_name", "us-east-1"),
        config=config
    )
    
    status_text.text("📋 Testing S3 connection...")
    logger.debug(f"Testing connection to bucket: {s3_bucket_name}")
    
    # Quick test - just check if we can access the bucket with timeout
    try:
        test_client.head_bucket(Bucket=s3_bucket_name)
        logger.debug("S3 connection test successful")
        status_text.text("✅ Connected! Loading data...")
    except Exception as bucket_error:
        logger.error(f"S3 bucket access failed: {bucket_error}")
        status_text.empty()
        st.warning("⚠️ **S3 Connection Issue**")
        st.warning("Could not connect to S3. The app will try to use cached data if available.")
        st.info("💡 **If you see cached data below, you can continue using the app.**")
        st.info("💡 **If not, check your S3 credentials or network connection.**")
        # Don't stop - try to load from cache instead
        status_text.text("📋 Attempting to load from cache...")
    
    # Skip PDF count for faster startup - just load data directly
    pdf_count = None
    logger.debug("Skipping PDF count for faster startup - proceeding to data loading...")
    
except ClientError as e:
    status_text.empty()
    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
    if error_code == '404':
        st.error(f"❌ S3 bucket '{s3_bucket_name}' not found.")
        st.error("Please check the bucket name in your secrets.toml")
    elif error_code == '403':
        st.error(f"❌ Access denied to S3 bucket '{s3_bucket_name}'.")
        st.error("Please check your AWS credentials and IAM permissions")
    else:
        st.error(f"❌ S3 connection error: {e}")
    st.stop()
except Exception as e:
    status_text.empty()
    st.error(f"❌ Failed to connect to S3: {e}")
    st.error("Please check:")
    st.error("1. S3 credentials in secrets.toml")
    st.error("2. Bucket name and region")
    st.error("3. Network connection")
    import traceback
    with st.expander("Show full error details"):
        st.code(traceback.format_exc())
    st.stop()

# Always load all files - caching handles performance
# First load will process all PDFs, then cached indefinitely for instant access

# Now load the actual data
logger.debug("Entering data loading section...")
try:
    status_text.text("📥 Loading PDF files from S3...")
    logger.debug("Status text updated, starting timer...")
    
    t0 = time.time()
    was_processing = False  # Track if we actually processed files
    logger.debug(f"Timer started at {t0}")
    
    # Check if we have merged data from smart refresh
    logger.debug("Checking for merged calls in session state...")
    if 'merged_calls' in st.session_state:
        logger.info("Found merged calls in session state, using cached data")
        # Use merged data from smart refresh
        call_data = st.session_state['merged_calls']
        errors = st.session_state.get('merged_errors', [])
        # Clear the temporary session state
        del st.session_state['merged_calls']
        if 'merged_errors' in st.session_state:
            del st.session_state['merged_errors']
        # Note: Disk cache already has the merged data from refresh, Streamlit cache will update on next access
        elapsed = time.time() - t0
        status_text.empty()
        logger.info(f"Merged data loaded in {elapsed:.2f} seconds")
    else:
        # Check if refresh is in progress - skip main data loading to prevent conflicts
        if st.session_state.get('refresh_in_progress', False):
            logger.info("⏸️ Refresh in progress - skipping main data load to prevent conflicts")
            # Use disk cache if available, but don't trigger a new load
            try:
                disk_result = load_cached_data_from_disk()
                # CRITICAL FIX: Check if disk_result is None before accessing its elements
                if disk_result and disk_result[0] and len(disk_result[0]) > 0:
                    call_data = disk_result[0]
                    # CRITICAL FIX: Check if disk_result is None before accessing disk_result[1]
                    errors = disk_result[1] if (disk_result and disk_result[1] is not None) else []
                    logger.info(f"📂 Using disk cache during refresh: {len(call_data)} calls")
                else:
                    call_data = []
                    errors = []
                    logger.info("📂 No disk cache available during refresh - will load after refresh completes")
            except Exception as e:
                logger.warning(f"Could not load disk cache during refresh: {e}")
                call_data = []
                errors = []
            elapsed = time.time() - t0
            status_text.empty()
        else:
            logger.debug("No merged calls found, proceeding with normal load from cache or S3")
            # Normal load from cache or S3
            # Load all files - first load will process all PDFs, then cached indefinitely for instant access
            # After first load, data is CACHED indefinitely - subsequent loads will be INSTANT until you manually refresh
            logger.debug("Setting up progress tracking...")
            
            # Initialize progress tracking
            if 'pdf_processing_progress' not in st.session_state:
                st.session_state.pdf_processing_progress = {'processed': 0, 'total': 0, 'errors': 0}
                logger.debug("Initialized new progress tracking in session state")
            else:
                logger.debug("Using existing progress tracking from session state")
            
            # Create progress bar placeholder
            progress_placeholder = st.empty()
            progress_bar = None
            logger.debug("Progress placeholder created")
            
            # Show progress if we're processing files
            def update_progress():
                if st.session_state.pdf_processing_progress['total'] > 0:
                    processed = st.session_state.pdf_processing_progress['processed']
                    total = st.session_state.pdf_processing_progress['total']
                    errors = st.session_state.pdf_processing_progress['errors']
                    progress = processed / total if total > 0 else 0
                    progress_placeholder.progress(progress, text=f"Processing PDFs: {processed}/{total} ({errors} errors)")
            
            # Load data (this will trigger processing if not cached)
            # SIMPLE approach: Just call the cached function. Streamlit's cache handles everything.
            # If cache exists, it's instant. If not, it loads from S3 (only happens once, then cached).
            logger.debug("Loading data - Streamlit cache will handle it automatically")
        
        try:
            # Add timeout wrapper
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Data loading timed out after 5 minutes")
            
            # Try to load with better error visibility
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.spinner("Loading PDFs from S3... This may take a few minutes for large datasets.")
            
            # Use cache_version to force cache refresh when refresh completes
            cache_version = st.session_state.get('_cache_version', 0)
            call_data, errors = load_all_calls_cached(cache_version=cache_version)
            logger.info(f"Data loaded. Got {len(call_data) if call_data else 0} calls")
            
            # Clear loading messages
            loading_placeholder.empty()
        except TimeoutError as e:
            logger.exception("Timeout during data loading")
            loading_placeholder.empty()
            status_text.empty()
            st.error("⏱️ **Loading Timeout**")
            st.error("The data loading is taking too long. This might be due to:")
            st.error("1. Slow S3 connection")
            st.error("2. Large number of files to process")
            st.error("3. Network issues")
            st.info("💡 **Quick Fixes:**")
            st.info("1. **Refresh the page** - if cache exists, it will load instantly")
            st.info("2. **Wait 2-3 minutes** and refresh - the cache may be building")
            st.info("3. **Check your internet connection**")
            st.info("4. If you're an admin, try the '🔄 Reload ALL Data' button after refresh")
            st.stop()
        except Exception as e:
            logger.exception("Error during data loading")
            loading_placeholder.empty()
            status_text.empty()
            st.error("❌ **Error Loading Data**")
            st.error(f"**Error:** {str(e)}")
            st.error("The app may be trying to load too many files at once.")
            st.info("💡 **Try this:**")
            st.info("1. **Refresh the page** - if cache exists, it will load instantly")
            st.info("2. Clear the cache by clicking '🔄 Reload ALL Data (Admin Only)' button (if you're an admin)")
            st.info("3. Wait a few minutes and refresh the page")
            st.info("4. Check the terminal/logs for detailed errors")
            with st.expander("Show full error details"):
                import traceback
                st.code(traceback.format_exc())
            st.stop()
        
        # Clear progress after loading
        was_processing = st.session_state.pdf_processing_progress.get('total', 0) > 0
        if was_processing:
            progress_placeholder.empty()
            st.session_state.pdf_processing_progress = {'processed': 0, 'total': 0, 'errors': 0, 'processing_start_time': None}
        
        elapsed = time.time() - t0
        status_text.empty()
    
    # Auto-load new files if detected during startup
    # OPTIMIZATION: Skip auto-refresh if manual refresh is in progress to prevent conflicts
    if 'auto_refresh_pending' in st.session_state and st.session_state.auto_refresh_pending > 0:
        if st.session_state.get('refresh_in_progress', False):
            logger.info("⏸️ Skipping auto-refresh - manual refresh in progress")
            # Clear the pending flag so it doesn't retry immediately
            del st.session_state.auto_refresh_pending
        else:
            new_count = st.session_state.auto_refresh_pending
            logger.info(f"🔄 Auto-loading {new_count} new files detected during startup...")
            with st.spinner(f"🔄 Loading {new_count} new PDF(s) in background..."):
                new_calls, new_errors, actual_new_count = load_new_calls_only()
                
                if isinstance(new_errors, str):
                    logger.error(f"❌ Error auto-loading new files: {new_errors}")
                    st.warning(f"⚠️ Could not auto-load new files: {new_errors}")
                elif actual_new_count > 0:
                    # Merge with existing data
                    disk_result = load_cached_data_from_disk()
                    # CRITICAL FIX: Check if disk_result is None before accessing its elements
                    existing_calls = disk_result[0] if (disk_result and disk_result[0] is not None) else []
                    all_calls_merged = existing_calls + new_calls
                    all_calls_merged = deduplicate_calls(all_calls_merged)
                    
                    # CRITICAL FIX: Use safe save function instead of direct file write
                    # This provides atomic writes, file locking, and proper error handling
                    # CRITICAL FIX: Wrap in try-except since save_cached_data_to_disk() re-raises exceptions
                    try:
                        save_cached_data_to_disk(
                            all_calls_merged,
                            new_errors if new_errors else [],
                            partial=False
                        )
                        logger.info(f"✅ Auto-loaded {actual_new_count} new files. Total: {len(all_calls_merged)} calls")
                        st.success(f"✅ Auto-loaded {actual_new_count} new file(s)! Total: {len(all_calls_merged)} calls")
                    except Exception as save_error:
                        logger.error(f"❌ CRITICAL: Failed to save auto-loaded cache: {save_error}")
                        logger.error(f"❌ Auto-loaded data will be lost on next restart - continuing anyway")
                        # Still show success message since files were loaded, just not saved
                        logger.info(f"✅ Auto-loaded {actual_new_count} new files. Total: {len(all_calls_merged)} calls (NOT SAVED)")
                        st.success(f"✅ Auto-loaded {actual_new_count} new file(s)! Total: {len(all_calls_merged)} calls")
                        st.warning(f"⚠️ Warning: Failed to save cache - data may be lost on restart")
                    
                    # Update call_data to include new files
                    call_data = all_calls_merged
                    errors = new_errors if new_errors else []
                    
                    # Clear Streamlit cache to force reload
                    load_all_calls_cached.clear()
                    
                    # Clear flags
                    del st.session_state.auto_refresh_pending
                    st.session_state.new_pdfs_notification_count = 0
                    
                    # Rerun to show updated data
                    st.rerun()
                else:
                    logger.info("ℹ️ No new files to load (may have been processed already)")
                    del st.session_state.auto_refresh_pending
                    st.session_state.new_pdfs_notification_count = 0
    
    # Check if we got valid data
    if not call_data and not errors:
        status_text.empty()
        st.warning("⚠️ No data loaded. This might be the first time loading, or there may be an issue.")
        st.info("💡 Try refreshing the page or clicking '🔄 Reload ALL Data (Admin Only)' if you're an admin.")
        st.stop()
    
    # Handle errors - could be a tuple (errors_list, info_message) or just errors
    if errors:
        if isinstance(errors, tuple) and len(errors) == 2:
            # New format: (errors_list, info_message)
            errors_list, info_msg = errors
            st.info(f"ℹ️ {info_msg}")
            if errors_list:
                if len(errors_list) <= 10:
                    for error in errors_list:
                        st.warning(f"⚠️ {error}")
                else:
                    st.warning(f"⚠️ {len(errors_list)} files had errors. Showing first 10:")
                    for error in errors_list[:10]:
                        st.warning(f"⚠️ {error}")
        elif isinstance(errors, list) and len(errors) <= 10:
            for error in errors:
                st.warning(f"⚠️ {error}")
        elif isinstance(errors, list) and len(errors) > 10:
            st.warning(f"⚠️ {len(errors)} files had errors. Showing first 10:")
            for error in errors[:10]:
                st.warning(f"⚠️ {error}")
        elif isinstance(errors, str):
            # Single error message
            st.error(f"❌ {errors}")
            st.stop()
    
    if call_data:
        # Check if we actually processed files or loaded from cache
        if was_processing and 'last_actual_processing_time' in st.session_state:
            # We actually processed files - show actual processing time
            actual_time = st.session_state['last_actual_processing_time']
            file_count = st.session_state.get('last_processing_file_count', len(call_data))
            if actual_time > 60:
                time_str = f"{actual_time/60:.1f} minutes"
            else:
                time_str = f"{actual_time:.1f}s"
            st.success(f"✅ Loaded {file_count} calls in {time_str}")
        elif 'last_actual_processing_time' in st.session_state:
            # Data loaded from cache - show when it was last processed
            last_time = st.session_state['last_actual_processing_time']
            if last_time > 60:
                time_str = f"{last_time/60:.1f} minutes"
            else:
                time_str = f"{last_time:.1f}s"
            if current_username and current_username.lower() in ["chloe", "shannon"]:
                st.success(f"✅ Loaded {len(call_data)} calls (from cache, originally processed in {time_str})")
        else:
            # First time or no processing time tracked - show cache retrieval time
            if current_username and current_username.lower() in ["chloe", "shannon"]:
                st.success(f"✅ Loaded {len(call_data)} calls (from cache)")
    else:
        st.error("❌ No call data found!")
        st.error("Possible issues:")
        st.error("1. No PDF files in S3 bucket (check bucket name and prefix)")
        st.error("2. PDF files couldn't be parsed")
        st.error("3. Check the prefix path if PDFs are in a subfolder")
        st.stop()
except Exception as e:
    status_text.empty()
    st.error(f"❌ Error loading data: {e}")
    st.error("Please check:")
    st.error("1. S3 credentials in secrets.toml")
    st.error("2. Bucket name and region")
    st.error("3. AWS permissions")
    import traceback
    with st.expander("Show full error"):
        st.code(traceback.format_exc())
    st.stop()

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
    unique_values = sorted([str(v).strip() for v in values if pd.notna(v) and str(v).strip()])
    
    # Create mapping: real value -> anonymous ID (Agent-001, Agent-002, etc.)
    mapping = {}
    for idx, value in enumerate(unique_values, start=1):
        mapping[value] = f"{prefix}-{idx:03d}"
    
    return mapping

# Store anonymization mappings in session state for consistency across the app
if 'anonymization_mappings' not in st.session_state:
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
            st.session_state.anonymization_mappings["agent_mapping"] = create_anonymous_mapping(
                mapping_source["Agent"].dropna().unique(), "Agent"
            )
        agent_mapping = st.session_state.anonymization_mappings["agent_mapping"]
        if "Agent" in df.columns and agent_mapping:
            # Use vectorized replace for much faster performance
            # Handle NaN values properly
            mask = df["Agent"].notna()
            df.loc[mask, "Agent"] = df.loc[mask, "Agent"].astype(str).str.strip().replace(agent_mapping)
    
    if "Company" in mapping_source.columns:
        if "company_mapping" not in st.session_state.anonymization_mappings:
            logger.info("Creating company anonymization mapping...")
            st.session_state.anonymization_mappings["company_mapping"] = create_anonymous_mapping(
                mapping_source["Company"].dropna().unique(), "Company"
            )
        company_mapping = st.session_state.anonymization_mappings["company_mapping"]
        if "Company" in df.columns and company_mapping:
            # Use vectorized replace for much faster performance
            # Handle NaN values properly
            mask = df["Company"].notna()
            df.loc[mask, "Company"] = df.loc[mask, "Company"].astype(str).str.strip().replace(company_mapping)
    
    if "Call ID" in mapping_source.columns:
        if "call_id_mapping" not in st.session_state.anonymization_mappings:
            logger.info("Creating call ID anonymization mapping...")
            st.session_state.anonymization_mappings["call_id_mapping"] = create_anonymous_mapping(
                mapping_source["Call ID"].dropna().unique(), "Call"
            )
        call_id_mapping = st.session_state.anonymization_mappings["call_id_mapping"]
        if "Call ID" in df.columns and call_id_mapping:
            # Use vectorized replace for much faster performance
            # Handle NaN values properly
            mask = df["Call ID"].notna()
            df.loc[mask, "Call ID"] = df.loc[mask, "Call ID"].astype(str).str.strip().replace(call_id_mapping)
    
    return df

# Convert call_date to datetime if it's not already
if "call_date" in meta_df.columns:
    # If call_date is already datetime, keep it; otherwise try to parse
    if meta_df["call_date"].dtype == 'object':
        meta_df["call_date"] = pd.to_datetime(meta_df["call_date"], errors="coerce")

# --- Normalize Agent IDs to bpagent## format ---
def normalize_agent_id(agent_str):
    """Normalize agent ID to bpagent## format (e.g., bpagent01, bpagent02)"""
    if pd.isna(agent_str) or not agent_str:
        return agent_str
    
    agent_str = str(agent_str).lower().strip()
    
    # Extract number from bpagent### pattern (could be bpagent01, bpagent030844482, etc.)
    match = re.search(r'bpagent(\d+)', agent_str)
    if match:
        # Get the number and take only first 2 digits (or pad to 2 digits)
        agent_num = match.group(1)
        # If number is longer than 2 digits, take first 2; otherwise pad to 2 digits
        if len(agent_num) >= 2:
            agent_num = agent_num[:2]  # Take first 2 digits
        else:
            agent_num = agent_num.zfill(2)  # Pad to 2 digits
        return f"bpagent{agent_num}"
    
    # If no match, return as is
    return agent_str

# --- Normalize QA fields ---
meta_df.rename(columns={
    "company":        "Company",
    "agent":          "Agent",
    "call_date":      "Call Date",
    "date_raw":       "Date Raw",
    "time":           "Call Time",
    "call_id":        "Call ID",
    "qa_score":       "QA Score",
    "label":          "Label",
    "reason":         "Reason",
    "outcome":        "Outcome",
    "summary":        "Summary",
    "strengths":      "Strengths",
    "challenges":     "Challenges",
    "coaching_suggestions": "Coaching Suggestions",
    "rubric_details": "Rubric Details",
    "rubric_pass_count": "Rubric Pass Count",
    "rubric_fail_count": "Rubric Fail Count",
}, inplace=True)

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
                    except:
                        pass
            return total
        return None
    meta_df["Call Duration (s)"] = meta_df.apply(compute_speaking_time, axis=1)
else:
    meta_df["Call Duration (s)"] = None

meta_df["Call Duration (min)"] = meta_df["Call Duration (s)"] / 60 if "Call Duration (s)" in meta_df.columns else None

# Ensure QA Score is numeric
if "QA Score" in meta_df.columns:
    meta_df["QA Score"] = pd.to_numeric(meta_df["QA Score"], errors="coerce")

# Date handling
if ("Call Date" not in meta_df.columns) or meta_df["Call Date"].isna().all():
    if "Date Raw" in meta_df.columns:
        meta_df["Call Date"] = pd.to_datetime(meta_df["Date Raw"], format="%m%d%Y", errors="coerce")
    else:
        st.sidebar.error("❌ Neither Call Date nor Date Raw found—cannot parse any dates.")
        st.stop()

meta_df.dropna(subset=["Call Date"], inplace=True)

# Normalize agent IDs AFTER column rename (works for both cached and new data)
# This ensures normalization works regardless of whether data came from cache or fresh load
if "Agent" in meta_df.columns:
    meta_df["Agent"] = meta_df["Agent"].apply(normalize_agent_id)

# Apply anonymization if user is anonymous
# Create mappings from full dataset before any filtering
if is_anonymous_user:
    import time
    anonymize_start = time.time()
    logger.info("🔒 Anonymous user detected - creating anonymization mappings from full dataset")
    # Create mappings from the full meta_df to ensure consistency
    # Only create mappings if they don't exist (faster on subsequent loads)
    if not st.session_state.anonymization_mappings:
        _ = anonymize_dataframe(meta_df, create_mappings_from=meta_df)
    # Now apply anonymization to meta_df (this is fast with vectorized operations)
    meta_df = anonymize_dataframe(meta_df, create_mappings_from=meta_df)
    anonymize_duration = time.time() - anonymize_start
    logger.info(f"🔒 Anonymization completed in {anonymize_duration:.2f}s")
    st.sidebar.warning("🔒 **Anonymous Mode**: All identifying information has been anonymized")

# --- Determine if agent view or admin view ---
# Normalize user_agent_id if it exists
if user_agent_id:
    user_agent_id = normalize_agent_id(user_agent_id)

# If user has agent_id, automatically filter to their data
if user_agent_id:
    # Agent view - filter to their calls only
    agent_calls_df = meta_df[meta_df["Agent"] == user_agent_id].copy()
    
    if agent_calls_df.empty:
        st.warning(f"⚠️ No calls found for agent: {user_agent_id}")
        st.info("If this is incorrect, please contact your administrator to update your agent ID mapping.")
        st.stop()
    
    # Use agent's data for filtering
    filter_df = agent_calls_df
    show_comparison = True  # Always show comparison for agents
    st.sidebar.info(f"📊 Showing your calls only ({len(agent_calls_df)} calls)")
else:
    # Admin/All data view
    filter_df = meta_df
    show_comparison = False
    st.sidebar.info(f"📊 Showing all data ({len(meta_df)} calls)")

# --- Sidebar Filters ---
st.sidebar.header("📊 Filter Data")

# Date filter
min_date = filter_df["Call Date"].min()
max_date = filter_df["Call Date"].max()
dates = filter_df["Call Date"].dropna().sort_values().dt.date.unique().tolist()

if not dates:
    st.warning("⚠️ No calls with valid dates to display.")
    st.stop()

# Remember last filter settings
if 'last_date_preset' not in st.session_state:
    st.session_state.last_date_preset = "All Time"
if 'last_date_range' not in st.session_state:
    st.session_state.last_date_range = None
if 'last_agents' not in st.session_state:
    st.session_state.last_agents = []
if 'last_score_range' not in st.session_state:
    st.session_state.last_score_range = None
if 'last_labels' not in st.session_state:
    st.session_state.last_labels = []
if 'last_search' not in st.session_state:
    st.session_state.last_search = ""
if 'last_preset_filter' not in st.session_state:
    st.session_state.last_preset_filter = "None"
if 'selected_rubric_codes' not in st.session_state:
    st.session_state.selected_rubric_codes = []
if 'rubric_filter_type' not in st.session_state:
    st.session_state.rubric_filter_type = "Any Status"

# Dark mode toggle (admin only)
if is_regular_admin():
    st.sidebar.markdown("---")
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False, help="Toggle dark mode (requires page refresh)")
    if dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        </style>
        """, unsafe_allow_html=True)

# Keyboard shortcuts info
with st.sidebar.expander("⌨️ Keyboard Shortcuts"):
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
    index=["All Time", "This Week", "Last 7 Days", "Last 30 Days", "Custom"].index(st.session_state.last_date_preset) if st.session_state.last_date_preset in ["All Time", "This Week", "Last 7 Days", "Last 30 Days", "Custom"] else 0
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
    default_date_range = st.session_state.last_date_range if st.session_state.last_date_range and isinstance(st.session_state.last_date_range, tuple) else (min(dates), max(dates))
    custom_input = st.sidebar.date_input("Select Date Range", value=default_date_range)
    if isinstance(custom_input, tuple) and len(custom_input) == 2:
        selected_dates = custom_input
        st.session_state.last_date_range = custom_input  # Save selection
    elif isinstance(custom_input, date):
        selected_dates = (custom_input, custom_input)
        st.session_state.last_date_range = selected_dates  # Save selection
    else:
        st.warning("⚠️ Please select a valid date range.")
        st.stop()

# Extract start_date and end_date from selected_dates (works for both preset and custom)
start_date, end_date = selected_dates

# Agent filter (only for admin view)
if not user_agent_id:
    available_agents = filter_df["Agent"].dropna().unique().tolist()
    # Restore last selection or use all agents
    default_agents = st.session_state.last_agents if st.session_state.last_agents and all(a in available_agents for a in st.session_state.last_agents) else available_agents
    selected_agents = st.sidebar.multiselect("👤 Select Agents", available_agents, default=default_agents)
    st.session_state.last_agents = selected_agents  # Save selection
else:
    # For agents, they only see their own data
    selected_agents = [user_agent_id]

# QA Score filter
if "QA Score" in meta_df.columns and not meta_df["QA Score"].isna().all():
    min_score = float(meta_df["QA Score"].min())
    max_score = float(meta_df["QA Score"].max())
    # Restore last score range or use full range
    default_score_range = st.session_state.last_score_range if st.session_state.last_score_range and st.session_state.last_score_range[0] >= min_score and st.session_state.last_score_range[1] <= max_score else (min_score, max_score)
    score_range = st.sidebar.slider(
        "📊 QA Score Range",
        min_value=min_score,
        max_value=max_score,
        value=default_score_range,
        step=1.0
    )
    st.session_state.last_score_range = score_range  # Save selection
else:
    score_range = None

# Label filter
if "Label" in meta_df.columns:
    available_labels = meta_df["Label"].dropna().unique().tolist()
    # Restore last selection or use all labels
    default_labels = st.session_state.last_labels if st.session_state.last_labels and all(l in available_labels for l in st.session_state.last_labels) else available_labels
    selected_labels = st.sidebar.multiselect("🏷️ Select Labels", available_labels, default=default_labels)
    st.session_state.last_labels = selected_labels  # Save selection
else:
    selected_labels = []

# Filter Presets
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Filter Presets")

# Load saved filter presets
if 'saved_filter_presets' not in st.session_state:
    st.session_state.saved_filter_presets = {}

# Quick preset filters
preset_options = ["None", "High Performers (90%+)", "Needs Improvement (<70%)", "Failed Rubric Items", "Recent Issues (Last 7 Days)"]
preset_index = preset_options.index(st.session_state.last_preset_filter) if st.session_state.last_preset_filter in preset_options else 0
preset_filters = st.sidebar.radio(
    "Quick Filters",
    options=preset_options,
    index=preset_index,
    help="Apply common filter combinations quickly"
)
st.session_state.last_preset_filter = preset_filters  # Save selection

# Saved filter presets management
st.sidebar.markdown("---")
with st.sidebar.expander("💾 Saved Filter Presets"):
    if st.session_state.saved_filter_presets:
        st.write("**Saved Presets:**")
        for preset_name in st.session_state.saved_filter_presets.keys():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📌 {preset_name}", key=f"load_{preset_name}"):
                    # Load preset
                    preset = st.session_state.saved_filter_presets[preset_name]
                    st.session_state.last_date_preset = preset.get('date_preset', 'All Time')
                    st.session_state.last_date_range = preset.get('date_range', None)
                    st.session_state.last_agents = preset.get('agents', [])
                    st.session_state.last_score_range = preset.get('score_range', None)
                    st.session_state.last_labels = preset.get('labels', [])
                    st.session_state.last_search = preset.get('search', '')
                    st.session_state.last_preset_filter = preset.get('preset_filter', 'None')
                    st.session_state.selected_rubric_codes = preset.get('rubric_codes', [])
                    st.session_state.rubric_filter_type = preset.get('rubric_filter_type', 'Any Status')
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{preset_name}", help=f"Delete {preset_name}"):
                    del st.session_state.saved_filter_presets[preset_name]
                    st.rerun()
    
    # Save current filter as preset
    st.markdown("---")
    preset_name = st.text_input("Save current filters as:", placeholder="e.g., 'Weekly Review'", key="new_preset_name")
    if st.button("💾 Save Preset") and preset_name:
        if preset_name in st.session_state.saved_filter_presets:
            st.warning(f"Preset '{preset_name}' already exists. Overwrite?")
        else:
            # Save current filter state
            current_preset = {
                'date_preset': st.session_state.last_date_preset,
                'date_range': st.session_state.last_date_range,
                'agents': st.session_state.last_agents,
                'score_range': st.session_state.last_score_range,
                'labels': st.session_state.last_labels,
                'search': st.session_state.last_search,
                'preset_filter': st.session_state.last_preset_filter,
                'rubric_codes': st.session_state.get('selected_rubric_codes', []),
                'rubric_filter_type': st.session_state.get('rubric_filter_type', 'Any Status'),
                'created_at': datetime.now().isoformat()
            }
            st.session_state.saved_filter_presets[preset_name] = current_preset
            st.success(f"✅ Preset '{preset_name}' saved!")
            st.rerun()

# Search functionality - Enhanced full-text search
st.sidebar.markdown("---")
search_text = st.sidebar.text_input(
    "🔍 Full-Text Search", 
    st.session_state.last_search,
    help="Search across Reason, Summary, Outcome, Strengths, Challenges, and Coaching Suggestions"
)
st.session_state.last_search = search_text  # Save search

# Rubric code filter - Enhanced to support multiple codes (pass/fail/any)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Rubric Code Filters")

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
            horizontal=True
        )
        
        selected_rubric_codes = st.sidebar.multiselect(
            f"Select Rubric Codes ({rubric_filter_type})",
            options=all_rubric_codes,
            default=st.session_state.selected_rubric_codes if st.session_state.selected_rubric_codes else [],
            help="Show calls that match these rubric codes"
        )
        st.session_state.selected_rubric_codes = selected_rubric_codes
        st.session_state.rubric_filter_type = rubric_filter_type
        
        # Also collect failed codes for backward compatibility
        failed_rubric_codes = [code for code in all_rubric_codes]
        selected_failed_codes = selected_rubric_codes if rubric_filter_type == "Failed Only" else []
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
    "⚠️ Alert Threshold (QA Score)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=5.0,
    help="Agents/calls below this score will be highlighted"
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
    (filter_df["Agent"].isin(selected_agents)) &
    (filter_df["Call Date"].dt.date >= start_date) &
    (filter_df["Call Date"].dt.date <= end_date)
].copy()

# Apply QA Score filter
if score_range:
    filtered_df = filtered_df[
        (filtered_df["QA Score"] >= score_range[0]) &
        (filtered_df["QA Score"] <= score_range[1])
    ].copy()

# Apply Label filter
if selected_labels:
    filtered_df = filtered_df[filtered_df["Label"].isin(selected_labels)].copy()

# Apply enhanced full-text search
if search_text:
    search_lower = search_text.lower()
    search_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
    
    # Search across multiple fields
    search_fields = ["Reason", "Summary", "Outcome", "Strengths", "Challenges", "Coaching Suggestions"]
    for field in search_fields:
        if field in filtered_df.columns:
            field_mask = filtered_df[field].astype(str).str.lower().str.contains(search_lower, na=False)
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
                        status = details.get('status', '').lower()
                        if rubric_filter_type == "Any Status":
                            rubric_mask[idx] = True
                            break
                        elif rubric_filter_type == "Failed Only" and status == 'fail':
                            rubric_mask[idx] = True
                            break
                        elif rubric_filter_type == "Passed Only" and status == 'pass':
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
                    if isinstance(details, dict) and details.get('status') == 'Fail':
                        failed_mask[idx] = True
                        break
    
    filtered_df = filtered_df[failed_mask].copy()

# Calculate overall averages for comparison (from all data, not filtered)
if show_comparison and user_agent_id:
    # Get overall averages from all agents' data in the same date range
    overall_df = meta_df[
        (meta_df["Call Date"].dt.date >= start_date) &
        (meta_df["Call Date"].dt.date <= end_date)
    ].copy()
    
    overall_avg_score = overall_df["QA Score"].mean() if "QA Score" in overall_df.columns else 0
    overall_total_pass = overall_df["Rubric Pass Count"].sum() if "Rubric Pass Count" in overall_df.columns else 0
    overall_total_fail = overall_df["Rubric Fail Count"].sum() if "Rubric Fail Count" in overall_df.columns else 0
    overall_pass_rate = (overall_total_pass / (overall_total_pass + overall_total_fail) * 100) if (overall_total_pass + overall_total_fail) > 0 else 0
    overall_total_calls = len(overall_df)
else:
    overall_avg_score = None
    overall_pass_rate = None
    overall_total_calls = None

if score_range:
    filtered_df = filtered_df[
        (filtered_df["QA Score"] >= score_range[0]) &
        (filtered_df["QA Score"] <= score_range[1])
    ]

if selected_labels:
    filtered_df = filtered_df[filtered_df["Label"].isin(selected_labels)]

# Search filter
if search_text:
    search_lower = search_text.lower()
    mask = (
        filtered_df["Reason"].astype(str).str.lower().str.contains(search_lower, na=False) |
        filtered_df["Summary"].astype(str).str.lower().str.contains(search_lower, na=False) |
        filtered_df["Outcome"].astype(str).str.lower().str.contains(search_lower, na=False)
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
                    if isinstance(details, dict) and details.get('status') == 'Fail':
                        return True
        return False
    
    filtered_df = filtered_df[filtered_df.apply(lambda row: has_failed_code(row, selected_failed_codes), axis=1)]

if filtered_df.empty:
    st.warning("⚠️ No data matches the current filter selection.")
    st.stop()

# --- Main Dashboard ---
if user_agent_id:
    st.title(f"📋 My QA Performance Dashboard - {user_agent_id}")
else:
    st.title("📋 QA Rubric Dashboard")

# Monitoring Dashboard (Chloe, Shannon, and Jerson only)
if is_super_admin():
    with st.expander("📊 System Monitoring & Metrics (Chloe, Shannon, and Jerson only)", expanded=False):
        st.markdown("### Usage Metrics")
        metrics = load_metrics()
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Sessions", metrics.get("sessions", 0))
        with metric_col2:
            total_errors = sum(e.get("count", 0) for e in metrics.get("errors", {}).values())
            st.metric("Total Errors", total_errors)
        with metric_col3:
            unique_features = len(metrics.get("features_used", {}))
            st.metric("Features Used", unique_features)
        
        # Show repeated failures
        repeated_failures = {k: v for k, v in metrics.get("errors", {}).items() if v.get("count", 0) >= 5}
        if repeated_failures:
            st.warning(f"⚠️ **{len(repeated_failures)} repeated failure(s) detected:**")
            for error_key, error_data in sorted(repeated_failures.items(), key=lambda x: x[1]["count"], reverse=True):
                st.markdown(f"- **{error_key}**: {error_data['count']} occurrences (First: {error_data.get('first_seen', 'N/A')}, Last: {error_data.get('last_seen', 'N/A')})")
        
        # Show feature usage
        if metrics.get("features_used"):
            st.markdown("### Feature Usage")
            feature_df = pd.DataFrame([
                {"Feature": k, "Usage Count": v}
                for k, v in sorted(metrics.get("features_used", {}).items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(feature_df, hide_index=True)
        
        # Show recent errors
        if metrics.get("errors"):
            st.markdown("### Recent Errors")
            error_df = pd.DataFrame([
                {
                    "Error": k.split(":")[0],
                    "Message": k.split(":", 1)[1] if ":" in k else k,
                    "Count": v.get("count", 0),
                    "Last Seen": v.get("last_seen", "N/A")
                }
                for k, v in sorted(metrics.get("errors", {}).items(), key=lambda x: x[1].get("last_seen", ""), reverse=True)[:10]
            ])
            st.dataframe(error_df, hide_index=True)
        
        if st.button("🔄 Refresh Metrics"):
            st.rerun()
        
        # Audit Log Viewer (Shannon and Chloe only)
        if current_username and current_username.lower() in ["chloe", "shannon"]:
            st.markdown("---")
            st.markdown("### 🔍 Audit Log")
            audit_file = Path("logs/audit_log.json")
            if audit_file.exists():
                try:
                    with open(audit_file, 'r') as f:
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
                        selected_action = st.selectbox("Filter by action:", ["All"] + action_types)
                        if selected_action != "All":
                            filtered_audit = audit_df[audit_df["action"] == selected_action]
                            st.dataframe(filtered_audit, hide_index=True)
                    else:
                        st.info("No audit entries yet.")
                except Exception as e:
                    st.error(f"Error loading audit log: {e}")
            else:
                st.info("Audit log file not found. Audit entries will be created as you use the system.")

# Data Validation Dashboard (Chloe, Shannon, and Jerson only)
if is_super_admin():
    with st.expander("🔍 Data Quality Validation (Chloe, Shannon, and Jerson only)", expanded=False):
        st.markdown("### Data Quality Metrics")
        
        validation_issues = []
        validation_stats = {}
        
        # Check for missing required fields
        required_fields = ['Agent', 'Call Date', 'QA Score', 'Label']
        for field in required_fields:
            if field in meta_df.columns:
                missing_count = meta_df[field].isna().sum()
                total_count = len(meta_df)
                missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
                validation_stats[field] = {
                    'missing': missing_count,
                    'total': total_count,
                    'pct': missing_pct
                }
                if missing_count > 0:
                    validation_issues.append(f"**{field}**: {missing_count} missing ({missing_pct:.1f}%)")
        
        # Check for invalid QA scores
        if "QA Score" in meta_df.columns:
            invalid_scores = meta_df[(meta_df["QA Score"] < 0) | (meta_df["QA Score"] > 100)]
            invalid_count = len(invalid_scores)
            if invalid_count > 0:
                validation_issues.append(f"**QA Score**: {invalid_count} invalid scores (outside 0-100%)")
            validation_stats['Invalid QA Scores'] = invalid_count
        
        # Check for missing rubric details
        if "Rubric Details" in meta_df.columns:
            missing_rubric = meta_df["Rubric Details"].isna().sum()
            if missing_rubric > 0:
                validation_issues.append(f"**Rubric Details**: {missing_rubric} calls missing rubric data")
            validation_stats['Missing Rubric'] = missing_rubric
        
        # Check for duplicate call IDs
        if "Call ID" in meta_df.columns:
            duplicates = meta_df[meta_df["Call ID"].duplicated(keep=False)]
            duplicate_count = len(duplicates)
            if duplicate_count > 0:
                validation_issues.append(f"**Call ID**: {duplicate_count} duplicate call IDs found")
            validation_stats['Duplicate Call IDs'] = duplicate_count
        
        # Display validation results
        if validation_issues:
            st.warning(f"⚠️ Found {len(validation_issues)} data quality issue(s):")
            for issue in validation_issues:
                st.markdown(f"- {issue}")
        else:
            st.success("✅ No data quality issues detected!")
        
        # Show detailed stats table
        if validation_stats:
            st.markdown("### Detailed Statistics")
            stats_df = pd.DataFrame([
                {'Field': k, 'Missing/Invalid Count': v.get('missing', v) if isinstance(v, dict) else v, 
                 'Total': str(v.get('total', 'N/A')) if isinstance(v, dict) else 'N/A',
                 'Percentage': f"{v.get('pct', 0):.1f}%" if isinstance(v, dict) else 'N/A'}
                for k, v in validation_stats.items()
            ])
            st.dataframe(stats_df, hide_index=True)

# Summary Metrics
if show_comparison and user_agent_id:
    # Agent view with comparison
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        my_calls = len(filtered_df)
        st.metric("My Calls", my_calls, delta=f"{my_calls - overall_total_calls}" if overall_total_calls else None)
    
    with col2:
        my_avg_score = filtered_df["QA Score"].mean() if "QA Score" in filtered_df.columns else 0
        delta_score = my_avg_score - overall_avg_score if overall_avg_score else None
        st.metric("My Avg Score", f"{my_avg_score:.1f}%", 
                 delta=f"{delta_score:+.1f}%" if delta_score is not None else None,
                 delta_color="normal" if delta_score and delta_score >= 0 else "inverse")
    
    with col3:
        my_total_pass = filtered_df["Rubric Pass Count"].sum() if "Rubric Pass Count" in filtered_df.columns else 0
        my_total_fail = filtered_df["Rubric Fail Count"].sum() if "Rubric Fail Count" in filtered_df.columns else 0
        my_pass_rate = (my_total_pass / (my_total_pass + my_total_fail) * 100) if (my_total_pass + my_total_fail) > 0 else 0
        delta_pass = my_pass_rate - overall_pass_rate if overall_pass_rate else None
        st.metric("My Pass Rate", f"{my_pass_rate:.1f}%",
                 delta=f"{delta_pass:+.1f}%" if delta_pass is not None else None,
                 delta_color="normal" if delta_pass and delta_pass >= 0 else "inverse")
    
    with col4:
        st.metric("Overall Avg Score", f"{overall_avg_score:.1f}%" if overall_avg_score else "N/A")
    
    with col5:
        st.metric("Overall Pass Rate", f"{overall_pass_rate:.1f}%" if overall_pass_rate else "N/A")
    
    # Comparison section
    st.subheader("📊 My Performance vs. Team Average")
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.write("**QA Score Comparison**")
        comparison_data = pd.DataFrame({
            'Metric': ['My Average', 'Team Average'],
            'QA Score': [my_avg_score, overall_avg_score]
        })
        fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
        comparison_data.plot(x='Metric', y='QA Score', kind='bar', ax=ax_comp, color=['steelblue', 'orange'])
        ax_comp.set_ylabel("QA Score (%)")
        ax_comp.set_title("My Score vs Team Average")
        ax_comp.axhline(y=alert_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({alert_threshold}%)')
        ax_comp.legend()
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_comp)
    
    with comp_col2:
        st.write("**Pass Rate Comparison**")
        pass_comparison = pd.DataFrame({
            'Metric': ['My Pass Rate', 'Team Pass Rate'],
            'Pass Rate': [my_pass_rate, overall_pass_rate]
        })
        fig_pass, ax_pass = plt.subplots(figsize=(8, 5))
        pass_comparison.plot(x='Metric', y='Pass Rate', kind='bar', ax=ax_pass, color=['green', 'lightgreen'])
        ax_pass.set_ylabel("Pass Rate (%)")
        ax_pass.set_title("My Pass Rate vs Team Average")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_pass)
    
else:
    # Admin/All data view
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", len(filtered_df))
    with col2:
        avg_score = filtered_df["QA Score"].mean() if "QA Score" in filtered_df.columns else 0
        st.metric("Avg QA Score", f"{avg_score:.1f}%" if avg_score else "N/A")
    with col3:
        total_pass = filtered_df["Rubric Pass Count"].sum() if "Rubric Pass Count" in filtered_df.columns else 0
        total_fail = filtered_df["Rubric Fail Count"].sum() if "Rubric Fail Count" in filtered_df.columns else 0
        pass_rate = (total_pass / (total_pass + total_fail) * 100) if (total_pass + total_fail) > 0 else 0
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col4:
        st.metric("Unique Agents", filtered_df["Agent"].nunique())

# --- Historical Baseline Comparisons (Benchmarking) ---
with st.expander("📊 Historical Baseline Comparisons", expanded=False):
    st.markdown("### Compare Current Performance to Historical Baselines")
    
    # Calculate baselines - convert date objects to pandas Timestamp for consistent comparison
    start_date_dt = pd.Timestamp(start_date) if not isinstance(start_date, pd.Timestamp) else start_date
    end_date_dt = pd.Timestamp(end_date) if not isinstance(end_date, pd.Timestamp) else end_date
    
    # Calculate baselines
    baselines = calculate_historical_baselines(meta_df, start_date_dt, end_date_dt)
    
    if baselines:
        current_avg_score = filtered_df["QA Score"].mean() if "QA Score" in filtered_df.columns else None
        current_pass_rate = calculate_pass_rate(filtered_df)
        
        baseline_col1, baseline_col2, baseline_col3 = st.columns(3)
        
        # Last 30 days comparison
        if 'last_30_days' in baselines:
            with baseline_col1:
                baseline_30 = baselines['last_30_days']
                if current_avg_score and baseline_30['avg_score']:
                    score_change_30 = current_avg_score - baseline_30['avg_score']
                    st.metric(
                        "vs Last 30 Days",
                        f"{current_avg_score:.1f}%",
                        delta=f"{score_change_30:+.1f}%",
                        delta_color="normal" if score_change_30 >= 0 else "inverse",
                        help=f"Baseline: {baseline_30['avg_score']:.1f}%"
                    )
        
        # Last 90 days comparison
        if 'last_90_days' in baselines:
            with baseline_col2:
                baseline_90 = baselines['last_90_days']
                if current_avg_score and baseline_90['avg_score']:
                    score_change_90 = current_avg_score - baseline_90['avg_score']
                    st.metric(
                        "vs Last 90 Days",
                        f"{current_avg_score:.1f}%",
                        delta=f"{score_change_90:+.1f}%",
                        delta_color="normal" if score_change_90 >= 0 else "inverse",
                        help=f"Baseline: {baseline_90['avg_score']:.1f}%"
                    )
        
        # Year-over-year comparison
        if 'year_over_year' in baselines:
            with baseline_col3:
                baseline_yoy = baselines['year_over_year']
                if current_avg_score and baseline_yoy['avg_score']:
                    score_change_yoy = current_avg_score - baseline_yoy['avg_score']
                    st.metric(
                        "vs Same Period Last Year",
                        f"{current_avg_score:.1f}%",
                        delta=f"{score_change_yoy:+.1f}%",
                        delta_color="normal" if score_change_yoy >= 0 else "inverse",
                        help=f"Baseline: {baseline_yoy['avg_score']:.1f}%"
                    )
    else:
        st.info("ℹ️ Insufficient historical data for baseline comparisons")
    
    # Benchmark visualization chart
    if baselines and current_avg_score:
        st.markdown("### Benchmark Comparison Chart")
        baseline_names = []
        baseline_scores = []
        
        if 'last_30_days' in baselines and baselines['last_30_days']['avg_score']:
            baseline_names.append('Last 30 Days')
            baseline_scores.append(baselines['last_30_days']['avg_score'])
        
        if 'last_90_days' in baselines and baselines['last_90_days']['avg_score']:
            baseline_names.append('Last 90 Days')
            baseline_scores.append(baselines['last_90_days']['avg_score'])
        
        if 'year_over_year' in baselines and baselines['year_over_year']['avg_score']:
            baseline_names.append('Same Period Last Year')
            baseline_scores.append(baselines['year_over_year']['avg_score'])
        
        if baseline_names:
            baseline_names.append('Current Period')
            baseline_scores.append(current_avg_score)
            
            fig_bench, ax_bench = plt.subplots(figsize=(10, 6))
            colors = ['steelblue' if i < len(baseline_names) - 1 else 'orange' for i in range(len(baseline_names))]
            bars = ax_bench.bar(baseline_names, baseline_scores, color=colors)
            ax_bench.set_ylabel("Average QA Score (%)")
            ax_bench.set_title("Current Performance vs Historical Baselines")
            ax_bench.axhline(y=alert_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({alert_threshold}%)')
            ax_bench.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_bench)

# --- Agent Leaderboard ---
if not user_agent_id:
    # Admin view - show all agents
    st.subheader("🏆 Agent Leaderboard")
    agent_performance = filtered_df.groupby("Agent").agg(
            Total_Calls=("Call ID", "count"),
        Avg_QA_Score=("QA Score", "mean"),
        Total_Pass=("Rubric Pass Count", "sum"),
        Total_Fail=("Rubric Fail Count", "sum"),
        Avg_Call_Duration=("Call Duration (min)", "mean")
    ).reset_index()

    # Calculate pass rate
    agent_performance["Pass_Rate"] = (agent_performance["Total_Pass"] / (agent_performance["Total_Pass"] + agent_performance["Total_Fail"]) * 100).fillna(0)
    
    # Add percentile rankings
    percentile_rankings = calculate_percentile_rankings(filtered_df, "QA Score")
    if not percentile_rankings.empty:
        agent_performance = agent_performance.merge(percentile_rankings[['Agent', 'percentile']], on='Agent', how='left')
        agent_performance['percentile'] = agent_performance['percentile'].fillna(0)
        
        # Add percentile badges
        def get_percentile_badge(pct):
            if pct >= 90:
                return "🏆 Top 10%"
            elif pct >= 75:
                return "🥇 Top 25%"
            elif pct >= 50:
                return "🥈 Top 50%"
            elif pct >= 25:
                return "🥉 Bottom 50%"
            else:
                return "📉 Bottom 25%"
        
        agent_performance['Percentile_Rank'] = agent_performance['percentile'].apply(get_percentile_badge)
        agent_performance = agent_performance.sort_values("Avg_QA_Score", ascending=False)
        
        # Display with percentile column
        display_cols = ['Agent', 'Total_Calls', 'Avg_QA_Score', 'Pass_Rate', 'Percentile_Rank', 'Avg_Call_Duration']
        st.dataframe(agent_performance[display_cols].round(1), hide_index=True)
    else:
        agent_performance = agent_performance.sort_values("Avg_QA_Score", ascending=False)
        st.dataframe(agent_performance.round(1), hide_index=True)
else:
    # Agent view - show only their performance summary
    st.subheader("📊 My Performance Summary")
    my_performance = filtered_df.groupby("Agent").agg(
        Total_Calls=("Call ID", "count"),
        Avg_QA_Score=("QA Score", "mean"),
        Total_Pass=("Rubric Pass Count", "sum"),
        Total_Fail=("Rubric Fail Count", "sum"),
        Avg_Call_Duration=("Call Duration (min)", "mean")
    ).reset_index()
    
    my_performance["Pass_Rate"] = (my_performance["Total_Pass"] / (my_performance["Total_Pass"] + my_performance["Total_Fail"]) * 100).fillna(0)
    
    # Create comparison table
    comparison_table = pd.DataFrame({
        'Metric': ['Total Calls', 'Avg QA Score', 'Pass Rate', 'Avg Call Duration (min)'],
        'My Performance': [
            str(int(my_performance["Total_Calls"].iloc[0])),
            f"{my_performance['Avg_QA_Score'].iloc[0]:.1f}%",
            f"{my_performance['Pass_Rate'].iloc[0]:.1f}%",
            f"{my_performance['Avg_Call_Duration'].iloc[0]:.1f}" if not pd.isna(my_performance['Avg_Call_Duration'].iloc[0]) else "N/A"
        ],
        'Team Average': [
            str(overall_total_calls) if overall_total_calls else "0",
            f"{overall_avg_score:.1f}%" if overall_avg_score else "0.0%",
            f"{overall_pass_rate:.1f}%" if overall_pass_rate else "0.0%",
            "N/A"  # Could calculate if needed
        ]
    })
    
    st.dataframe(comparison_table, hide_index=True)

# --- Performance Alerts ---
st.subheader("⚠️ Performance Alerts")
alerts_df = filtered_df[filtered_df["QA Score"] < alert_threshold] if "QA Score" in filtered_df.columns else pd.DataFrame()
if len(alerts_df) > 0:
    st.warning(f"⚠️ {len(alerts_df)} call(s) below threshold ({alert_threshold}%)")
    alert_summary = alerts_df.groupby("Agent").agg(
        Low_Score_Calls=("Call ID", "count"),
        Avg_Score=("QA Score", "mean")
    ).reset_index().sort_values("Low_Score_Calls", ascending=False)
    st.dataframe(alert_summary)
else:
    st.success(f"✅ All calls meet the threshold ({alert_threshold}%)")

# --- At-Risk Agent Detection (Predictive Analytics) ---
if not user_agent_id:  # Admin view only
    with st.expander("🔮 At-Risk Agent Detection", expanded=False):
        st.markdown("### Early Warning System for Agents at Risk")
        st.caption("Identifies agents who may drop below threshold based on recent trends, volatility, and proximity to threshold")
        
        at_risk_agents = identify_at_risk_agents(filtered_df, threshold=alert_threshold)
        
        if at_risk_agents:
            st.warning(f"⚠️ Found {len(at_risk_agents)} agent(s) at risk")
            
            risk_data = []
            for agent_info in at_risk_agents:
                risk_data.append({
                    'Agent': agent_info['agent'],
                    'Risk Score': f"{agent_info['risk_score']:.0f}/100",
                    'Recent Avg Score': f"{agent_info['recent_avg']:.1f}%",
                    'Trend': "📉 Declining" if agent_info['trend_slope'] < 0 else "📈 Improving",
                    'Volatility': f"{agent_info['volatility']:.1f}",
                    'Distance to Threshold': f"{agent_info['proximity_to_threshold']:.1f}%",
                    'Recent Calls': agent_info['recent_calls']
                })
            
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, hide_index=True)
            
            # Show risk factors for top at-risk agent
            if at_risk_agents:
                top_risk = at_risk_agents[0]
                st.markdown(f"**Why is {top_risk['agent']} at risk?**")
                risk_factors = []
                if top_risk['trend_slope'] < -0.5:
                    risk_factors.append(f"📉 Declining trend (slope: {top_risk['trend_slope']:.2f})")
                if top_risk['volatility'] > 10:
                    risk_factors.append(f"📊 High volatility ({top_risk['volatility']:.1f})")
                if top_risk['proximity_to_threshold'] <= 10:
                    risk_factors.append(f"⚠️ Close to threshold ({top_risk['proximity_to_threshold']:.1f}% away)")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"- {factor}")
        else:
            st.success("✅ No agents currently identified as at risk")

# --- QA Score Trends Over Time ---
st.subheader("📈 QA Score Trends Over Time")
col_trend1, col_trend2 = st.columns(2)

with col_trend1:
    if len(filtered_df) > 0 and "QA Score" in filtered_df.columns:
        # Daily average QA scores
        daily_scores = filtered_df.groupby(filtered_df["Call Date"].dt.date)["QA Score"].mean().reset_index()
        daily_scores.columns = ["Date", "Avg QA Score"]
        
        fig_trend, ax_trend = plt.subplots(figsize=(10, 5))
        ax_trend.plot(daily_scores["Date"], daily_scores["Avg QA Score"], marker="o", linewidth=2, color="steelblue")
        ax_trend.set_xlabel("Date")
        ax_trend.set_ylabel("Average QA Score (%)")
        ax_trend.set_title("QA Score Trend Over Time")
        ax_trend.grid(True, alpha=0.3)
        ax_trend.axhline(y=alert_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({alert_threshold}%)')
        ax_trend.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_trend)

with col_trend2:
    # Pass/Fail Rate Trends
    st.write("**Pass/Fail Rate Trends**")
    if len(filtered_df) > 0:
        daily_stats = filtered_df.groupby(filtered_df["Call Date"].dt.date).agg(
            Total_Pass=("Rubric Pass Count", "sum"),
            Total_Fail=("Rubric Fail Count", "sum")
        ).reset_index()
        daily_stats.columns = ["Date", "Total_Pass", "Total_Fail"]
        daily_stats["Total"] = daily_stats["Total_Pass"] + daily_stats["Total_Fail"]
        daily_stats["Pass_Rate"] = (daily_stats["Total_Pass"] / daily_stats["Total"] * 100).fillna(0)
        daily_stats["Fail_Rate"] = (daily_stats["Total_Fail"] / daily_stats["Total"] * 100).fillna(0)
        
        fig_pf, ax_pf = plt.subplots(figsize=(10, 5))
        ax_pf.plot(daily_stats["Date"], daily_stats["Pass_Rate"], marker="o", linewidth=2, label="Pass Rate", color="green")
        ax_pf.plot(daily_stats["Date"], daily_stats["Fail_Rate"], marker="s", linewidth=2, label="Fail Rate", color="red")
        ax_pf.set_xlabel("Date")
        ax_pf.set_ylabel("Rate (%)")
        ax_pf.set_title("Pass/Fail Rate Trends Over Time")
        ax_pf.grid(True, alpha=0.3)
        ax_pf.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_pf)

# --- Trend Forecasting (Predictive Analytics) ---
with st.expander("🔮 Trend Forecasting", expanded=False):
    st.markdown("### Predict Future QA Scores")
    st.caption("Forecasts future QA scores based on historical trends using time series analysis")
    
    forecast_days = st.selectbox("Forecast Period", [7, 14, 30], index=0, help="Number of days to forecast ahead")
    
    if len(filtered_df) > 0 and "QA Score" in filtered_df.columns and "Call Date" in filtered_df.columns:
        with st.spinner("Calculating forecast..."):
            forecast_result = predict_future_scores(filtered_df, days_ahead=forecast_days)
        
        if forecast_result:
            # Display forecast
            forecast_df = pd.DataFrame({
                'Date': forecast_result['dates'],
                'Forecast': forecast_result['forecast'],
                'Lower Bound': forecast_result['lower_bound'],
                'Upper Bound': forecast_result['upper_bound']
            })
            
            # Create forecast chart
            fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            daily_scores = filtered_df.groupby(filtered_df["Call Date"].dt.date)["QA Score"].mean().reset_index()
            daily_scores.columns = ["Date", "Avg QA Score"]
            ax_forecast.plot(daily_scores["Date"], daily_scores["Avg QA Score"], 
                           marker="o", linewidth=2, color="steelblue", label="Historical")
            
            # Plot forecast
            forecast_dates = pd.to_datetime(forecast_df['Date'])
            ax_forecast.plot(forecast_dates, forecast_df['Forecast'], 
                           marker="s", linewidth=2, color="orange", label="Forecast")
            ax_forecast.fill_between(forecast_dates, forecast_df['Lower Bound'], forecast_df['Upper Bound'],
                                   alpha=0.3, color="orange", label="95% Confidence Interval")
            
            ax_forecast.axhline(y=alert_threshold, color='r', linestyle='--', alpha=0.5, 
                              label=f'Threshold ({alert_threshold}%)')
            ax_forecast.set_xlabel("Date")
            ax_forecast.set_ylabel("Average QA Score (%)")
            ax_forecast.set_title(f"QA Score Forecast ({forecast_days} days ahead) - {forecast_result['method'].title()} Method")
            ax_forecast.grid(True, alpha=0.3)
            ax_forecast.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_forecast)
            
            # Show forecast summary
            avg_forecast = forecast_df['Forecast'].mean()
            st.metric("Average Forecasted Score", f"{avg_forecast:.1f}%", 
                     delta=f"{avg_forecast - daily_scores['Avg QA Score'].iloc[-1]:+.1f}%",
                     delta_color="normal" if avg_forecast >= daily_scores['Avg QA Score'].iloc[-1] else "inverse",
                     help="Average of forecasted scores vs most recent historical score")
        else:
            st.warning("⚠️ Insufficient data for forecasting (need at least 7 days of historical data)")
    else:
        st.info("ℹ️ No data available for forecasting")

# --- Rubric Code Analysis ---
st.subheader("🔍 Rubric Code Analysis")
if "Rubric Details" in filtered_df.columns:
    # Collect all rubric code statistics
    code_stats = {}
    for idx, row in filtered_df.iterrows():
        rubric_details = row.get("Rubric Details", {})
        if isinstance(rubric_details, dict):
            for code, details in rubric_details.items():
                if isinstance(details, dict):
                    status = details.get('status', 'N/A')
                    note = details.get('note', '')
                    
                    if code not in code_stats:
                        code_stats[code] = {'total': 0, 'pass': 0, 'fail': 0, 'na': 0, 'fail_notes': []}
                    
                    code_stats[code]['total'] += 1
                    if status == 'Pass':
                        code_stats[code]['pass'] += 1
                    elif status == 'Fail':
                        code_stats[code]['fail'] += 1
                        if note:
                            code_stats[code]['fail_notes'].append(note)
                    elif status == 'N/A':
                        code_stats[code]['na'] += 1
    
    if code_stats:
        rubric_analysis = pd.DataFrame([
            {
                'Code': code,
                'Total': stats['total'],
                'Pass': stats['pass'],
                'Fail': stats['fail'],
                'N/A': stats['na'],
                'Pass_Rate': (stats['pass'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                'Fail_Rate': (stats['fail'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                'Most_Common_Fail_Reason': max(set(stats['fail_notes']), key=stats['fail_notes'].count) if stats['fail_notes'] else 'N/A'
            }
            for code, stats in code_stats.items()
        ]).sort_values('Fail_Rate', ascending=False)
        
        st.dataframe(rubric_analysis.round(1))
        
        # Top failing codes chart
        col_rub1, col_rub2 = st.columns(2)
        with col_rub1:
            st.write("**Top 10 Failing Rubric Codes**")
            top_fail = rubric_analysis[rubric_analysis['Fail'] > 0].head(10)
            if len(top_fail) > 0:
                fig_fail, ax_fail = plt.subplots(figsize=(10, 6))
                top_fail.plot(x='Code', y='Fail_Rate', kind='barh', ax=ax_fail, color='red')
                ax_fail.set_xlabel("Fail Rate (%)")
                ax_fail.set_ylabel("Rubric Code")
                ax_fail.set_title("Top 10 Failing Rubric Codes")
                plt.tight_layout()
                st.pyplot(fig_fail)
        
        with col_rub2:
            # Rubric Code Heatmap (simplified - showing pass/fail rates)
            st.write("**Rubric Code Performance Heatmap**")
            # Group codes by major category (e.g., 1.x.x, 2.x.x)
            rubric_analysis['Category'] = rubric_analysis['Code'].str.split('.').str[0]
            category_stats = rubric_analysis.groupby('Category').agg(
                Avg_Pass_Rate=('Pass_Rate', 'mean'),
                Avg_Fail_Rate=('Fail_Rate', 'mean'),
                Total_Fails=('Fail', 'sum')
            ).reset_index().sort_values('Avg_Fail_Rate', ascending=False)
            
            fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
            colors = ['green' if x < 20 else 'orange' if x < 40 else 'red' for x in category_stats['Avg_Fail_Rate']]
            category_stats.plot(x='Category', y='Avg_Fail_Rate', kind='bar', ax=ax_heat, color=colors)
            ax_heat.set_ylabel("Average Fail Rate (%)")
            ax_heat.set_xlabel("Rubric Category")
            ax_heat.set_title("Fail Rate by Rubric Category")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_heat)

# --- Agent-Specific Trends ---
if user_agent_id:
    # Agent view - show their trend with team comparison
    st.subheader("📈 My Performance Trend vs Team Average")
    
    # Add trajectory analysis
    agent_data = filtered_df[filtered_df["Agent"] == user_agent_id]
    if len(agent_data) > 0:
        trajectory = classify_trajectory(filtered_df, agent=user_agent_id)
        
        traj_col1, traj_col2, traj_col3, traj_col4 = st.columns(4)
        with traj_col1:
            traj_icon = {
                'improving': '📈',
                'declining': '📉',
                'stable': '➡️',
                'volatile': '📊',
                'insufficient_data': '❓'
            }.get(trajectory['trajectory'], '❓')
            traj_label = {
                'improving': 'Improving',
                'declining': 'Declining',
                'stable': 'Stable',
                'volatile': 'Volatile',
                'insufficient_data': 'Insufficient Data'
            }.get(trajectory['trajectory'], 'Unknown')
            st.metric("Trajectory", f"{traj_icon} {traj_label}")
        
        with traj_col2:
            st.metric("Current Score", f"{trajectory.get('current_score', 0):.1f}%")
        
        with traj_col3:
            projected = trajectory.get('projected_score', 0)
            delta = projected - trajectory.get('current_score', 0) if projected else 0
            st.metric("Projected (7 days)", f"{projected:.1f}%", 
                     delta=f"{delta:+.1f}%",
                     delta_color="normal" if delta >= 0 else "inverse",
                     help="Projected score if current trend continues")
        
        with traj_col4:
            st.metric("Volatility", f"{trajectory.get('volatility', 0):.1f}",
                     help="Standard deviation of scores (lower is more consistent)")
    
    agent_trends_col1, agent_trends_col2 = st.columns(2)
    
    with agent_trends_col1:
        st.write("**My QA Score Trend**")
        if len(agent_data) > 0:
            agent_daily = agent_data.groupby(agent_data["Call Date"].dt.date).agg(
                Avg_QA_Score=("QA Score", "mean")
            ).reset_index()
            agent_daily.columns = ["Date", "My_Score"]
            
            # Get team average for same dates
            overall_daily = overall_df.groupby(overall_df["Call Date"].dt.date).agg(
                Avg_QA_Score=("QA Score", "mean")
            ).reset_index()
            overall_daily.columns = ["Date", "Team_Avg"]
            
            # Merge on date
            trend_comparison = pd.merge(agent_daily, overall_daily, on="Date", how="outer").sort_values("Date")
            
            fig_agent, ax_agent = plt.subplots(figsize=(10, 5))
            ax_agent.plot(trend_comparison["Date"], trend_comparison["My_Score"], marker="o", linewidth=2, label="My Score", color="steelblue")
            ax_agent.plot(trend_comparison["Date"], trend_comparison["Team_Avg"], marker="s", linewidth=2, label="Team Average", color="orange", linestyle="--")
            ax_agent.set_xlabel("Date")
            ax_agent.set_ylabel("Average QA Score (%)")
            ax_agent.set_title(f"My Performance Trend vs Team Average")
            ax_agent.grid(True, alpha=0.3)
            ax_agent.axhline(y=alert_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({alert_threshold}%)')
            ax_agent.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_agent)
    
    with agent_trends_col2:
        st.write("**My Pass Rate Trend vs Team**")
        if len(agent_data) > 0:
            agent_pass_daily = agent_data.groupby(agent_data["Call Date"].dt.date).agg(
                Total_Pass=("Rubric Pass Count", "sum"),
                Total_Fail=("Rubric Fail Count", "sum")
            ).reset_index()
            agent_pass_daily["Total"] = agent_pass_daily["Total_Pass"] + agent_pass_daily["Total_Fail"]
            agent_pass_daily["My_Pass_Rate"] = (agent_pass_daily["Total_Pass"] / agent_pass_daily["Total"] * 100).fillna(0)
            
            team_pass_daily = overall_df.groupby(overall_df["Call Date"].dt.date).agg(
                Total_Pass=("Rubric Pass Count", "sum"),
                Total_Fail=("Rubric Fail Count", "sum")
            ).reset_index()
            team_pass_daily["Total"] = team_pass_daily["Total_Pass"] + team_pass_daily["Total_Fail"]
            team_pass_daily["Team_Pass_Rate"] = (team_pass_daily["Total_Pass"] / team_pass_daily["Total"] * 100).fillna(0)
            
            pass_comparison = pd.merge(
                agent_pass_daily[["Call Date", "My_Pass_Rate"]],
                team_pass_daily[["Call Date", "Team_Pass_Rate"]],
                on="Call Date",
                how="outer"
            ).sort_values("Call Date")
            
            fig_pass_trend, ax_pass_trend = plt.subplots(figsize=(10, 5))
            ax_pass_trend.plot(pass_comparison["Call Date"], pass_comparison["My_Pass_Rate"], marker="o", linewidth=2, label="My Pass Rate", color="green")
            ax_pass_trend.plot(pass_comparison["Call Date"], pass_comparison["Team_Pass_Rate"], marker="s", linewidth=2, label="Team Pass Rate", color="lightgreen", linestyle="--")
            ax_pass_trend.set_xlabel("Date")
            ax_pass_trend.set_ylabel("Pass Rate (%)")
            ax_pass_trend.set_title("My Pass Rate Trend vs Team Average")
            ax_pass_trend.grid(True, alpha=0.3)
            ax_pass_trend.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_pass_trend)

else:
    # Admin view - agent selection and comparison
    st.subheader("👤 Agent-Specific Performance Trends")
    if len(filtered_df) > 0 and len(selected_agents) > 0:
        agent_trends_col1, agent_trends_col2 = st.columns(2)
        
        with agent_trends_col1:
            selected_agent_for_trend = st.selectbox("Select Agent for Trend Analysis", selected_agents)
            
            agent_data = filtered_df[filtered_df["Agent"] == selected_agent_for_trend]
            if len(agent_data) > 0:
                agent_daily = agent_data.groupby(agent_data["Call Date"].dt.date).agg(
                    Avg_QA_Score=("QA Score", "mean"),
                    Call_Count=("Call ID", "count")
                ).reset_index()
                agent_daily.columns = ["Date", "Avg_QA_Score", "Call_Count"]
                
                fig_agent, ax_agent = plt.subplots(figsize=(10, 5))
                ax_agent.plot(agent_daily["Date"], agent_daily["Avg_QA_Score"], marker="o", linewidth=2, label="QA Score")
                ax_agent.set_xlabel("Date")
                ax_agent.set_ylabel("Average QA Score (%)")
                ax_agent.set_title(f"Performance Trend: {selected_agent_for_trend}")
                ax_agent.grid(True, alpha=0.3)
                ax_agent.axhline(y=alert_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold')
                ax_agent.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_agent)
        
        with agent_trends_col2:
            # Agent Comparison
            st.write("**Agent Comparison**")
            compare_agents = st.multiselect("Select agents to compare", selected_agents, default=selected_agents[:min(3, len(selected_agents))])
            
            if len(compare_agents) > 0:
                compare_data = filtered_df[filtered_df["Agent"].isin(compare_agents)]
                agent_comparison = compare_data.groupby("Agent").agg(
                    Avg_QA_Score=("QA Score", "mean"),
                    Total_Calls=("Call ID", "count"),
                    Pass_Rate=("Rubric Pass Count", lambda x: (x.sum() / (x.sum() + compare_data.loc[x.index, "Rubric Fail Count"].sum()) * 100) if (x.sum() + compare_data.loc[x.index, "Rubric Fail Count"].sum()) > 0 else 0)
                ).reset_index()
                
                fig_compare, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # QA Score comparison
                agent_comparison.plot(x='Agent', y='Avg_QA_Score', kind='bar', ax=ax1, color='steelblue')
                ax1.set_ylabel("Avg QA Score (%)")
                ax1.set_title("Average QA Score Comparison")
                ax1.axhline(y=alert_threshold, color='r', linestyle='--', alpha=0.5)
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Pass Rate comparison
                agent_comparison.plot(x='Agent', y='Pass_Rate', kind='bar', ax=ax2, color='green')
                ax2.set_ylabel("Pass Rate (%)")
                ax2.set_title("Pass Rate Comparison")
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig_compare)

# --- QA Score Distribution and Label Distribution ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 QA Score Distribution")
    if "QA Score" in filtered_df.columns:
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        filtered_df["QA Score"].hist(bins=20, ax=ax_dist, edgecolor="black", color="steelblue")
        ax_dist.set_xlabel("QA Score (%)")
        ax_dist.set_ylabel("Number of Calls")
        ax_dist.set_title("Distribution of QA Scores")
        ax_dist.axvline(x=alert_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({alert_threshold}%)')
        ax_dist.legend()
        plt.tight_layout()
        st.pyplot(fig_dist)

with col_right:
    st.subheader("🏷️ Label Distribution")
    if "Label" in filtered_df.columns:
        label_counts = filtered_df["Label"].value_counts()
        fig_label, ax_label = plt.subplots(figsize=(8, 5))
        label_counts.plot(kind="bar", ax=ax_label, color="steelblue")
        ax_label.set_xlabel("Label")
        ax_label.set_ylabel("Number of Calls")
        ax_label.set_title("Call Labels Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_label)

# --- Coaching Insights Aggregation ---
st.subheader("💡 Coaching Insights")
if "Coaching Suggestions" in filtered_df.columns:
    all_coaching = []
    for idx, row in filtered_df.iterrows():
        coaching = row.get("Coaching Suggestions", [])
        if isinstance(coaching, list):
            all_coaching.extend(coaching)
        elif isinstance(coaching, str) and coaching:
            all_coaching.append(coaching)
    
    if all_coaching:
        from collections import Counter
        coaching_counts = Counter(all_coaching)
        top_coaching = pd.DataFrame(
            coaching_counts.most_common(10),
            columns=['Coaching Suggestion', 'Frequency']
        )
        
        col_coach1, col_coach2 = st.columns(2)
        with col_coach1:
            st.write("**Most Common Coaching Suggestions**")
            st.dataframe(top_coaching)
        
        with col_coach2:
            fig_coach, ax_coach = plt.subplots(figsize=(8, 6))
            top_coaching.plot(x='Coaching Suggestion', y='Frequency', kind='barh', ax=ax_coach, color='orange')
            ax_coach.set_xlabel("Frequency")
            ax_coach.set_title("Top 10 Coaching Suggestions")
            plt.tight_layout()
            st.pyplot(fig_coach)
    else:
        st.info("No coaching suggestions found in the filtered data.")

# --- Full Rubric Reference ---
st.subheader("📚 QA Rubric Reference")
if rubric_data:
    col_rubric_header1, col_rubric_header2 = st.columns([3, 1])
    with col_rubric_header1:
        st.info(f"📋 Complete rubric with {len(rubric_data)} items. Use the tabs below to browse by section or search all items.")
    with col_rubric_header2:
        # Load and serve the pre-formatted Excel rubric file
        try:
            import os
            rubric_excel_path = os.path.join(os.path.dirname(__file__), "Separatetab-rubric33.xlsx")
            if os.path.exists(rubric_excel_path):
                with open(rubric_excel_path, 'rb') as f:
                    rubric_excel_bytes = f.read()
                
                st.download_button(
                    label="📥 Download Rubric (Excel)",
                    data=rubric_excel_bytes,
                    file_name="QA_Rubric_v33.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ Rubric Excel file not found")
                st.info("Place 'Separatetab-rubric33.xlsx' in the dashboard directory")
        except Exception as e:
            st.error(f"Error loading rubric Excel: {e}")
    
    rubric_tab1, rubric_tab2 = st.tabs(["🔍 Search All Items", "📂 Browse by Section"])
    
    with rubric_tab1:
        # Search interface
        col_search1, col_search2 = st.columns([3, 1])
        with col_search1:
            rubric_search = st.text_input("🔍 Search rubric", placeholder="Enter code (e.g., 1.1.0), section, item, or criterion...", key="full_rubric_search")
        with col_search2:
            show_all = st.checkbox("Show all", value=not bool(rubric_search), key="show_all_rubric")
        
        if rubric_search and not show_all:
            search_lower = rubric_search.lower()
            filtered_items = [
                item for item in rubric_data
                if (search_lower in item.get('code', '').lower() or
                    search_lower in item.get('section', '').lower() or
                    search_lower in item.get('item', '').lower() or
                    search_lower in item.get('criterion', '').lower())
            ]
            st.write(f"**Found {len(filtered_items)} matching items**")
        else:
            filtered_items = rubric_data
            st.write(f"**All {len(rubric_data)} rubric items**")
        
        # Display filtered items with pagination
        items_per_page = 20
        if len(filtered_items) > items_per_page:
            total_pages = (len(filtered_items) - 1) // items_per_page + 1
            page_num = st.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=1, key="rubric_page")
            start_idx = (page_num - 1) * items_per_page
            end_idx = start_idx + items_per_page
            display_items = filtered_items[start_idx:end_idx]
            st.caption(f"Showing items {start_idx + 1}-{min(end_idx, len(filtered_items))} of {len(filtered_items)}")
        else:
            display_items = filtered_items
        
        # Display items
        for item in display_items:
            with st.expander(f"{item.get('code', 'N/A')} - {item.get('item', 'N/A')} | {item.get('section', 'N/A')} | Weight: {item.get('weight', 'N/A')}", expanded=False):
                st.write(f"**Section:** {item.get('section', 'N/A')}")
                st.write(f"**Item:** {item.get('item', 'N/A')}")
                st.write(f"**Criterion:** {item.get('criterion', 'N/A')}")
                st.write(f"**Weight:** {item.get('weight', 'N/A')}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**✅ Pass Criteria:**")
                    st.info(item.get('pass', 'N/A'))
                with col2:
                    st.markdown("**❌ Fail Criteria:**")
                    st.error(item.get('fail', 'N/A'))
                with col3:
                    st.markdown("**➖ N/A Criteria:**")
                    st.warning(item.get('na', 'N/A'))
                
                if item.get('agent_script_example'):
                    st.markdown("**💬 Agent Script Example:**")
                    st.code(item.get('agent_script_example'), language=None)
    
    with rubric_tab2:
        # Group by section
        sections = {}
        for item in rubric_data:
            section = item.get('section', 'Other')
            if section not in sections:
                sections[section] = []
            sections[section].append(item)
        
        selected_section = st.selectbox("Select Section", sorted(sections.keys()), key="rubric_section")
        
        if selected_section:
            section_items = sections[selected_section]
            st.write(f"**{len(section_items)} items in {selected_section}**")
            
            # Pagination for section items too
            if len(section_items) > items_per_page:
                section_total_pages = (len(section_items) - 1) // items_per_page + 1
                section_page_num = st.number_input(f"Page (1-{section_total_pages})", min_value=1, max_value=section_total_pages, value=1, key="section_page")
                section_start_idx = (section_page_num - 1) * items_per_page
                section_end_idx = section_start_idx + items_per_page
                display_section_items = section_items[section_start_idx:section_end_idx]
                st.caption(f"Showing items {section_start_idx + 1}-{min(section_end_idx, len(section_items))} of {len(section_items)}")
            else:
                display_section_items = section_items
            
            for item in display_section_items:
                with st.expander(f"{item.get('code', 'N/A')} - {item.get('item', 'N/A')} | Weight: {item.get('weight', 'N/A')}", expanded=False):
                    st.write(f"**Section:** {item.get('section', 'N/A')}")
                    st.write(f"**Item:** {item.get('item', 'N/A')}")
                    st.write(f"**Criterion:** {item.get('criterion', 'N/A')}")
                    st.write(f"**Weight:** {item.get('weight', 'N/A')}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**✅ Pass Criteria:**")
                        st.info(item.get('pass', 'N/A'))
                    with col2:
                        st.markdown("**❌ Fail Criteria:**")
                        st.error(item.get('fail', 'N/A'))
                    with col3:
                        st.markdown("**➖ N/A Criteria:**")
                        st.warning(item.get('na', 'N/A'))
                    
                    if item.get('agent_script_example'):
                        st.markdown("**💬 Agent Script Example:**")
                        st.code(item.get('agent_script_example'), language=None)
else:
    st.warning("⚠️ Rubric file not found. Please ensure 'Rubric_v33.json' is in the dashboard directory.")

st.markdown("---")

# --- Individual Call Details ---
st.subheader("📋 Individual Call Details")
if len(filtered_df) > 0:
    call_options = filtered_df["Call ID"].tolist()
    if call_options:
        # Call selection for export
        st.markdown("### Select Calls for Export")
        select_all_col1, select_all_col2 = st.columns([1, 4])
        with select_all_col1:
            if st.button("✅ Select All"):
                st.session_state.selected_call_ids = call_options.copy()
                st.rerun()
        with select_all_col2:
            if st.button("❌ Clear Selection"):
                st.session_state.selected_call_ids = []
                st.rerun()
        
        # Multi-select for calls
        if 'selected_call_ids' not in st.session_state:
            st.session_state.selected_call_ids = []
        
        # Filter out invalid default values (calls that no longer exist in options)
        # This prevents StreamlitAPIException when old selections are not in current options
        valid_defaults = [call_id for call_id in st.session_state.selected_call_ids if call_id in call_options]
        if len(valid_defaults) != len(st.session_state.selected_call_ids):
            removed_count = len(st.session_state.selected_call_ids) - len(valid_defaults)
            logger.debug(f"Removed {removed_count} invalid call IDs from selection defaults")
            st.session_state.selected_call_ids = valid_defaults
        
        selected_for_export = st.multiselect(
            "Choose calls to export (you can select multiple):",
            options=call_options,
            default=valid_defaults,
            format_func=lambda x: f"{x[:50]}... - {filtered_df[filtered_df['Call ID']==x]['QA Score'].iloc[0] if len(filtered_df[filtered_df['Call ID']==x]) > 0 and 'QA Score' in filtered_df.columns and not pd.isna(filtered_df[filtered_df['Call ID']==x]['QA Score'].iloc[0]) else 'N/A'}%"
        )
        st.session_state.selected_call_ids = selected_for_export
        
        if selected_for_export:
            st.info(f"📋 {len(selected_for_export)} call(s) selected for export")
        
        st.markdown("---")
        st.markdown("### View Call Details")
        selected_call_id = st.selectbox(
            "Select a call to view details:",
            options=call_options,
            format_func=lambda x: f"{x[:50]}... - {filtered_df[filtered_df['Call ID']==x]['QA Score'].iloc[0] if len(filtered_df[filtered_df['Call ID']==x]) > 0 and 'QA Score' in filtered_df.columns and not pd.isna(filtered_df[filtered_df['Call ID']==x]['QA Score'].iloc[0]) else 'N/A'}%"
        )
        
        if selected_call_id:
            call_details = filtered_df[filtered_df["Call ID"] == selected_call_id].iloc[0]
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.write("**Call Information**")
                st.write(f"**Call ID:** {call_details.get('Call ID', 'N/A')}")
                st.write(f"**Agent:** {call_details.get('Agent', 'N/A')}")
                st.write(f"**Date:** {call_details.get('Call Date', 'N/A')}")
                st.write(f"**Time:** {call_details.get('Call Time', 'N/A')}")
                qa_score = call_details.get('QA Score', 'N/A')
                st.write(f"**QA Score:** {qa_score}%" if isinstance(qa_score, (int, float)) else f"**QA Score:** {qa_score}")
                st.write(f"**Label:** {call_details.get('Label', 'N/A')}")
                call_dur = call_details.get('Call Duration (min)', 'N/A')
                if isinstance(call_dur, (int, float)):
                    st.write(f"**Call Length:** {call_dur:.2f} min")
                else:
                    st.write(f"**Call Length:** {call_dur}")
                
                st.write("**Reason:**")
                st.write(call_details.get('Reason', 'N/A'))
                
                st.write("**Outcome:**")
                st.write(call_details.get('Outcome', 'N/A'))
            
            with detail_col2:
                st.write("**Summary**")
                st.write(call_details.get('Summary', 'N/A'))
                
                st.write("**Strengths**")
                st.write(call_details.get('Strengths', 'N/A'))
                
                st.write("**Challenges**")
                st.write(call_details.get('Challenges', 'N/A'))
                
                st.write("**Coaching Suggestions**")
                coaching = call_details.get('Coaching Suggestions', [])
                if isinstance(coaching, list):
                    for suggestion in coaching:
                        st.write(f"- {suggestion}")
                else:
                    st.write(coaching if coaching else 'N/A')
            
            # Rubric Details
            st.write("**Rubric Details**")
            rubric_details = call_details.get('Rubric Details', {})
            if isinstance(rubric_details, dict) and rubric_details:
                rubric_df = pd.DataFrame([
                    {
                        'Code': code,
                        'Status': details.get('status', 'N/A') if isinstance(details, dict) else 'N/A',
                        'Note': details.get('note', '') if isinstance(details, dict) else ''
                    }
                    for code, details in rubric_details.items()
                ])
                st.dataframe(rubric_df)
                
                # Export individual call report
                st.markdown("---")
                call_dur_export = call_details.get('Call Duration (min)', 'N/A')
                if isinstance(call_dur_export, (int, float)):
                    call_dur_formatted = f"{call_dur_export:.2f}"
                else:
                    call_dur_formatted = call_dur_export
                
                report_text = f"""
# QA Call Report

## Call Information
- **Call ID:** {call_details.get('Call ID', 'N/A')}
- **Agent:** {call_details.get('Agent', 'N/A')}
- **Date:** {call_details.get('Call Date', 'N/A')}
- **Time:** {call_details.get('Call Time', 'N/A')}
- **QA Score:** {call_details.get('QA Score', 'N/A')}%
- **Label:** {call_details.get('Label', 'N/A')}
- **Call Length:** {call_dur_formatted} min

## Call Details
**Reason:** {call_details.get('Reason', 'N/A')}

**Outcome:** {call_details.get('Outcome', 'N/A')}

**Summary:**
{call_details.get('Summary', 'N/A')}

**Strengths:**
{call_details.get('Strengths', 'N/A')}

**Challenges:**
{call_details.get('Challenges', 'N/A')}

**Coaching Suggestions:**
{chr(10).join(['- ' + s for s in call_details.get('Coaching Suggestions', [])]) if isinstance(call_details.get('Coaching Suggestions'), list) else call_details.get('Coaching Suggestions', 'N/A')}

## Rubric Details
{chr(10).join([f"- {code}: {details.get('status', 'N/A')} - {details.get('note', '')}" for code, details in rubric_details.items() if isinstance(details, dict)])}

---
Generated by QA Dashboard • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                st.download_button(
                    label="📄 Export Call Report (TXT)",
                    data=report_text,
                    file_name=f"call_report_{call_details.get('Call ID', 'unknown')}.txt",
                    mime="text/plain"
                )
            else:
                st.write("No rubric details available")

# --- Call Volume Analysis ---
st.subheader("📞 Call Volume Analysis")
if len(filtered_df) > 0:
    vol_col1, vol_col2 = st.columns(2)
    
    with vol_col1:
        st.write("**Call Volume by Agent**")
        agent_volume = filtered_df.groupby("Agent").agg(
            Total_Calls=("Call ID", "count"),
            Avg_QA_Score=("QA Score", "mean")
        ).reset_index().sort_values("Total_Calls", ascending=False)
        
        fig_vol, ax_vol = plt.subplots(figsize=(10, 6))
        agent_volume.plot(x='Agent', y='Total_Calls', kind='bar', ax=ax_vol, color='steelblue')
        ax_vol.set_ylabel("Number of Calls")
        ax_vol.set_xlabel("Agent")
        ax_vol.set_title("Call Volume by Agent")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_vol)
    
    with vol_col2:
        st.write("**Call Volume Over Time**")
        daily_volume = filtered_df.groupby(filtered_df["Call Date"].dt.date).size().reset_index()
        daily_volume.columns = ["Date", "Call Count"]
        
        fig_vol_time, ax_vol_time = plt.subplots(figsize=(10, 5))
        ax_vol_time.plot(daily_volume["Date"], daily_volume["Call Count"], marker="o", linewidth=2, color="purple")
        ax_vol_time.set_xlabel("Date")
        ax_vol_time.set_ylabel("Number of Calls")
        ax_vol_time.set_title("Call Volume Trend Over Time")
        ax_vol_time.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_vol_time)

# --- Time of Day Analysis ---
st.subheader("⏰ Time of Day Analysis")
if "Call Time" in filtered_df.columns and len(filtered_df) > 0:
    time_col1, time_col2 = st.columns(2)
    
    with time_col1:
        st.write("**QA Score by Time of Day**")
        # Extract hour from time
        filtered_df["Hour"] = pd.to_datetime(filtered_df["Call Time"], format="%H:%M:%S", errors="coerce").dt.hour
        time_scores = filtered_df.groupby("Hour")["QA Score"].mean().reset_index()
        time_scores = time_scores.dropna()
        
        if len(time_scores) > 0:
            fig_time, ax_time = plt.subplots(figsize=(10, 5))
            ax_time.plot(time_scores["Hour"], time_scores["QA Score"], marker="o", linewidth=2, color="teal")
            ax_time.set_xlabel("Hour of Day")
            ax_time.set_ylabel("Average QA Score (%)")
            ax_time.set_title("QA Score by Time of Day")
            ax_time.grid(True, alpha=0.3)
            ax_time.set_xticks(range(0, 24, 2))
            plt.tight_layout()
            st.pyplot(fig_time)
    
    with time_col2:
        st.write("**Call Volume by Time of Day**")
        time_volume = filtered_df.groupby("Hour").size().reset_index()
        time_volume.columns = ["Hour", "Call Count"]
        time_volume = time_volume.dropna()
        
        if len(time_volume) > 0:
            fig_time_vol, ax_time_vol = plt.subplots(figsize=(10, 5))
            ax_time_vol.bar(time_volume["Hour"], time_volume["Call Count"], color="orange", alpha=0.7)
            ax_time_vol.set_xlabel("Hour of Day")
            ax_time_vol.set_ylabel("Number of Calls")
            ax_time_vol.set_title("Call Volume by Time of Day")
            ax_time_vol.set_xticks(range(0, 24, 2))
            plt.tight_layout()
            st.pyplot(fig_time_vol)

# --- Reason and Outcome Analysis ---
st.subheader("🎯 Call Reason & Outcome Analysis")
if "Reason" in filtered_df.columns or "Outcome" in filtered_df.columns:
    reason_col1, reason_col2 = st.columns(2)
    
    with reason_col1:
        if "Reason" in filtered_df.columns:
            st.write("**Most Common Call Reasons**")
            reason_counts = filtered_df["Reason"].value_counts().head(10)
            if len(reason_counts) > 0:
                fig_reason, ax_reason = plt.subplots(figsize=(8, 6))
                reason_counts.plot(kind="barh", ax=ax_reason, color="steelblue")
                ax_reason.set_xlabel("Number of Calls")
                ax_reason.set_title("Top 10 Call Reasons")
                plt.tight_layout()
                st.pyplot(fig_reason)
    
    with reason_col2:
        if "Outcome" in filtered_df.columns:
            st.write("**Most Common Outcomes**")
            outcome_counts = filtered_df["Outcome"].value_counts().head(10)
            if len(outcome_counts) > 0:
                fig_outcome, ax_outcome = plt.subplots(figsize=(8, 6))
                outcome_counts.plot(kind="barh", ax=ax_outcome, color="green")
                ax_outcome.set_xlabel("Number of Calls")
                ax_outcome.set_title("Top 10 Outcomes")
                plt.tight_layout()
                st.pyplot(fig_outcome)

# --- Anomaly Detection ---
st.markdown("---")
st.subheader("🚨 Anomaly Detection")

if "QA Score" in filtered_df.columns and "Call Date" in filtered_df.columns and len(filtered_df) > 1:
    # Detect anomalies: sudden score drops/spikes
    filtered_df_sorted = filtered_df.sort_values("Call Date")
    
    # Calculate rolling average (last 5 calls)
    filtered_df_sorted["Rolling_Avg"] = filtered_df_sorted["QA Score"].rolling(window=5, min_periods=1).mean()
    filtered_df_sorted["Score_Change"] = filtered_df_sorted["QA Score"] - filtered_df_sorted["Rolling_Avg"]
    
    # Define anomaly thresholds
    anomaly_threshold = 20  # 20 point deviation from rolling average
    anomalies = filtered_df_sorted[
        (filtered_df_sorted["Score_Change"].abs() > anomaly_threshold)
    ].copy()
    
    if len(anomalies) > 0:
        st.warning(f"⚠️ **{len(anomalies)} anomaly/anomalies detected:**")
        
        anomaly_col1, anomaly_col2 = st.columns(2)
        
        with anomaly_col1:
            st.write("**Score Drops (Sudden Decreases)**")
            drops = anomalies[anomalies["Score_Change"] < -anomaly_threshold].sort_values("Score_Change")
            if len(drops) > 0:
                drop_display = drops[["Call ID", "Agent", "Call Date", "QA Score", "Score_Change"]].head(10)
                drop_display["Score_Change"] = drop_display["Score_Change"].apply(lambda x: f"{x:.1f}%")
                st.dataframe(drop_display, hide_index=True)
            else:
                st.info("No significant score drops detected")
        
        with anomaly_col2:
            st.write("**Score Spikes (Sudden Increases)**")
            spikes = anomalies[anomalies["Score_Change"] > anomaly_threshold].sort_values("Score_Change", ascending=False)
            if len(spikes) > 0:
                spike_display = spikes[["Call ID", "Agent", "Call Date", "QA Score", "Score_Change"]].head(10)
                spike_display["Score_Change"] = spike_display["Score_Change"].apply(lambda x: f"+{x:.1f}%")
                st.dataframe(spike_display, hide_index=True)
            else:
                st.info("No significant score spikes detected")
        
        # Anomaly trend chart
        st.write("**Anomaly Timeline**")
        fig_anomaly, ax_anomaly = plt.subplots(figsize=(14, 6))
        ax_anomaly.plot(filtered_df_sorted["Call Date"], filtered_df_sorted["QA Score"], alpha=0.3, label="QA Score", color="gray")
        ax_anomaly.plot(filtered_df_sorted["Call Date"], filtered_df_sorted["Rolling_Avg"], label="Rolling Average", color="blue", linewidth=2)
        ax_anomaly.scatter(anomalies["Call Date"], anomalies["QA Score"], color="red", s=100, label="Anomalies", zorder=5)
        ax_anomaly.set_xlabel("Call Date")
        ax_anomaly.set_ylabel("QA Score (%)")
        ax_anomaly.set_title("QA Score with Anomaly Detection")
        ax_anomaly.legend()
        ax_anomaly.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_anomaly)
    else:
        st.success("✅ No anomalies detected in the filtered data")
else:
    st.info("ℹ️ Need at least 2 calls with QA scores to detect anomalies")

# --- Advanced Analytics ---
st.markdown("---")
st.subheader("📊 Advanced Analytics")

analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["📅 Week-over-Week Comparison", "📈 Agent Improvement Trends", "❌ Failure Analysis"])

with analytics_tab1:
    st.markdown("### Week-over-Week Performance Comparison")
    
    if "Call Date" in filtered_df.columns and "QA Score" in filtered_df.columns:
        # Group by week
        filtered_df["Week"] = pd.to_datetime(filtered_df["Call Date"]).dt.to_period("W").astype(str)
        weekly_stats = filtered_df.groupby("Week").agg({
            "QA Score": ["mean", "count"],
            "Rubric Pass Count": "sum",
            "Rubric Fail Count": "sum"
        }).reset_index()
        
        weekly_stats.columns = ["Week", "Avg_QA_Score", "Call_Count", "Total_Pass", "Total_Fail"]
        weekly_stats["Pass_Rate"] = (weekly_stats["Total_Pass"] / (weekly_stats["Total_Pass"] + weekly_stats["Total_Fail"]) * 100).fillna(0)
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
                ax_wow_score.plot(weekly_stats["Week"], weekly_stats["Avg_QA_Score"], marker="o", linewidth=2, label="Avg QA Score")
                ax_wow_score.set_xlabel("Week")
                ax_wow_score.set_ylabel("Average QA Score (%)")
                ax_wow_score.set_title("Week-over-Week QA Score Trend")
                ax_wow_score.grid(True, alpha=0.3)
                ax_wow_score.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_wow_score)
                
                # Show WoW changes
                st.write("**Week-over-Week Changes**")
                wow_display = weekly_stats[["Week", "Avg_QA_Score", "WoW_Score_Change", "Call_Count", "WoW_CallCount_Change"]].copy()
                wow_display["WoW_Score_Change"] = wow_display["WoW_Score_Change"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                wow_display["WoW_CallCount_Change"] = wow_display["WoW_CallCount_Change"].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "N/A")
                wow_display.columns = ["Week", "Avg QA Score", "WoW Change", "Call Count", "WoW Count Change"]
                st.dataframe(wow_display, hide_index=True)
            
            with wow_col2:
                st.write("**Pass Rate Week-over-Week**")
                fig_wow_pass, ax_wow_pass = plt.subplots(figsize=(12, 6))
                ax_wow_pass.plot(weekly_stats["Week"], weekly_stats["Pass_Rate"], marker="s", linewidth=2, color="green", label="Pass Rate")
                ax_wow_pass.set_xlabel("Week")
                ax_wow_pass.set_ylabel("Pass Rate (%)")
                ax_wow_pass.set_title("Week-over-Week Pass Rate Trend")
                ax_wow_pass.grid(True, alpha=0.3)
                ax_wow_pass.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_wow_pass)
        else:
            st.info("ℹ️ Need at least 2 weeks of data for week-over-week comparison")

with analytics_tab2:
    st.markdown("### Agent Improvement Trends")
    
    if "Agent" in filtered_df.columns and "Call Date" in filtered_df.columns and "QA Score" in filtered_df.columns:
        # Group by agent and week
        filtered_df["Week"] = pd.to_datetime(filtered_df["Call Date"]).dt.to_period("W").astype(str)
        agent_weekly = filtered_df.groupby(["Agent", "Week"]).agg({
            "QA Score": "mean",
            "Call ID": "count"
        }).reset_index()
        agent_weekly.columns = ["Agent", "Week", "Avg_QA_Score", "Call_Count"]
        
        # Calculate improvement (first week vs last week for each agent)
        agent_improvement = []
        for agent in agent_weekly["Agent"].unique():
            agent_data = agent_weekly[agent_weekly["Agent"] == agent].sort_values("Week")
            if len(agent_data) > 1:
                first_score = agent_data.iloc[0]["Avg_QA_Score"]
                last_score = agent_data.iloc[-1]["Avg_QA_Score"]
                improvement = last_score - first_score
                agent_improvement.append({
                    "Agent": agent,
                    "First Week Score": f"{first_score:.1f}%",
                    "Last Week Score": f"{last_score:.1f}%",
                    "Improvement": f"{improvement:+.1f}%",
                    "Trend": "📈 Improving" if improvement > 0 else "📉 Declining" if improvement < 0 else "➡️ Stable"
                })
        
        if agent_improvement:
            improvement_df = pd.DataFrame(agent_improvement)
            improvement_df = improvement_df.sort_values("Improvement", key=lambda x: x.str.replace('%', '').str.replace('+', '').astype(float), ascending=False)
            st.dataframe(improvement_df, hide_index=True)
            
            # Show trend chart for selected agents
            selected_agents_trend = st.multiselect(
                "Select agents to view trend:",
                options=filtered_df["Agent"].unique().tolist(),
                default=filtered_df["Agent"].unique().tolist()[:5] if len(filtered_df["Agent"].unique()) > 5 else filtered_df["Agent"].unique().tolist()
            )
            
            if selected_agents_trend:
                fig_agent_trend, ax_agent_trend = plt.subplots(figsize=(14, 6))
                for agent in selected_agents_trend:
                    agent_data = agent_weekly[agent_weekly["Agent"] == agent].sort_values("Week")
                    ax_agent_trend.plot(agent_data["Week"], agent_data["Avg_QA_Score"], marker="o", label=agent, linewidth=2)
                
                ax_agent_trend.set_xlabel("Week")
                ax_agent_trend.set_ylabel("Average QA Score (%)")
                ax_agent_trend.set_title("Agent Performance Trends Over Time")
                ax_agent_trend.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_agent_trend.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_agent_trend)
        else:
            st.info("ℹ️ Need multiple weeks of data per agent to show improvement trends")
    else:
        st.warning("⚠️ Missing required columns for agent improvement analysis")

with analytics_tab3:
    st.markdown("### Most Common Failure Reasons")
    
    if "Rubric Details" in filtered_df.columns:
        # Collect all failed rubric codes with their frequencies
        failure_reasons = {}
        for idx, row in filtered_df.iterrows():
            rubric_details = row.get("Rubric Details", {})
            if isinstance(rubric_details, dict):
                for code, details in rubric_details.items():
                    if isinstance(details, dict) and details.get('status', '').lower() == 'fail':
                        if code not in failure_reasons:
                            failure_reasons[code] = {
                                'count': 0,
                                'calls': set(),
                                'notes': []
                            }
                        failure_reasons[code]['count'] += 1
                        failure_reasons[code]['calls'].add(row.get('Call ID', ''))
                        note = details.get('note', '')
                        if note and note not in failure_reasons[code]['notes']:
                            failure_reasons[code]['notes'].append(note)
        
        if failure_reasons:
            # Sort by frequency
            sorted_failures = sorted(failure_reasons.items(), key=lambda x: x[1]['count'], reverse=True)
            
            failure_col1, failure_col2 = st.columns([2, 1])
            
            with failure_col1:
                st.write("**Top Failure Reasons**")
                failure_data = []
                for code, data in sorted_failures[:20]:  # Top 20
                    failure_data.append({
                        "Rubric Code": code,
                        "Failure Count": data['count'],
                        "Affected Calls": len(data['calls']),
                        "Sample Notes": data['notes'][0][:50] + "..." if data['notes'] else "N/A"
                    })
                
                failure_df = pd.DataFrame(failure_data)
                st.dataframe(failure_df, hide_index=True)
            
            with failure_col2:
                st.write("**Failure Distribution**")
                top_10_failures = sorted_failures[:10]
                codes = [item[0] for item in top_10_failures]
                counts = [item[1]['count'] for item in top_10_failures]
                
                fig_fail, ax_fail = plt.subplots(figsize=(8, 6))
                ax_fail.barh(range(len(codes)), counts, color="red", alpha=0.7)
                ax_fail.set_yticks(range(len(codes)))
                ax_fail.set_yticklabels(codes)
                ax_fail.set_xlabel("Failure Count")
                ax_fail.set_title("Top 10 Failure Reasons")
                plt.tight_layout()
                st.pyplot(fig_fail)
            
            # Show detailed view for selected failure code
            selected_failure_code = st.selectbox(
                "View details for failure code:",
                options=[code for code, _ in sorted_failures],
                help="Select a failure code to see detailed information"
            )
            
            if selected_failure_code:
                failure_info = failure_reasons[selected_failure_code]
                st.markdown(f"### Failure Code: {selected_failure_code}")
                st.metric("Total Failures", failure_info['count'])
                st.metric("Affected Calls", len(failure_info['calls']))
                
                if failure_info['notes']:
                    st.write("**Sample Failure Notes:**")
                    for note in failure_info['notes'][:5]:  # Show first 5 notes
                        st.text_area("Note", value=note, height=68, disabled=True, key=f"note_{hash(note)}", label_visibility="collapsed")
        else:
            st.info("ℹ️ No failed rubric items found in the filtered data")
    else:
        st.warning("⚠️ Rubric Details column not found")

# --- Export Options ---
st.markdown("---")
st.subheader("📥 Export Data")

# Export Templates
if 'export_templates' not in st.session_state:
    st.session_state.export_templates = {
        "Default (All Columns)": {"columns": "all", "format": "excel"},
        "Summary Only": {"columns": ["Call ID", "Agent", "Call Date", "QA Score", "Label"], "format": "excel"},
        "Detailed Report": {"columns": ["Call ID", "Agent", "Call Date", "QA Score", "Label", "Reason", "Outcome", "Summary"], "format": "excel"}
    }

with st.expander("📋 Export Templates", expanded=False):
    template_col1, template_col2 = st.columns(2)
    
    with template_col1:
        st.write("**Saved Templates:**")
        selected_template = st.selectbox(
            "Choose template:",
            options=list(st.session_state.export_templates.keys()),
            help="Select a template to customize or use as-is"
        )
    
    with template_col2:
        if st.button("➕ Save Current as Template"):
            template_name = st.text_input("Template name:", key="new_template_name")
            if template_name:
                # Get selected columns (will be set below)
                st.session_state.export_templates[template_name] = {
                    "columns": "all",  # Will be customized
                    "format": "excel"
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
            default=template.get("columns", available_columns) if isinstance(template.get("columns"), list) else available_columns,
            help="Choose which columns to include in export"
        )
        export_format = st.radio("Export format:", ["Excel", "CSV"], index=0 if template.get("format") == "excel" else 1)
        
        # Update template
        st.session_state.export_templates[selected_template]["columns"] = selected_columns
        st.session_state.export_templates[selected_template]["format"] = export_format.lower()

# Get selected template columns
selected_template_name = st.selectbox("Use template:", ["None"] + list(st.session_state.export_templates.keys()), key="export_template_select")
if selected_template_name != "None":
    template = st.session_state.export_templates[selected_template_name]
    template_columns = template.get("columns", "all")
    if template_columns == "all":
        export_df = filtered_df.copy()
    else:
        # Only include columns that exist
        available_template_cols = [col for col in template_columns if col in filtered_df.columns]
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
        # Recalculate agent performance for export
        agent_perf_export = filtered_df.groupby("Agent").agg(
            Total_Calls=("Call ID", "count"),
            Avg_QA_Score=("QA Score", "mean"),
            Total_Pass=("Rubric Pass Count", "sum"),
            Total_Fail=("Rubric Fail Count", "sum"),
            Avg_Call_Duration=("Call Duration (min)", "mean")
        ).reset_index()
        agent_perf_export["Pass_Rate"] = (agent_perf_export["Total_Pass"] / (agent_perf_export["Total_Pass"] + agent_perf_export["Total_Fail"]) * 100).fillna(0)
        agent_perf_export = agent_perf_export.sort_values("Avg_QA_Score", ascending=False)
        agent_perf_export.to_excel(writer, sheet_name="Agent Leaderboard", index=False)

# Export buttons
export_col1, export_col2 = st.columns(2)

with export_col1:
    # Track export generation for audit (download buttons don't have callbacks)
    export_key = f"export_excel_{start_date}_{end_date}_{len(export_df)}"
    if export_key not in st.session_state:
        if current_username and current_username.lower() in ["chloe", "shannon"]:
            log_audit_event(current_username, "export_data", f"Generated Excel export: {start_date} to {end_date}, {len(export_df)} rows")
        st.session_state[export_key] = True
    
    st.download_button(
        label="📥 Download QA Data (Excel)",
        data=excel_buffer.getvalue(),
        file_name=f"qa_report_{start_date}_to_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
        if current_username and current_username.lower() in ["chloe", "shannon"]:
            log_audit_event(current_username, "export_data", f"Generated CSV export: {start_date} to {end_date}, {len(export_df_csv)} rows")
        st.session_state[export_csv_key] = True
    
    st.download_button(
        label="📥 Download QA Data (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"qa_report_{start_date}_to_{end_date}.csv",
        mime="text/csv"
    )

# Export selected individual calls (if any are selected)
if 'selected_call_ids' not in st.session_state:
    st.session_state.selected_call_ids = []

if len(filtered_df) > 0:
    st.markdown("### Export Selected Calls")
    st.caption("Select calls from the 'Individual Call Details' section above, then export them here")
    
    # Show selected calls count
    if st.session_state.selected_call_ids:
        selected_calls_df = filtered_df[filtered_df['Call ID'].isin(st.session_state.selected_call_ids)]
        st.info(f"📋 {len(selected_calls_df)} call(s) selected for export")
        
        export_selected_col1, export_selected_col2 = st.columns(2)
        
        with export_selected_col1:
            # Excel export for selected calls
            selected_excel_buffer = io.BytesIO()
            with ExcelWriter(selected_excel_buffer, engine="xlsxwriter") as writer:
                selected_export_df = selected_calls_df.copy()
                for col in selected_export_df.columns:
                    selected_export_df[col] = selected_export_df[col].map(_clean)
                selected_export_df.to_excel(writer, sheet_name="Selected Calls", index=False)
            
            st.download_button(
                label="📥 Export Selected (Excel)",
                data=selected_excel_buffer.getvalue(),
                file_name=f"selected_calls_{start_date}_to_{end_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with export_selected_col2:
            # CSV export for selected calls
            selected_csv_buffer = io.StringIO()
            selected_export_df_csv = selected_calls_df.copy()
            for col in selected_export_df_csv.columns:
                selected_export_df_csv[col] = selected_export_df_csv[col].map(_clean_csv)
            selected_export_df_csv.to_csv(selected_csv_buffer, index=False)
            
            st.download_button(
                label="📥 Export Selected (CSV)",
                data=selected_csv_buffer.getvalue(),
                file_name=f"selected_calls_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )
        
        if st.button("🗑️ Clear Selection"):
            st.session_state.selected_call_ids = []
            st.rerun()
    else:
        st.caption("💡 No calls selected. Select calls from the 'Individual Call Details' section above.")

st.markdown("---")
st.markdown("Built with ❤️ by [Valence](https://www.getvalenceai.com) | QA Dashboard © 2025")
