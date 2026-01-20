# Crash Risk Analysis - streamlit_app_1_3_4.py

## Critical Issues (High Crash Risk)

### 1. DataFrame Indexing Without Empty Checks ⚠️ **CRITICAL**

**Location:** Lines 4654, 4886

**Issue:** Using `.iloc[0]` and `.iloc[-1]` without verifying the DataFrame has data.

```python
# Line 4654 - predict_future_scores_simple()
if scores.nunique() == 1:
    avg_score = scores.iloc[0]  # ❌ CRASH if scores is empty

# Line 4886 - classify_trajectory()
last_score = scores.iloc[-1]  # ❌ CRASH if scores is empty
```

**Risk:** `IndexError` if DataFrame is empty, even after length checks.

**Fix Required:**
```python
# Line 4654 - Add empty check
if scores.nunique() == 1 and len(scores) > 0:
    avg_score = scores.iloc[0]

# Line 4886 - Add empty check before accessing
if len(scores) > 0:
    last_score = scores.iloc[-1]
else:
    return {"trajectory": "insufficient_data", ...}
```

---

### 2. Threading Resource Leaks ⚠️ **HIGH**

**Location:** Lines 1851-1855, 5402-5404

**Issue:** Daemon threads with timeouts may leave S3 connections/resources open.

```python
# Line 1851-1855
s3_thread = threading.Thread(target=s3_operation, daemon=True)
s3_thread.start()
s3_thread.join(timeout=5)  # Thread may still be running after timeout

# Line 5402-5404
page_thread = threading.Thread(target=fetch_page, daemon=True)
page_thread.start()
page_thread.join(timeout=max_page_time)
```

**Risk:** 
- Daemon threads killed abruptly may not clean up resources
- S3 connections may remain open
- Memory leaks from unclosed connections

**Fix Required:**
- Use context managers for S3 operations
- Implement proper cleanup in thread functions
- Consider using `threading.Event` for graceful shutdown

---

### 3. S3 Secrets Access Without Validation ⚠️ **HIGH**

**Location:** Lines 560-565, 1303-1306, 1423-1428, and many others

**Issue:** Direct access to `st.secrets["s3"]` without checking if keys exist.

```python
# Line 560-565 - Module level initialization
s3_client = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],  # ❌ KeyError if missing
    aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],  # ❌ KeyError
    region_name=st.secrets["s3"].get("region_name", "us-east-1"),
)
```

**Risk:** `KeyError` crashes the app if secrets are misconfigured.

**Fix Required:**
```python
# Use .get() with defaults or validate first
if "s3" not in st.secrets:
    logger.error("S3 secrets not configured")
    return None, None

aws_access_key = st.secrets["s3"].get("aws_access_key_id")
if not aws_access_key:
    logger.error("Missing aws_access_key_id")
    return None, None
```

---

### 4. Exception Re-raising in Critical Paths ⚠️ **MEDIUM-HIGH**

**Location:** Lines 464, 5412

**Issue:** Catching exceptions and re-raising can crash the app.

```python
# Line 464 - st_pyplot_safe()
except Exception as e:
    # ... handle MediaFileStorageError ...
    else:
        logger.warning(f"Error displaying matplotlib figure: {e}")
        raise  # ❌ Re-raises exception, crashes app

# Line 5412 - Thread error handling
elif not error_queue.empty():
    page_error = error_queue.get()
    raise page_error  # ❌ Re-raises, crashes app
```

**Risk:** Unhandled exceptions propagate and crash the Streamlit app.

**Fix Required:**
- Wrap in try/except at higher level
- Return error indicators instead of raising
- Use Streamlit error handling (st.error) instead of crashing

---

## Medium Risk Issues

### 5. DataFrame Operations Without None Checks

**Location:** Multiple locations (lines 8471, 8488, 8505, 9108)

**Issue:** Using `.loc[]` operations without verifying DataFrame is not None/empty.

```python
# Line 8471
df.loc[mask, "Agent"] = df.loc[mask, "Agent"].astype(str).str.strip().replace(agent_mapping)
```

**Risk:** `AttributeError` if `df` is None, or `KeyError` if column missing.

**Fix:** Add defensive checks:
```python
if df is not None and not df.empty and "Agent" in df.columns:
    df.loc[mask, "Agent"] = ...
```

---

### 6. JSON Parsing Without Size Limits

**Location:** Lines 1843-1845, 3546-3548

**Issue:** Loading entire JSON files into memory without size validation.

```python
# Line 1843-1845
cached_data = json.loads(b"".join(chunks).decode("utf-8"))  # ❌ No size limit
```

**Risk:** Memory exhaustion with very large cache files.

**Fix:** Already partially addressed (line 1804-1811 checks size), but should be consistent everywhere.

---

### 7. File Lock Timeout Handling

**Location:** Lines 122-226 (cache_file_lock)

**Issue:** Lock timeout can raise `LockTimeoutError` which may not be caught everywhere.

**Risk:** Unhandled `LockTimeoutError` crashes operations.

**Fix:** Ensure all callers handle `LockTimeoutError` gracefully.

---

## Low Risk Issues (Edge Cases)

### 8. Matplotlib Figure Cleanup

**Location:** Lines 437-472 (st_pyplot_safe)

**Status:** ✅ **GOOD** - Proper cleanup with `plt.close("all")` and `gc.collect()`

**Note:** This is well-handled, but ensure it's used everywhere matplotlib is used.

---

### 9. Session State Access

**Location:** Multiple locations

**Status:** ✅ **GOOD** - Most accesses are protected with try/except blocks

**Note:** Continue using defensive patterns.

---

## Recommendations Summary

### Immediate Fixes (Critical):
1. ✅ Add empty DataFrame checks before `.iloc[0]` and `.iloc[-1]` (lines 4654, 4886)
2. ✅ Validate S3 secrets before accessing (lines 560-565)
3. ✅ Improve thread cleanup for S3 operations (lines 1851-1855, 5402-5404)
4. ✅ Replace exception re-raising with error handling (lines 464, 5412)

### Short-term Improvements:
5. Add None/empty checks for all DataFrame operations
6. Implement consistent error handling patterns
7. Add size limits for all JSON loading operations
8. Document exception handling strategy

### Long-term Improvements:
9. Refactor S3 client creation into a single, well-tested function
10. Add comprehensive unit tests for edge cases
11. Implement circuit breaker pattern for S3 operations
12. Add monitoring/alerting for repeated failures

---

## Testing Recommendations

Test these scenarios:
- Empty DataFrames in all analysis functions
- Missing S3 credentials
- S3 connection timeouts
- Corrupted cache files
- Very large cache files (>100MB)
- Concurrent file access
- Thread timeouts

---

## Code Quality Notes

**Good Practices Found:**
- ✅ File locking implementation
- ✅ Atomic JSON writes
- ✅ Matplotlib cleanup
- ✅ Defensive programming in many places
- ✅ Comprehensive error logging

**Areas for Improvement:**
- More consistent error handling patterns
- Better separation of concerns
- More defensive checks before operations
- Consistent use of helper functions for common operations
