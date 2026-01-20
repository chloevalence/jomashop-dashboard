# Testing the Dev Branch Locally

## Quick Start

1. **Ensure you're on the dev branch:**
   ```bash
   git checkout dev
   git pull origin dev
   ```

2. **Install/update dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Verify your configuration:**
   - Check that `.streamlit/secrets.toml` exists and has your S3 credentials
   - The app will use `MAX_DAYS_TO_LOAD = 30` to filter calls

4. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app_1_3_4.py
   ```

5. **Access the app:**
   - Open your browser to `http://localhost:8501`
   - Log in with your credentials

## What to Test

### Verify Date Filtering is Working

1. **Check the logs** - After the app loads, look for log messages like:
   ```
   Filtered calls from 33118 to 4558 (keeping last 30 days)
   ```
   or
   ```
   Filtered 33118 calls to 4558 calls (last 30 days only)
   ```

2. **Check memory usage** - The app should use much less memory:
   - Before: ~1.5GB (loading all 33,000+ calls)
   - After: ~100-200MB (loading only last 30 days)

3. **Verify data range** - In the dashboard:
   - Check the date range of calls shown
   - Should only show calls from the last 30 days
   - Check the "Call Date" column in any data tables

4. **Monitor the console** - Watch for:
   - "Memory usage before S3 cache load" messages
   - "Filtered X calls to Y calls" messages
   - Lower memory usage warnings

## Troubleshooting

### If filtering doesn't work:

1. **Check the date field in your data:**
   - The filter looks for: "Call Date", "call_date", "date", "timestamp", "Date"
   - Verify your call data has one of these fields

2. **Check the logs:**
   ```bash
   tail -f logs/app_$(date +%Y%m%d).log
   ```

3. **Verify MAX_DAYS_TO_LOAD is set:**
   - Should be `30` in `streamlit_app_1_3_4.py` around line 57

### If the app crashes:

1. **Check memory:**
   - Even with filtering, if you have many concurrent sessions, memory can still be high
   - Try closing other browser tabs/sessions

2. **Check S3 connection:**
   - Verify your S3 credentials in `.streamlit/secrets.toml`
   - Ensure the S3 bucket is accessible

3. **Check for errors in logs:**
   ```bash
   tail -50 logs/app_$(date +%Y%m%d).log
   ```

## Expected Behavior

- **On first load:** App loads S3 cache, filters to last 30 days, shows filtered data
- **Memory usage:** Should be ~100-200MB instead of ~1.5GB
- **Load time:** Should be faster since less data is processed
- **Data shown:** Only calls from the last 30 days should appear

## Changing the Date Range

To test with a different number of days, edit `streamlit_app_1_3_4.py`:

```python
MAX_DAYS_TO_LOAD = 7  # Last 7 days
# or
MAX_DAYS_TO_LOAD = None  # Load all data (no filtering)
```

Then restart the Streamlit app.
