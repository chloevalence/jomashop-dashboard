# Troubleshooting: App Stuck on Loading

## Issue: White Screen with Spinner

If your Streamlit app shows a white screen with a spinner and won't load, here's how to fix it:

## Quick Fixes

### 1. **Check the Terminal/Console**
The app logs errors to the terminal where you ran `streamlit run`. Look for:
- Error messages
- Stack traces
- "Processing PDFs" messages

**What to look for:**
```bash
# In your terminal where Streamlit is running
# Look for lines like:
ERROR: Failed to load data: ...
Exception: ...
```

### 2. **Wait a Bit Longer**
If you just uploaded many new PDFs, the first load can take 5-10 minutes depending on:
- Number of PDFs (100+ files = several minutes)
- PDF file sizes
- Network speed to S3

**Signs it's working:**
- Terminal shows "Processing PDFs: X/Y"
- Spinner is still spinning (not frozen)
- No error messages

### 3. **Clear the Cache**
If the cache got corrupted or stuck:

**Option A: Via UI (Admin Only)**
1. Wait for the app to load (or refresh)
2. Click "🔄 Reload ALL Data (Admin Only)" button
3. Wait for it to complete

**Option B: Via Terminal**
```bash
# Stop Streamlit (Ctrl+C)
# Clear Streamlit cache
rm -rf ~/.streamlit/cache
# Restart Streamlit
streamlit run streamlit_app_1_3_4.py
```

### 4. **Check S3 Connection**
Verify your S3 credentials are correct:

```bash
aws s3 ls s3://bpo-centers/ --recursive | grep "\.pdf$" | wc -l
```

This should show the number of PDFs. If it fails, your credentials are wrong.

### 5. **Check Logs**
Check the application logs:

```bash
# View today's log
tail -100 logs/app_$(date +%Y%m%d).log

# Or view all logs
ls -lt logs/app_*.log | head -1 | awk '{print $NF}' | xargs tail -100
```

### 6. **Limit Initial Load (Temporary Fix)**
If you have too many files, you can temporarily limit the load:

1. Edit `streamlit_app_1_3_4.py`
2. Find line ~823: `call_data, errors = load_all_calls_cached(max_files=None)`
3. Change to: `call_data, errors = load_all_calls_cached(max_files=100)`
4. Restart Streamlit
5. Once it loads, use "Refresh New Data" to load the rest

**⚠️ Note:** This is a temporary workaround. Remove the limit after initial load.

## Common Error Messages

### "Failed to load data: ..."
- **Cause:** S3 connection issue, timeout, or parsing error
- **Fix:** Check S3 credentials, network connection, or wait and retry

### "No data loaded"
- **Cause:** Cache was cleared but reload failed
- **Fix:** Click "🔄 Reload ALL Data" button

### "Access denied to S3 bucket"
- **Cause:** Wrong AWS credentials or IAM permissions
- **Fix:** Check `secrets.toml` S3 credentials

### "S3 bucket not found"
- **Cause:** Wrong bucket name
- **Fix:** Check `bucket_name` in `secrets.toml`

## Prevention

1. **Use "Refresh New Data" button** instead of full reload when possible
2. **Don't clear cache** unless necessary
3. **Monitor terminal** during first load to see progress
4. **Upload files in batches** rather than all at once

## Still Stuck?

1. **Check terminal output** - most errors appear there
2. **Check logs** - `logs/app_YYYYMMDD.log`
3. **Restart Streamlit** - sometimes helps clear stuck state
4. **Clear cache** - removes corrupted cached data

## Performance Tips

- **First load:** Can take 5-10 minutes for 100+ PDFs
- **Subsequent loads:** Should be instant (cached)
- **Refresh new data:** Only processes new files (much faster)
- **Full reload:** Only needed if cache is corrupted

