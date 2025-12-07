# Quick Start - Run Your Dashboard

## Run Locally

```bash
streamlit run streamlit_app_1_3_4.py
```

This will:
1. Open your browser automatically
2. Connect to S3 using your credentials
3. Download and parse all PDFs from your bucket
4. Display the dashboard

## What to Expect

1. **Login Screen**: Use one of the credentials from your secrets.toml
2. **Loading**: The app will show "⏳ Loading call data from S3…"
3. **Success Message**: "✅ Loaded X calls in Y seconds"
4. **Dashboard**: All your charts and metrics

## Troubleshooting

### If you see "Access Denied" or "Bucket not found"
- Double-check your S3 credentials in secrets.toml
- Verify bucket name and region are correct
- Make sure your AWS user has S3 permissions

### If you see "No PDF files found"
- Check that PDFs exist in your S3 bucket
- Verify the `prefix` path is correct (empty if PDFs are in root)

### If parsing fails for some PDFs
- Check the console for error messages
- The app will continue with successfully parsed PDFs

## For Streamlit Cloud

If you've already updated secrets on Streamlit Cloud:
1. Push your code to GitHub
2. The app will automatically redeploy
3. It will use the secrets you configured in Streamlit Cloud

