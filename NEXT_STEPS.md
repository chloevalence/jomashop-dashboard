# Next Steps - Migration to S3

## Step 1: Install Dependencies

Install the new required packages:

```bash
pip install boto3 pdfplumber
```

Or if you're using a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Test the PDF Parser

Test the PDF parser with your sample PDF to see what data it extracts:

```bash
python3 test_pdf_parser.py "20250715_060053_TYIO-bpagent030844482%40nextiva.com-%2B14178141239-IN.pdf"
```

**What to look for:**
- Does it extract the date/time correctly from the filename?
- Does it find emotion counts (happy, angry, sad, neutral)?
- Does it extract the average happiness value?
- Does it find company and agent names?

If data is missing or incorrect, you'll need to adjust the regex patterns in `pdf_parser.py` based on your actual PDF format.

## Step 3: Configure S3 Credentials

Add S3 configuration to your Streamlit secrets. Create or edit `.streamlit/secrets.toml`:

```toml
[s3]
aws_access_key_id = "YOUR_AWS_ACCESS_KEY_ID"
aws_secret_access_key = "YOUR_AWS_SECRET_ACCESS_KEY"
region_name = "us-east-1"  # Change to your bucket's region
bucket_name = "your-bucket-name"
prefix = ""  # Optional: if PDFs are in a subfolder, e.g., "calls/" or "reports/"
```

**For Streamlit Cloud:**
- Go to your app settings
- Add secrets in the "Secrets" section
- Use the same format as above

## Step 4: Verify S3 Bucket Access

Make sure:
- Your S3 bucket exists and contains PDF files
- Your AWS credentials have permissions:
  - `s3:ListBucket` on the bucket
  - `s3:GetObject` on the bucket objects
- The bucket name in secrets matches your actual bucket

## Step 5: Run the Streamlit App

```bash
streamlit run streamlit_app_1_3_4.py
```

The app should now:
1. Connect to S3 instead of Firebase
2. List all PDF files in your bucket
3. Download and parse each PDF
4. Display the dashboard with your data

## Step 6: Adjust PDF Parser (If Needed)

If the parser doesn't extract all the data correctly:

1. **Check what text is in your PDF:**
   ```bash
   python3 -c "import pdfplumber; pdf = pdfplumber.open('20250715_060053_TYIO-bpagent030844482%40nextiva.com-%2B14178141239-IN.pdf'); print('\n'.join([page.extract_text() for page in pdf.pages]))"
   ```

2. **Update regex patterns in `pdf_parser.py`:**
   - Look at the `extract_data_from_text()` function
   - Adjust the regex patterns to match your PDF's actual text format
   - Common fields to check:
     - Emotion counts (happy, angry, sad, neutral)
     - Average happiness percentage
     - Company name
     - Agent name
     - Call duration
     - Low confidence values

## Troubleshooting

### "ModuleNotFoundError: No module named 'pdfplumber'"
- Run: `pip install pdfplumber boto3`

### "No PDF files found in S3 bucket"
- Check your bucket name in secrets
- Verify PDFs are actually in the bucket
- Check the `prefix` setting if PDFs are in a subfolder

### "Access denied to S3 bucket"
- Verify AWS credentials are correct
- Check IAM permissions (see S3_SETUP.md)

### Missing or incorrect data in dashboard
- Test the PDF parser first (Step 2)
- Adjust regex patterns in `pdf_parser.py` based on your PDF format
- Check the extracted PDF text to see what's actually in the file

## Need Help?

1. Test the PDF parser first to see what it extracts
2. If data is missing, examine the actual PDF text content
3. Update the regex patterns in `pdf_parser.py` to match your format
4. Re-test until all fields are extracted correctly

