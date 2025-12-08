# AWS CLI Setup Guide

## ✅ Installation Complete

AWS CLI has been installed via Homebrew. Verify installation:

```bash
aws --version
```

## 🔧 Configuration

### 1. Configure AWS Credentials

You can use the same credentials from your `secrets.toml` file:

```bash
aws configure
```

You'll be prompted for:
- **AWS Access Key ID**: (from `secrets.toml` → `s3.aws_access_key_id`)
- **AWS Secret Access Key**: (from `secrets.toml` → `s3.aws_secret_access_key`)
- **Default region name**: (from `secrets.toml` → `s3.region_name`, e.g., `us-east-1`)
- **Default output format**: `json` (recommended)

**Alternative**: Set credentials via environment variables:
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 2. Verify Configuration

```bash
aws s3 ls
```

This should list your S3 buckets. If you see your bucket, you're configured correctly!

## 📤 Uploading PDFs to S3

### Upload a Single File

```bash
aws s3 cp /path/to/file.pdf s3://your-bucket-name/prefix/
```

**Example:**
```bash
aws s3 cp ~/Downloads/report.pdf s3://your-bucket-name/reports/
```

### Upload Multiple Files

```bash
aws s3 cp /path/to/directory/ s3://your-bucket-name/prefix/ --recursive
```

**Example:**
```bash
aws s3 cp ~/Downloads/qa-reports/ s3://your-bucket-name/reports/ --recursive
```

### Upload Only PDF Files

```bash
aws s3 sync /path/to/directory/ s3://your-bucket-name/prefix/ --exclude "*" --include "*.pdf"
```

### Upload with Progress

```bash
aws s3 cp /path/to/file.pdf s3://your-bucket-name/prefix/ --progress
```

## 🔍 Useful Commands

### List Files in Bucket

```bash
aws s3 ls s3://your-bucket-name/prefix/
```

### List All Files Recursively

```bash
aws s3 ls s3://your-bucket-name/prefix/ --recursive
```

### Count PDF Files

```bash
aws s3 ls s3://your-bucket-name/prefix/ --recursive | grep "\.pdf$" | wc -l
```

### Download a File

```bash
aws s3 cp s3://your-bucket-name/prefix/file.pdf ~/Downloads/
```

### Delete a File

```bash
aws s3 rm s3://your-bucket-name/prefix/file.pdf
```

### Sync Directory (Upload Only New/Changed Files)

```bash
aws s3 sync /local/directory/ s3://your-bucket-name/prefix/ --exclude "*" --include "*.pdf"
```

## 📋 Quick Reference

Replace these with your actual values from `secrets.toml`:
- `your-bucket-name` → Value from `s3.bucket_name`
- `prefix` → Value from `s3.prefix` (can be empty `""` if no prefix)
- `us-east-1` → Value from `s3.region_name`

## 🚀 Example Workflow

1. **Configure once:**
   ```bash
   aws configure
   ```

2. **Upload new PDFs:**
   ```bash
   aws s3 cp ~/Downloads/new-report.pdf s3://your-bucket-name/reports/
   ```

3. **Verify upload:**
   ```bash
   aws s3 ls s3://your-bucket-name/reports/ | grep new-report.pdf
   ```

4. **Refresh dashboard:**
   - Go to your Streamlit app
   - Click "🔄 Refresh New Data" button
   - New PDF will appear in the dashboard!

## 💡 Tips

- Use `--dryrun` to test commands without actually uploading:
  ```bash
  aws s3 cp file.pdf s3://bucket/prefix/ --dryrun
  ```

- Use `--exclude` and `--include` to filter files:
  ```bash
  aws s3 sync ./ s3://bucket/prefix/ --exclude "*" --include "*.pdf" --exclude "*test*"
  ```

- Monitor upload progress with `--progress` flag

- Use `aws s3 sync` instead of `cp` for incremental uploads (only new/changed files)

## 🔐 Security Note

Your AWS credentials are stored in:
- `~/.aws/credentials` (Linux/Mac)
- `%USERPROFILE%\.aws\credentials` (Windows)

Keep these files secure and never commit them to Git!

