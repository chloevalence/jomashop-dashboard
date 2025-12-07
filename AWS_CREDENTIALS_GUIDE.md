# How to Find AWS S3 Credentials

This guide explains how to find your AWS Access Key ID, Secret Access Key, bucket name, region, and prefix.

## 1. AWS Access Key ID and Secret Access Key

### Option A: If you already have credentials
If you or your team already has AWS credentials:
- Check your password manager (1Password, LastPass, etc.)
- Ask your AWS administrator or DevOps team
- Look in your existing AWS configuration files:
  - `~/.aws/credentials` (on your local machine)
  - Environment variables
  - CI/CD pipeline secrets

### Option B: Create new credentials (if you have AWS Console access)

1. **Log in to AWS Console**
   - Go to https://console.aws.amazon.com
   - Sign in with your AWS account

2. **Navigate to IAM (Identity and Access Management)**
   - Search for "IAM" in the AWS services search bar
   - Click on "IAM"

3. **Go to Users**
   - Click "Users" in the left sidebar
   - Find your user (or the user you want to create credentials for)
   - Click on the username

4. **Create Access Key**
   - Click the "Security credentials" tab
   - Scroll down to "Access keys" section
   - Click "Create access key"
   - Choose "Application running outside AWS" (for Streamlit)
   - Click "Next"
   - Add a description (optional, e.g., "Streamlit Dashboard")
   - Click "Create access key"

5. **Copy Your Credentials**
   - **Access Key ID**: Copy this immediately (you'll see it once)
   - **Secret Access Key**: Click "Show" and copy this immediately
   - ⚠️ **IMPORTANT**: Save these securely! You won't be able to see the Secret Access Key again.

6. **Download the credentials file** (optional but recommended)
   - Click "Download .csv file" to save a backup

## 2. S3 Bucket Name

1. **Go to S3 in AWS Console**
   - Search for "S3" in AWS services
   - Click on "S3" or "Buckets"

2. **Find Your Bucket**
   - You'll see a list of all your S3 buckets
   - The bucket name is displayed in the list
   - Example: `my-call-reports-bucket` or `jomashop-dashboard-data`

3. **Copy the bucket name exactly as shown**

## 3. Region Name

1. **In the S3 Buckets page**
   - Look at the "Region" column next to your bucket name
   - Common regions: `us-east-1`, `us-west-2`, `eu-west-1`, etc.

2. **Or click on your bucket**
   - The region is shown in the bucket properties/details

## 4. Prefix (Optional)

The prefix is the folder path within your S3 bucket where the PDF files are stored.

### If PDFs are in the root of the bucket:
```
your-bucket/
  ├── file1.pdf
  ├── file2.pdf
```
**Prefix = `""` (empty string)** or omit the prefix line

### If PDFs are in a folder:
```
your-bucket/
  └── calls/
      ├── file1.pdf
      ├── file2.pdf
```
**Prefix = `"calls/"`**

### If PDFs are in nested folders:
```
your-bucket/
  └── reports/
      └── 2025/
          ├── file1.pdf
          ├── file2.pdf
```
**Prefix = `"reports/2025/"`**

### How to find the prefix:
1. Click on your S3 bucket in AWS Console
2. Navigate to where your PDF files are stored
3. Look at the folder path in the breadcrumb at the top
4. Copy that path (include the trailing `/` if it's a folder)

## 5. Update Your secrets.toml

Once you have all the information, update your `.streamlit/secrets.toml` file:

```toml
[s3]
aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"  # Your Access Key ID
aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"  # Your Secret Access Key
region_name = "us-east-1"  # Your bucket's region
bucket_name = "your-bucket-name"  # Your S3 bucket name
prefix = ""  # Folder path (empty if PDFs are in root, or "folder/" if in a subfolder)
```

## Security Best Practices

1. **Never commit secrets.toml to git**
   - Make sure `.streamlit/secrets.toml` is in your `.gitignore`
   - Use Streamlit Cloud secrets for production (not a file)

2. **Use IAM roles when possible**
   - For production, consider using IAM roles instead of access keys
   - Access keys are for development/testing

3. **Limit permissions**
   - Only grant `s3:ListBucket` and `s3:GetObject` permissions
   - Don't grant write or delete permissions unless needed

## Testing Your Credentials

After configuring, test with:

```bash
python3 -c "
import boto3
s3 = boto3.client('s3',
    aws_access_key_id='YOUR_KEY',
    aws_secret_access_key='YOUR_SECRET',
    region_name='us-east-1'
)
print('✅ Connection successful!')
print('Buckets:', [b['Name'] for b in s3.list_buckets()['Buckets']])
"
```

## Troubleshooting

### "Access Denied" error
- Check that your IAM user has S3 permissions
- Verify the bucket name is correct
- Ensure the region matches your bucket's region

### "Bucket not found" error
- Double-check the bucket name spelling
- Verify you're using the correct AWS account
- Check the region matches

### "No PDF files found"
- Verify the prefix path is correct
- Check that PDF files actually exist in that location
- Ensure the prefix ends with `/` if it's a folder path

