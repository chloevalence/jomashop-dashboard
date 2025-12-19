"""
Quick test script to verify S3 connection and see what's happening.
Run this to debug S3 issues before running the full Streamlit app.
"""
import streamlit as st
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

st.set_page_config(page_title="S3 Connection Test", layout="wide")
st.title("🔍 S3 Connection Test")

try:
    # Load secrets
    st.write("### Step 1: Loading secrets...")
    aws_access_key_id = st.secrets["s3"]["aws_access_key_id"]
    aws_secret_access_key = st.secrets["s3"]["aws_secret_access_key"]
    region_name = st.secrets["s3"].get("region_name", "us-east-1")
    bucket_name = st.secrets["s3"]["bucket_name"]
    prefix = st.secrets["s3"].get("prefix", "")
    
    st.success("✅ Secrets loaded")
    st.write(f"- Region: {region_name}")
    st.write(f"- Bucket: {bucket_name}")
    st.write(f"- Prefix: '{prefix}' (empty if root)")
    
    # Initialize S3 client
    st.write("### Step 2: Initializing S3 client...")
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    st.success("✅ S3 client initialized")
    
    # Test connection
    st.write("### Step 3: Testing connection...")
    try:
        # List buckets to verify credentials
        buckets = s3_client.list_buckets()
        st.success("✅ AWS connection successful")
        st.write(f"Found {len(buckets['Buckets'])} bucket(s) in account")
    except Exception as e:
        st.error(f"❌ Failed to list buckets: {e}")
        st.stop()
    
    # Check if bucket exists
    st.write("### Step 4: Checking bucket access...")
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        st.success(f"✅ Bucket '{bucket_name}' exists and is accessible")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == '404':
            st.error(f"❌ Bucket '{bucket_name}' not found")
        elif error_code == '403':
            st.error(f"❌ Access denied to bucket '{bucket_name}'")
        else:
            st.error(f"❌ Error accessing bucket: {e}")
        st.stop()
    
    # List objects
    st.write("### Step 5: Listing PDF files...")
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        pdf_keys = []
        total_objects = 0
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    total_objects += 1
                    key = obj['Key']
                    if key.lower().endswith('.pdf'):
                        pdf_keys.append(key)
        
        st.success(f"✅ Found {len(pdf_keys)} PDF file(s) out of {total_objects} total object(s)")
        
        if pdf_keys:
            st.write("**PDF files found:**")
            for i, key in enumerate(pdf_keys[:10], 1):  # Show first 10
                st.write(f"{i}. {key}")
            if len(pdf_keys) > 10:
                st.write(f"... and {len(pdf_keys) - 10} more")
        else:
            st.warning("⚠️ No PDF files found!")
            if total_objects > 0:
                st.write(f"Found {total_objects} non-PDF object(s). Check your prefix path.")
            else:
                st.write("No objects found in this location. Check:")
                st.write("1. Bucket name is correct")
                st.write("2. Prefix path is correct (if PDFs are in a subfolder)")
                st.write("3. PDFs are actually uploaded to S3")
        
    except Exception as e:
        st.error(f"❌ Error listing objects: {e}")
        import traceback
        with st.expander("Show full error"):
            st.code(traceback.format_exc())
    
    st.write("---")
    st.success("🎉 S3 connection test complete!")
    st.write("If all steps passed, your Streamlit app should work.")

except KeyError as e:
    st.error(f"❌ Missing configuration in secrets: {e}")
    st.write("Please check your `.streamlit/secrets.toml` file.")
except NoCredentialsError:
    st.error("❌ AWS credentials not found")
    st.write("Please configure S3 credentials in secrets.")
except Exception as e:
    st.error(f"❌ Unexpected error: {e}")
    import traceback
    with st.expander("Show full error"):
        st.code(traceback.format_exc())

