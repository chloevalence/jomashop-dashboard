"""
Data loading functions for S3 PDF processing.

This module handles downloading and parsing PDF files from S3, including
retry logic and progress tracking.
"""

import boto3
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError
from concurrent.futures import ThreadPoolExecutor, as_completed

from pdf_parser import parse_pdf_from_bytes
from utils import track_error, logger


def process_pdf_with_retry(
    s3_client: Any,
    bucket_name: str,
    key: str,
    max_retries: int = 3
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Process a single PDF with retry logic for transient errors.
    
    Args:
        s3_client: Boto3 S3 client instance.
        bucket_name: S3 bucket name.
        key: S3 object key (file path).
        max_retries: Maximum number of retry attempts.
        
    Returns:
        Tuple of (parsed_data_dict, error_message). One will be None.
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Download PDF from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            pdf_bytes = response['Body'].read()
            
            # Extract filename from key
            filename = key.split('/')[-1]
            
            # Parse PDF
            parsed_data = parse_pdf_from_bytes(pdf_bytes, filename)
            
            if parsed_data:
                # Add S3 metadata
                parsed_data['_id'] = key
                parsed_data['_s3_key'] = key
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
                error_msg = f"{key}: {str(e)}"
                track_error("PDF_Processing", error_msg)
                return None, error_msg
    
    return None, f"{key}: {str(last_error)}"


def load_all_calls_internal(
    s3_client: Any,
    bucket_name: str,
    prefix: str,
    max_files: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Internal function to load PDF files from S3 bucket.
    
    Args:
        s3_client: Boto3 S3 client instance.
        bucket_name: S3 bucket name.
        prefix: S3 prefix/folder path.
        max_files: Maximum number of PDFs to load (None = load all).
        progress_callback: Optional callback function for progress updates.
        
    Returns:
        Tuple of (call_data_list, errors_list).
    """
    try:
        all_calls = []
        
        # List all PDF files in the S3 bucket
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=1000
            )
            
            # Collect all PDF file keys with their modification dates
            pdf_keys_with_dates = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith('.pdf'):
                            pdf_keys_with_dates.append({
                                'key': key,
                                'last_modified': obj.get('LastModified', datetime.min)
                            })
        except Exception as e:
            error_msg = f"Error listing S3 objects: {e}"
            track_error("S3_List", error_msg)
            return [], [error_msg]
        
        if not pdf_keys_with_dates:
            return [], ["No PDF files found in S3 bucket"]
        
        # Sort by modification date (most recent first)
        pdf_keys_with_dates.sort(key=lambda x: x['last_modified'], reverse=True)
        
        # Limit the number of files if specified
        if max_files and max_files > 0:
            pdf_keys_with_dates = pdf_keys_with_dates[:max_files]
        
        # Extract just the keys for processing
        pdf_keys = [item['key'] for item in pdf_keys_with_dates]
        total = len(pdf_keys)
        
        # Process PDFs in parallel
        errors = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_key = {
                executor.submit(process_pdf_with_retry, s3_client, bucket_name, key): key
                for key in pdf_keys
            }
            
            # Collect results as they complete
            processed = 0
            for future in as_completed(future_to_key):
                parsed_data, error = future.result()
                if parsed_data:
                    all_calls.append(parsed_data)
                elif error:
                    errors.append(error)
                
                processed += 1
                if progress_callback:
                    progress_callback(processed, total, len(errors))
        
        # Sort by call_date if available
        try:
            all_calls.sort(
                key=lambda x: x.get('call_date', datetime.min)
                if isinstance(x.get('call_date'), datetime) else datetime.min,
                reverse=True
            )
        except Exception:
            pass  # If sorting fails, just return unsorted
        
        return all_calls, errors
        
    except NoCredentialsError as e:
        error_msg = "AWS credentials not found. Please configure S3 credentials in secrets."
        track_error("S3_NoCredentials", str(e))
        return [], [error_msg]
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchBucket':
            error_msg = f"S3 bucket '{bucket_name}' not found."
            track_error(f"S3_{error_code}", error_msg)
            return [], [error_msg]
        elif error_code == 'AccessDenied':
            error_msg = f"Access denied to S3 bucket '{bucket_name}'. Check your credentials."
            track_error(f"S3_{error_code}", error_msg)
            return [], [error_msg]
        else:
            error_msg = f"S3 error: {e}"
            track_error(f"S3_{error_code}", error_msg)
            return [], [error_msg]
    except Exception as e:
        error_msg = f"Unexpected error loading from S3: {e}"
        track_error("S3_Unexpected", error_msg)
        logger.exception("Unexpected error in load_all_calls_internal")
        return [], [error_msg]

