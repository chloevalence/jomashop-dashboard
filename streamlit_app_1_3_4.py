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
from pdf_parser import parse_pdf_from_bytes

st.set_page_config(page_title="Emotion Dashboard", layout="wide")

# Initialize S3 client from secrets
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
        region_name=st.secrets["s3"].get("region_name", "us-east-1")
    )
    s3_bucket_name = st.secrets["s3"]["bucket_name"]
    s3_prefix = st.secrets["s3"].get("prefix", "")  # Optional prefix/folder path
except KeyError as e:
    st.error(f"❌ Missing S3 configuration in secrets: {e}")
    st.error("Please check your `.streamlit/secrets.toml` file and ensure all S3 fields are set.")
    st.stop()
except Exception as e:
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
            
            # Collect all PDF file keys
            pdf_keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.lower().endswith('.pdf'):
                            pdf_keys.append(key)
        except Exception as e:
            return [], f"Error listing S3 objects: {e}"
        
        if not pdf_keys:
            return [], "No PDF files found in S3 bucket"
        
        # Store total count before limiting
        total_pdfs = len(pdf_keys)
        
        # Limit the number of files if specified
        if max_files and max_files > 0:
            pdf_keys = pdf_keys[:max_files]
        
        # Download and parse each PDF
        errors = []
        total = len(pdf_keys)
        
        for i, key in enumerate(pdf_keys):
            try:
                # Download PDF from S3 (with timeout per file)
                response = s3_client_with_timeout.get_object(Bucket=s3_bucket_name, Key=key)
                pdf_bytes = response['Body'].read()
                
                # Extract filename from key
                filename = key.split('/')[-1]
                
                # Parse PDF
                parsed_data = parse_pdf_from_bytes(pdf_bytes, filename)
                
                if parsed_data:
                    # Add S3 metadata
                    parsed_data['_id'] = key
                    parsed_data['_s3_key'] = key
                    all_calls.append(parsed_data)
                else:
                    errors.append(f"Failed to parse {filename}")
                    
            except Exception as e:
                # Collect errors to show later
                errors.append(f"{key}: {str(e)}")
                continue
        
        # Sort by call_date if available
        try:
            all_calls.sort(key=lambda x: x.get('call_date', datetime.min) if isinstance(x.get('call_date'), datetime) else datetime.min)
        except:
            pass  # If sorting fails, just return unsorted
        
        # Return with info about total vs loaded
        if max_files and total_pdfs > max_files:
            return all_calls, (errors, f"Loaded {len(all_calls)} of {total_pdfs} PDF files (limited to {max_files})")
        else:
            return all_calls, errors
        
    except NoCredentialsError:
        return [], "AWS credentials not found. Please configure S3 credentials in secrets."
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchBucket':
            return [], f"S3 bucket '{s3_bucket_name}' not found."
        elif error_code == 'AccessDenied':
            return [], f"Access denied to S3 bucket '{s3_bucket_name}'. Check your credentials."
        else:
            return [], f"S3 error: {e}"
    except Exception as e:
        return [], f"Unexpected error loading from S3: {e}"

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_calls():
    """Cached wrapper for loading calls."""
    calls, errors = load_all_calls_internal()
    return calls

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
    authenticator.login("main", "Login")
    st.stop()

# If they submitted bad creds, show error and stay on login
if auth_status is False:
    st.error("❌ Username or password is incorrect")
    st.stop()

# Get current user info
current_username = st.session_state.get("username")
current_name = st.session_state.get("name")

# Check if user is an agent (has agent_id mapping) or admin
user_agent_id = None
is_admin = False

# Try to get agent_id from secrets - add this mapping to secrets.toml
try:
    user_mapping = st.secrets.get("user_mapping", {})
    if current_username and current_username in user_mapping:
        agent_id_value = user_mapping[current_username].get("agent_id", "")
        if agent_id_value:
            user_agent_id = agent_id_value
        else:
            # No agent_id means admin
            is_admin = True
    elif current_username in ["chloe", "shannon"]:  # Default admins
        is_admin = True
    else:
        # No mapping found - default to admin view for now
        # You can add mappings in secrets.toml to restrict access
        is_admin = True
except Exception as e:
    # If mapping doesn't exist, default to admin view
    is_admin = True

st.sidebar.success(f"Welcome, {current_name} 👋")

# Show view mode
if user_agent_id:
    st.sidebar.info(f"👤 Agent View: {user_agent_id}")
elif is_admin:
    st.sidebar.info("👑 Admin View: All Data")
else:
    st.sidebar.info("👤 User View: All Data")

if st.button("Refresh data"):
    st.cache_data.clear()
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
status_text = st.empty()
status_text.text("📋 Connecting to S3...")

try:
    # Quick connection test first
    import botocore.config
    config = botocore.config.Config(
        connect_timeout=5,
        read_timeout=10,
        retries={'max_attempts': 1}
    )
    test_client = boto3.client(
        's3',
        aws_access_key_id=st.secrets["s3"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["s3"]["aws_secret_access_key"],
        region_name=st.secrets["s3"].get("region_name", "us-east-1"),
        config=config
    )
    
    status_text.text("📋 Testing S3 connection...")
    # Quick test - just check if we can access the bucket
    test_client.head_bucket(Bucket=s3_bucket_name)
    status_text.text("✅ Connected! Listing PDF files...")
    
    # Quick count of PDFs
    try:
        paginator = test_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket_name, Prefix=s3_prefix, MaxKeys=100)
        pdf_count = sum(1 for page in pages for obj in page.get('Contents', []) if obj['Key'].lower().endswith('.pdf'))
        if pdf_count > 0:
            # Limit to first 20 files for initial load (can be increased later)
            max_files_to_load = 20
            if pdf_count > max_files_to_load:
                status_text.text(f"✅ Found {pdf_count} PDF file(s). Loading first {max_files_to_load} for testing...")
            else:
                status_text.text(f"✅ Found {pdf_count} PDF file(s). Loading...")
    except:
        pass  # If counting fails, just continue
    
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

# Now load the actual data
try:
    status_text.text("📥 Loading PDF files from S3...")
    
    # Limit to first 20 files for initial testing (increase this number later)
    max_files_to_load = 20
    
    t0 = time.time()
    
    # Load data - limited to max_files_to_load
    call_data, errors = load_all_calls_internal(max_files=max_files_to_load)
    
    elapsed = time.time() - t0
    status_text.empty()
    
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
        st.success(f"✅ Loaded {len(call_data)} calls in {elapsed:.2f}s")
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

# Convert call_date to datetime if it's not already
if "call_date" in meta_df.columns:
    # If call_date is already datetime, keep it; otherwise try to parse
    if meta_df["call_date"].dtype == 'object':
        meta_df["call_date"] = pd.to_datetime(meta_df["call_date"], errors="coerce")

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

# --- Determine if agent view or admin view ---
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

preset_option = st.sidebar.selectbox(
    "📆 Date Range",
    options=["All Time", "This Week", "Last 7 Days", "Last 30 Days", "Custom"],
    index=0
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
else:
    custom_input = st.sidebar.date_input("Select Date Range", value=(min(dates), max(dates)))
    if isinstance(custom_input, tuple) and len(custom_input) == 2:
        selected_dates = custom_input
    elif isinstance(custom_input, date):
        selected_dates = (custom_input, custom_input)
    else:
        st.warning("⚠️ Please select a valid date range.")
        st.stop()

start_date, end_date = selected_dates

# Agent filter (only for admin view)
if not user_agent_id:
    available_agents = filter_df["Agent"].dropna().unique().tolist()
    selected_agents = st.sidebar.multiselect("👤 Select Agents", available_agents, default=available_agents)
else:
    # For agents, they only see their own data
    selected_agents = [user_agent_id]

# QA Score filter
if "QA Score" in meta_df.columns and not meta_df["QA Score"].isna().all():
    min_score = float(meta_df["QA Score"].min())
    max_score = float(meta_df["QA Score"].max())
    score_range = st.sidebar.slider(
        "📊 QA Score Range",
        min_value=min_score,
        max_value=max_score,
        value=(min_score, max_score),
        step=1.0
    )
else:
    score_range = None

# Label filter
if "Label" in meta_df.columns:
    available_labels = meta_df["Label"].dropna().unique().tolist()
    selected_labels = st.sidebar.multiselect("🏷️ Select Labels", available_labels, default=available_labels)
else:
    selected_labels = []

# Search functionality
search_text = st.sidebar.text_input("🔍 Search (Reason/Summary/Outcome)", "")

# Rubric code filter - collect all failed codes
failed_rubric_codes = []
if "Rubric Details" in meta_df.columns:
    for idx, row in meta_df.iterrows():
        rubric_details = row.get("Rubric Details", {})
        if isinstance(rubric_details, dict):
            for code, details in rubric_details.items():
                if isinstance(details, dict) and details.get('status') == 'Fail':
                    if code not in failed_rubric_codes:
                        failed_rubric_codes.append(code)
    
    failed_rubric_codes.sort()
    if failed_rubric_codes:
        selected_failed_codes = st.sidebar.multiselect(
            "❌ Filter by Failed Rubric Codes",
            options=failed_rubric_codes,
            help="Show only calls that failed these specific rubric codes"
        )
    else:
        selected_failed_codes = []

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

# Apply filters
filtered_df = filter_df[
    (filter_df["Agent"].isin(selected_agents)) &
    (filter_df["Call Date"].dt.date >= start_date) &
    (filter_df["Call Date"].dt.date <= end_date)
].copy()

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

# --- Agent Leaderboard ---
if not user_agent_id:
    # Admin view - show all agents
    st.subheader("🏆 Agent Performance")
    agent_performance = filtered_df.groupby("Agent").agg(
        Total_Calls=("Call ID", "count"),
        Avg_QA_Score=("QA Score", "mean"),
        Total_Pass=("Rubric Pass Count", "sum"),
        Total_Fail=("Rubric Fail Count", "sum"),
        Avg_Call_Duration=("Call Duration (min)", "mean")
    ).reset_index()

    # Calculate pass rate
    agent_performance["Pass_Rate"] = (agent_performance["Total_Pass"] / (agent_performance["Total_Pass"] + agent_performance["Total_Fail"]) * 100).fillna(0)
    agent_performance = agent_performance.sort_values("Avg_QA_Score", ascending=False)

    st.dataframe(agent_performance.round(1), use_container_width=True)
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
            int(my_performance["Total_Calls"].iloc[0]),
            f"{my_performance['Avg_QA_Score'].iloc[0]:.1f}%",
            f"{my_performance['Pass_Rate'].iloc[0]:.1f}%",
            f"{my_performance['Avg_Call_Duration'].iloc[0]:.1f}" if not pd.isna(my_performance['Avg_Call_Duration'].iloc[0]) else "N/A"
        ],
        'Team Average': [
            overall_total_calls if overall_total_calls else "N/A",
            f"{overall_avg_score:.1f}%" if overall_avg_score else "N/A",
            f"{overall_pass_rate:.1f}%" if overall_pass_rate else "N/A",
            "N/A"  # Could calculate if needed
        ]
    })
    
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

# --- Performance Alerts ---
st.subheader("⚠️ Performance Alerts")
alerts_df = filtered_df[filtered_df["QA Score"] < alert_threshold] if "QA Score" in filtered_df.columns else pd.DataFrame()
if len(alerts_df) > 0:
    st.warning(f"⚠️ {len(alerts_df)} call(s) below threshold ({alert_threshold}%)")
    alert_summary = alerts_df.groupby("Agent").agg(
        Low_Score_Calls=("Call ID", "count"),
        Avg_Score=("QA Score", "mean")
    ).reset_index().sort_values("Low_Score_Calls", ascending=False)
    st.dataframe(alert_summary, use_container_width=True)
else:
    st.success(f"✅ All calls meet the threshold ({alert_threshold}%)")

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
        
        st.dataframe(rubric_analysis.round(1), use_container_width=True)
        
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
    agent_trends_col1, agent_trends_col2 = st.columns(2)
    
    with agent_trends_col1:
        st.write("**My QA Score Trend**")
        agent_data = filtered_df[filtered_df["Agent"] == user_agent_id]
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
            st.dataframe(top_coaching, use_container_width=True)
        
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
                st.write(f"**Call Length:** {call_dur} min" if isinstance(call_dur, (int, float)) else f"**Call Length:** {call_dur}")
                
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
                st.dataframe(rubric_df, use_container_width=True)
                
                # Export individual call report
                st.markdown("---")
                report_text = f"""
# QA Call Report

## Call Information
- **Call ID:** {call_details.get('Call ID', 'N/A')}
- **Agent:** {call_details.get('Agent', 'N/A')}
- **Date:** {call_details.get('Call Date', 'N/A')}
- **Time:** {call_details.get('Call Time', 'N/A')}
- **QA Score:** {call_details.get('QA Score', 'N/A')}%
- **Label:** {call_details.get('Label', 'N/A')}
- **Call Length:** {call_details.get('Call Duration (min)', 'N/A')} min

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

# --- Export Options ---
st.markdown("---")
st.subheader("📥 Export Data")

# Create Excel export
excel_buffer = io.BytesIO()
with ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    export_df = filtered_df.copy()
    
    # Clean data for export
    from datetime import datetime, timezone
    def _clean(val):
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        if isinstance(val, datetime) and val.tzinfo is not None:
            val = val.astimezone(timezone.utc).replace(tzinfo=None)
            return val
        return val
    
    for col in export_df.columns:
        export_df[col] = export_df[col].map(_clean)
    
    export_df.to_excel(writer, sheet_name="QA Data", index=False)
    agent_performance.to_excel(writer, sheet_name="Agent Performance", index=False)

st.download_button(
    label="📥 Download QA Data (Excel)",
    data=excel_buffer.getvalue(),
    file_name=f"qa_report_{start_date}_to_{end_date}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")
st.markdown("Built with ❤️ by [Valence](https://www.getvalenceai.com) | QA Dashboard © 2025")
