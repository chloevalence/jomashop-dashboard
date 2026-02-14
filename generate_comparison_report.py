#!/usr/bin/env python3
"""
Contact Center Comparison Report Generator

Generates a PDF report comparing Jomashop's previous contact center performance
with BPO Centers performance after the transition date (July 7, 2025).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from pathlib import Path
import json
import argparse
import sys
import re
from typing import List, Dict, Optional, Any
import warnings

# Configure matplotlib (must be imported before warnings)
import matplotlib

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()
log_dir = PROJECT_ROOT / "logs"
CACHE_FILE = log_dir / "cached_calls_data.json"

# Transition date: July 7, 2025
TRANSITION_DATE = datetime(2025, 7, 7)

# BPO cohort start date (data from this date onward)
BPO_COHORT_START_DATE = datetime(2025, 8, 1)

# Pass threshold: QA score >= 70 = pass (Excellent + Good)
# Pass = Excellent (90-100) + Good (70-89)
PASS_THRESHOLD = 70

# Rounding tolerance for validation
ROUNDING_TOLERANCE = 0.01

# Tolerance for pass_rate vs bucket reconciliation
PASS_RATE_RECONCILE_TOLERANCE = 0.01

# Consistency std change threshold: no material change if abs(delta) < this
CONSISTENCY_NO_CHANGE_THRESHOLD = 0.5

# Pass definition text (must include ≥ or >= for validation)
PASS_DEFINITION_TEXT = "Passing is defined as a QA score ≥ 70%."
PASS_DEFINITION_TEXT_ASCII = "Passing is defined as a QA score >= 70%."


def _validate_pass_definition_text() -> None:
    """Ensure pass definition includes ≥ 70% or >= 70%."""
    if "≥ 70%" not in PASS_DEFINITION_TEXT and ">= 70%" not in PASS_DEFINITION_TEXT:
        raise ValueError(
            f"Pass definition text must include '≥ 70%' or '>= 70%'. "
            f"Got: {PASS_DEFINITION_TEXT!r}"
        )

# Agent leaderboard Top N (remainder goes to Other/Unknown row)
AGENT_LEADERBOARD_TOP_N = 15

# Set style for professional-looking charts
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


# Known agent mappings (from user specification)
KNOWN_AGENT_MAPPINGS = {
    # Agent 1: Jesus
    "unknown": "Agent 1",
    "bp016803073": "Agent 1",
    "bp016803074": "Agent 1",
    # Agent 2: Gerardo
    "bpagent024577540": "Agent 2",
    # Agent 3: Edgar
    "bpagent030844482": "Agent 3",
    # Agent 4: Osiris
    "bpagent047779576": "Agent 4",
    # Agent 6: Daniela
    "bpagent065185612": "Agent 6",
    # Agent 7: Yasmin
    "bpagent072229421": "Agent 7",
    # Agent 8: Moises
    "bpagent089724913": "Agent 8",
    # Agent 9: Marcos
    "bpagent093540654": "Agent 9",
    # Agent 10: Angel
    "bpagent102256681": "Agent 10",
    # Agent 5: (Left, but calls need to be accessible to admins)
    "bpagent051705087": "Agent 5",
    # Agent 11: (No agent account, but viewable to admins)
    "bpagent113827380": "Agent 11",
}


def load_agent_mapping():
    """Load agent ID mapping from file, or return known mappings if file doesn't exist."""
    mapping_file = PROJECT_ROOT / "logs" / "agent_id_mapping.json"
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                # Merge with known mappings (known mappings take precedence)
                merged_mapping = {**mapping, **KNOWN_AGENT_MAPPINGS}
                return merged_mapping
        except Exception as e:
            print(
                f"Warning: Failed to load agent mapping file: {e}, using known mappings only"
            )
            return KNOWN_AGENT_MAPPINGS.copy()
    return KNOWN_AGENT_MAPPINGS.copy()


def get_or_create_agent_mapping(agent_id_lower):
    """Get agent number for an agent ID, or create a new mapping deterministically."""
    mapping = load_agent_mapping()

    # Check if already mapped
    if agent_id_lower in mapping:
        return mapping[agent_id_lower]

    # Check if already in normalized format (e.g., "agent 1", "agent 2")
    # This handles cases where cache already has normalized values
    if agent_id_lower.startswith("agent "):
        # Already normalized, return as-is (with proper capitalization)
        agent_num_str = agent_id_lower.replace("agent ", "").strip()
        try:
            agent_num = int(agent_num_str)
            return f"Agent {agent_num}"
        except ValueError:
            pass  # Not a valid number, continue with mapping logic

    # Check special cases first
    if agent_id_lower == "unknown":
        return "Agent 1"
    if agent_id_lower in ["bp016803073", "bp016803074"]:
        return "Agent 1"
    if agent_id_lower.startswith("bp01"):
        return "Agent 1"

    # For new agent IDs, assign deterministically based on hash
    # This ensures the same agent ID always gets the same number
    import hashlib

    hash_value = int(hashlib.md5(agent_id_lower.encode()).hexdigest(), 16)
    # Use modulo to get a number between 1 and 99, but skip 5 (no agent 5 per user spec)
    agent_number = (hash_value % 99) + 1
    if agent_number == 5:
        agent_number = 11  # Skip 5, use 11 instead

    return f"Agent {agent_number}"


def normalize_agent_id(agent_str):
    """Normalize agent ID to Agent ## format using stable mapping system."""
    if pd.isna(agent_str) or not agent_str:
        return agent_str

    agent_str_lower = str(agent_str).lower().strip()

    # Use mapping system
    return get_or_create_agent_mapping(agent_str_lower)


def extract_products_from_text(text):
    """
    Extract products from text fields (Summary, Reason, Outcome).
    Uses keyword matching to identify watch brands, jewelry types, and product categories.

    Args:
        text: Text string to extract products from (can be None or empty)

    Returns:
        List of extracted products, or empty list if none found
    """
    if not text or pd.isna(text):
        return []

    text_lower = str(text).lower()
    products = []

    # Watch brands (common luxury and fashion watch brands)
    watch_brands = [
        "rolex",
        "omega",
        "tag heuer",
        "breitling",
        "cartier",
        "patek philippe",
        "audemars piguet",
        "jaeger-lecoultre",
        "vacheron constantin",
        "panerai",
        "iwc",
        "zenith",
        "tudor",
        "longines",
        "tissot",
        "hamilton",
        "seiko",
        "citizen",
        "casio",
        "fossil",
        "michael kors",
        "bulova",
        "movado",
        "baume & mercier",
        "montblanc",
        "hublot",
        "richard mille",
        "ap",
        "patek",
        "jlc",
        "vc",
        "iwc schaffhausen",
        "grand seiko",
    ]

    # Product categories (singular/plural normalized to same display name)
    PRODUCT_CATEGORY_NORMALIZE = {
        "watch": "Watch",
        "watches": "Watch",
        "timepiece": "Timepiece",
        "timepieces": "Timepiece",
        "wristwatch": "Wristwatch",
        "wristwatches": "Wristwatch",
        "bracelet": "Bracelet",
        "bracelets": "Bracelet",
        "necklace": "Necklace",
        "necklaces": "Necklace",
        "ring": "Ring",
        "rings": "Ring",
        "earrings": "Earrings",
        "pendant": "Pendant",
        "pendants": "Pendant",
        "jewelry": "Jewelry",
        "jewellery": "Jewelry",
        "accessories": "Accessories",
    }
    product_categories = list(PRODUCT_CATEGORY_NORMALIZE.keys())

    # Check for watch brands
    for brand in watch_brands:
        if brand in text_lower:
            # Normalize brand name (capitalize properly)
            brand_normalized = brand.title()
            if brand == "ap":
                brand_normalized = "Audemars Piguet"
            elif brand == "jlc":
                brand_normalized = "Jaeger-LeCoultre"
            elif brand == "vc":
                brand_normalized = "Vacheron Constantin"
            elif brand == "patek":
                brand_normalized = "Patek Philippe"
            elif brand == "iwc schaffhausen":
                brand_normalized = "IWC"
            elif brand == "grand seiko":
                brand_normalized = "Grand Seiko"

            if brand_normalized not in products:
                products.append(brand_normalized)

    # Check for product categories
    for category in product_categories:
        if category in text_lower:
            category_normalized = PRODUCT_CATEGORY_NORMALIZE[category]
            if category_normalized not in products:
                products.append(category_normalized)

    # Look for SKU patterns (alphanumeric codes, model numbers)
    # Pattern: alphanumeric codes that might be SKUs or model numbers
    sku_pattern = r"\b[A-Z]{2,}\d{3,}\b|\b\d{4,}[A-Z]{2,}\b"
    sku_matches = re.findall(sku_pattern, str(text), re.IGNORECASE)
    for sku in sku_matches:
        if len(sku) >= 6 and sku.upper() not in products:  # Only add substantial codes
            products.append(f"SKU: {sku.upper()}")

    # Look for model numbers (patterns like "Submariner", "Speedmaster", etc.)
    # Common watch model patterns
    model_keywords = [
        "submariner",
        "gmt",
        "daytona",
        "yacht-master",
        "explorer",
        "datejust",
        "speedmaster",
        "seamaster",
        "constellation",
        "de ville",
        "aqua terra",
        "carrera",
        "monaco",
        "aquaracer",
        "formula 1",
        "link",
        "calibre",
        "navitimer",
        "chronomat",
        "superocean",
        "avenger",
        "transocean",
    ]

    for model in model_keywords:
        if model in text_lower:
            model_normalized = model.title().replace("-", " ")
            if model_normalized not in products:
                products.append(f"Model: {model_normalized}")

    return products


def extract_date_from_call_id(call_id: str) -> Optional[datetime]:
    """Extract date from call_id format: YYYYMMDD_HHMMSS_..."""
    if pd.isna(call_id) or not call_id:
        return None

    call_id_str = str(call_id).strip()
    if len(call_id_str) >= 8:
        try:
            date_str = call_id_str[:8]  # First 8 characters = YYYYMMDD
            return datetime.strptime(date_str, "%Y%m%d")
        except (ValueError, TypeError):
            return None
    return None


def apply_global_filters(
    df: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    """
    Apply shared global filters to produce a canonical filtered dataframe.
    All report sections must use this canonical df for consistency.

    Filters:
    - require non-null call_id
    - require non-null QA Score (for QA-dependent metrics)
    - ensure Call Date is valid (non-NaT)
    - dedupe strictly by call_id (keep first)

    Returns:
        Filtered and deduplicated DataFrame. total_calls = len(result) = nunique(call_id).
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # Ensure call_id column exists
    id_col = None
    for col in out.columns:
        if col.lower() in ("call_id", "call id", "callid"):
            id_col = col
            break
    if id_col and "call_id" not in out.columns:
        out["call_id"] = out[id_col].astype(str)
    elif "_id" in out.columns and "call_id" not in out.columns:
        out["call_id"] = out["_id"].astype(str)
    elif "call_id" not in out.columns:
        out["call_id"] = "row_" + out.index.astype(str)

    # Require non-null call_id
    out = out[out["call_id"].notna() & (out["call_id"].astype(str).str.strip() != "")]

    # Require non-null QA Score (for QA-dependent metrics)
    qa_col = None
    for c in out.columns:
        if str(c).strip().lower() == "qa score":
            qa_col = c
            break
    if qa_col:
        out = out[out[qa_col].notna()]

    # Ensure Call Date is valid
    if "Call Date" in out.columns:
        out["Call Date"] = pd.to_datetime(out["Call Date"], errors="coerce")
        out = out[out["Call Date"].notna()]

    # Dedupe by call_id (keep first)
    out = out.drop_duplicates(subset=["call_id"], keep="first").copy()

    return out


def canonicalize_label(label: str) -> str:
    """
    Canonicalize a category label for aggregation.
    - strip whitespace
    - lowercase
    - collapse internal multiple spaces
    Returns canonical form (used for aggregation keys).
    """
    if pd.isna(label) or label is None:
        return ""
    s = str(label).strip().lower()
    s = " ".join(s.split())  # collapse multiple spaces
    return s


def label_to_display(canonical: str) -> str:
    """Convert canonical label to Title Case for display."""
    if not canonical:
        return "Unknown"
    return canonical.title()


def validate_report_inputs(
    previous_df: Optional[pd.DataFrame],
    bpo_df: pd.DataFrame,
) -> None:
    """
    Assert required columns exist. Fail fast with actionable error messages.
    Required: call_id (or Call ID), cohort label (implicit), qa_score, handle_time (AHT),
    agent_id (Agent), reason, outcome, date (Call Date).
    """
    required_per_cohort = {
        "date": ["Call Date", "call_date", "date_raw"],
        "qa_score": ["QA Score", "qa_score"],
        "agent_id": ["Agent", "agent"],
        "reason": ["Reason", "reason"],
        "outcome": ["Outcome", "outcome"],
        "handle_time": ["AHT", "handle_time_minutes", "Handle Time"],
    }
    # call_id can be Call ID, call_id, or _id

    def _has_any(df: pd.DataFrame, candidates: List[str]) -> bool:
        cols_lower = {c.lower(): c for c in df.columns}
        return any(c.lower() in cols_lower for c in candidates)

    def _has_date(df: pd.DataFrame) -> bool:
        return _has_any(df, required_per_cohort["date"])

    def _has_qa(df: pd.DataFrame) -> bool:
        return _has_any(df, required_per_cohort["qa_score"])

    def _has_agent(df: pd.DataFrame) -> bool:
        return _has_any(df, required_per_cohort["agent_id"])

    errors = []
    if bpo_df is not None and len(bpo_df) > 0:
        if not _has_date(bpo_df):
            errors.append("BPO: Missing date column (Call Date, call_date, or date_raw)")
        if not _has_qa(bpo_df):
            errors.append("BPO: Missing QA Score column")
        if not _has_agent(bpo_df):
            errors.append("BPO: Missing Agent column")

    if previous_df is not None and len(previous_df) > 0:
        if not _has_date(previous_df):
            errors.append("Previous: Missing date column")
        if not _has_qa(previous_df):
            errors.append("Previous: Missing QA Score column")

    if errors:
        raise ValueError(
            "validate_report_inputs failed. Missing required columns:\n  - "
            + "\n  - ".join(errors)
        )


def validate_report_outputs(
    cohort_config: Dict[str, Any],
    previous_metrics: Dict,
    bpo_metrics: Dict,
    previous_reasons: Optional[Dict],
    bpo_reasons: Optional[Dict],
    previous_outcomes: Optional[Dict],
    bpo_outcomes: Optional[Dict],
) -> None:
    """
    Check: cohort totals reconcile, percentages use correct denominators,
    no duplicate labels, trend grain consistent, pass rate matches bucketed distribution.
    """
    errors = []
    # Cohort totals and calls_per_day: abs((cpd * days) - total_calls) < 1
    for name, cfg in cohort_config.items():
        tc = cfg.get("total_calls", 0)
        days = cfg.get("days_in_period", 1)
        cpd = cfg.get("calls_per_day")
        if cpd is not None and days > 0:
            if abs((cpd * days) - tc) >= 1:
                errors.append(
                    f"{name}: calls_per_day * days_in_period does not reconcile with total_calls "
                    f"(got {cpd} * {days} = {cpd * days}, total_calls = {tc})"
                )

    # Pass rate vs bucket sum (pass = Excellent + Good when PASS_THRESHOLD=70)
    for name, metrics in [("Previous", previous_metrics), ("BPO", bpo_metrics)]:
        pr = metrics.get("pass_rate")
        exc = metrics.get("excellent_pct", 0) or 0
        good = metrics.get("good_pct", 0) or 0
        if pr is not None:
            bucket_sum = exc + good
            if abs(pr - bucket_sum) > PASS_RATE_RECONCILE_TOLERANCE:
                errors.append(
                    f"{name}: pass_rate ({pr:.2f}%) does not match "
                    f"Excellent+Good bucket sum ({bucket_sum:.2f}%)"
                )

    # Top-10 percentages plausible (<= 100%)
    for name, dist in [("Reasons-Prev", previous_reasons), ("Reasons-BPO", bpo_reasons)]:
        if dist and isinstance(dist, dict):
            total = sum(dist.values())
            for k, v in list(dist.items())[:10]:
                pct = (v / total * 100) if total > 0 else 0
                if pct > 100:
                    errors.append(f"{name}: category '{k}' has pct > 100%")

    for name, dist in [
        ("Outcomes-Prev", previous_outcomes),
        ("Outcomes-BPO", bpo_outcomes),
    ]:
        if dist and isinstance(dist, dict):
            total = sum(dist.values())
            for k, v in list(dist.items())[:10]:
                pct = (v / total * 100) if total > 0 else 0
                if pct > 100:
                    errors.append(f"{name}: category '{k}' has pct > 100%")

    if errors:
        raise ValueError(
            "validate_report_outputs failed:\n  - " + "\n  - ".join(errors)
        )


def _compute_agent_reconciliation_stats(
    bpo_df: pd.DataFrame, total_calls_bpo: int
) -> Dict[str, Any]:
    """
    Compute agent call counts using nunique(call_id) for reconciliation.
    Returns agent_sum_all, top_n_sum, other_unknown for runtime log.
    """
    if bpo_df is None or len(bpo_df) == 0 or "Agent" not in bpo_df.columns:
        return {
            "agent_sum_all": 0,
            "top_n_sum": None,
            "other_unknown": 0,
        }

    df = bpo_df.copy()
    df["Agent"] = df["Agent"].fillna("Unknown Agent")
    df.loc[df["Agent"].astype(str).str.strip() == "", "Agent"] = "Unknown Agent"

    # Use nunique(call_id) for true call counts
    call_id_col = "call_id" if "call_id" in df.columns else None
    if call_id_col is None:
        for c in df.columns:
            if c.lower() in ("call_id", "call id", "callid"):
                call_id_col = c
                break
    if call_id_col is None:
        # Fallback: after global filters we have call_id; if not, use row count
        call_id_col = "call_id" if "call_id" in df.columns else None

    if call_id_col:
        calls_per_agent = df.groupby("Agent")[call_id_col].nunique()
    else:
        calls_per_agent = df.groupby("Agent").size()

    agent_sum_all = int(calls_per_agent.sum())
    sorted_agents = calls_per_agent.sort_values(ascending=False)
    top_n = sorted_agents.head(AGENT_LEADERBOARD_TOP_N)
    top_n_sum = int(top_n.sum()) if len(top_n) > 0 else 0
    other_unknown = total_calls_bpo - top_n_sum

    return {
        "agent_sum_all": agent_sum_all,
        "top_n_sum": top_n_sum,
        "other_unknown": max(0, other_unknown),
    }


def validate_agent_totals(bpo_df: pd.DataFrame, total_calls: int) -> None:
    """
    Require sum(agent_call_counts) == total_calls.
    Uses nunique(call_id) for true call counts. Expects canonical df_bpo (already filtered/deduped).
    """
    if bpo_df is None or len(bpo_df) == 0 or "Agent" not in bpo_df.columns:
        return

    df = bpo_df.copy()
    df["Agent"] = df["Agent"].fillna("Unknown Agent")
    df.loc[df["Agent"].astype(str).str.strip() == "", "Agent"] = "Unknown Agent"

    call_id_col = "call_id" if "call_id" in df.columns else "Call ID"
    if call_id_col not in df.columns:
        return  # Skip if no call_id
    calls_per_agent = df.groupby("Agent")[call_id_col].nunique()
    sum_counts = int(calls_per_agent.sum())
    if abs(sum_counts - total_calls) >= 1:
        raise ValueError(
            f"Agent leaderboard does not reconcile to cohort totals: "
            f"sum(agent_call_counts)={sum_counts} != total_calls_bpo={total_calls}"
        )


def validate_report(
    previous_df: Optional[pd.DataFrame],
    bpo_df: pd.DataFrame,
    cohort_config: Dict[str, Any],
    previous_metrics: Dict,
    bpo_metrics: Dict,
) -> None:
    """
    Master validation block. Stops report generation if any check fails.
    """
    _validate_pass_definition_text()
    validate_report_inputs(previous_df, bpo_df)
    validate_report_outputs(
        cohort_config,
        previous_metrics,
        bpo_metrics,
        previous_metrics.get("reason_distribution") if previous_metrics else None,
        bpo_metrics.get("reason_distribution"),
        previous_metrics.get("outcome_distribution") if previous_metrics else None,
        bpo_metrics.get("outcome_distribution"),
    )
    # Agent totals must reconcile for BPO
    bpo_cfg = cohort_config.get("BPO Centers", {})
    if bpo_cfg.get("total_calls", 0) > 0:
        validate_agent_totals(bpo_df, bpo_cfg["total_calls"])


def _get_s3_config() -> Optional[Dict[str, str]]:
    """Get S3 bucket and credentials from .streamlit/secrets.toml or env."""
    config = {}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            tomllib = None
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if tomllib is not None and secrets_path.exists():
        try:
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
            s3 = secrets.get("s3", {})
            if s3.get("bucket_name"):
                config["bucket"] = s3["bucket_name"]
                config["prefix"] = s3.get("prefix", "cache")
                config["aws_access_key_id"] = s3.get("aws_access_key_id", "")
                config["aws_secret_access_key"] = s3.get("aws_secret_access_key", "")
                config["region_name"] = s3.get("region_name", "us-east-1")
        except Exception:
            pass
    if not config.get("bucket"):
        bucket = os.environ.get("BPO_S3_BUCKET")
        if bucket:
            config["bucket"] = bucket
            config["prefix"] = os.environ.get("BPO_S3_PREFIX", "cache")
            config["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID", "")
            config["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
            config["region_name"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return config if config.get("bucket") else None


def _load_bpo_call_data_from_s3() -> List[Dict[str, Any]]:
    """Load and merge BPO month caches from S3. Returns list of call dicts or empty list."""
    config = _get_s3_config()
    if not config:
        return []
    try:
        import boto3
        from botocore.exceptions import ClientError

        kwargs = {"region_name": config.get("region_name", "us-east-1")}
        if config.get("aws_access_key_id") and config.get("aws_secret_access_key"):
            kwargs["aws_access_key_id"] = config["aws_access_key_id"]
            kwargs["aws_secret_access_key"] = config["aws_secret_access_key"]
        client = boto3.client("s3", **kwargs)
        prefix = (config.get("prefix") or "cache").rstrip("/") + "/"
        prefix_full = prefix + "cached_calls_data_"
        paginator = client.get_paginator("list_objects_v2")
        all_calls = []
        seen_ids = set()

        for page in paginator.paginate(Bucket=config["bucket"], Prefix=prefix_full):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if not key.endswith(".json") or "cached_calls_data_" not in key:
                    continue
                try:
                    resp = client.get_object(Bucket=config["bucket"], Key=key)
                    data = json.loads(resp["Body"].read().decode("utf-8"))
                    if isinstance(data, dict):
                        calls = data.get("call_data", data.get("calls", data.get("data", [])))
                    else:
                        calls = data if isinstance(data, list) else []
                    for c in calls:
                        cid = c.get("Call ID") or c.get("call_id") or str(c.get("_id", ""))
                        if cid and cid not in seen_ids:
                            seen_ids.add(cid)
                            all_calls.append(c)
                        elif not cid:
                            all_calls.append(c)
                except (ClientError, json.JSONDecodeError, KeyError, AttributeError) as e:
                    print(f"   ⚠️  Skipped {key}: {e}")
                    continue
        return all_calls
    except ImportError:
        return []
    except Exception as e:
        print(f"   ⚠️  S3 fallback failed: {e}")
        return []


def load_bpo_centers_data(cache_path: Path, start_date: datetime) -> pd.DataFrame:
    """
    Load BPO Centers data from cache, split by transition date.
    If local cache is empty, tries to load from S3 month caches (secrets.toml or env).
    """
    print(f"📂 Loading BPO Centers data from: {cache_path}")

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)

        # Extract call data
        if isinstance(cached_data, dict):
            call_data = cached_data.get(
                "call_data", cached_data.get("calls", cached_data.get("data", []))
            )
        else:
            call_data = cached_data

        if not call_data:
            print("   Local cache is empty (app may run in monthly-only mode). Trying S3 month caches...")
            call_data = _load_bpo_call_data_from_s3()
            if call_data:
                print(f"✅ Loaded {len(call_data)} calls from S3 month caches")
            else:
                raise ValueError(
                    "No call data in cache file and could not load from S3. "
                    "Either run the Streamlit app without a month selected so it writes the full cache to disk, "
                    "or set S3 access (e.g. .streamlit/secrets.toml with [s3] or env BPO_S3_BUCKET and AWS_*)."
                )
        else:
            print(f"✅ Loaded {len(call_data)} calls from cache")

        # Convert to DataFrame
        df = pd.DataFrame(call_data)

        # Normalize column names (case-insensitive mapping)
        # Priority: call_date > date_raw for Call Date
        column_mapping = {
            "call_date": "Call Date",
            "agent": "Agent",
            "qa_score": "QA Score",
            "label": "Label",
            "reason": "Reason",
            "outcome": "Outcome",
            "summary": "Summary",
            "rubric_pass_count": "Rubric Pass Count",
            "rubric_fail_count": "Rubric Fail Count",
            "rubric_details": "Rubric Details",
        }

        # Create rename dictionary by checking all columns (case-insensitive)
        rename_dict = {}
        columns_lower = {col.lower(): col for col in df.columns}

        for old_col_lower, new_col in column_mapping.items():
            if old_col_lower in columns_lower:
                old_col = columns_lower[old_col_lower]
                # Only rename if the new name doesn't already exist or if it's different
                if new_col not in df.columns or old_col != new_col:
                    rename_dict[old_col] = new_col

        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)

        # Handle date_raw as fallback if Call Date doesn't exist yet
        # Track which columns we've tried to use so we don't drop them prematurely
        date_source_col = None
        if "Call Date" not in df.columns:
            if "date_raw" in df.columns:
                df["Call Date"] = pd.to_datetime(df["date_raw"], errors="coerce")
                date_source_col = "date_raw"
            elif "call_date" in df.columns:
                df["Call Date"] = pd.to_datetime(df["call_date"], errors="coerce")
                date_source_col = "call_date"

        # Ensure Call Date is datetime
        if "Call Date" in df.columns:
            # Check if it's already datetime, if not convert it
            if not pd.api.types.is_datetime64_any_dtype(df["Call Date"]):
                df["Call Date"] = pd.to_datetime(df["Call Date"], errors="coerce")

            # Only drop the source column if Call Date was successfully created (has valid dates)
            if date_source_col and date_source_col in df.columns:
                # Check if Call Date has any valid (non-NaT) values
                if df["Call Date"].notna().any():
                    df = df.drop(columns=[date_source_col])
                    date_source_col = None  # Mark as dropped
        else:
            # Try to extract from other date fields (case-insensitive)
            # Only look for columns that haven't been dropped yet
            date_col = None
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ["date_raw", "call_date"]:
                    date_col = col
                    break

            if date_col:
                df["Call Date"] = pd.to_datetime(df[date_col], errors="coerce")
                # Only drop if Call Date was successfully created
                if "Call Date" in df.columns and df["Call Date"].notna().any():
                    df = df.drop(columns=[date_col])
            else:
                raise ValueError("No date column found in BPO Centers data")

        # Filter out rows with NaT (null datetime) values - similar to load_previous_center_data
        valid_dates = df["Call Date"].notna()
        df = df[valid_dates].copy()

        if len(df) == 0:
            print("⚠️  Warning: No calls with valid dates in BPO Centers data")
            return pd.DataFrame()

        # Normalize agent IDs - check for both "Agent" and "agent" columns (case-insensitive)
        for col in df.columns:
            if col.lower() == "agent":
                if col != "Agent":
                    df.rename(columns={col: "Agent"}, inplace=True)
                break

        if "Agent" in df.columns:
            # Apply normalization to all agent IDs
            original_sample = df["Agent"].head(5).tolist() if len(df) > 0 else []
            df["Agent"] = df["Agent"].apply(normalize_agent_id)
            normalized_sample = df["Agent"].head(5).tolist() if len(df) > 0 else []
            if original_sample:
                print(
                    f"   Agent ID normalization: {original_sample[0]} -> {normalized_sample[0]}"
                )
                # Verify normalization worked
                unique_agents = df["Agent"].unique()[:10]
                print(
                    f"   Sample normalized agent IDs: {', '.join(map(str, unique_agents))}"
                )
        else:
            print("   ⚠️  Warning: No 'Agent' column found in data")

        # Calculate AHT from speaking_time_per_speaker if available
        if "AHT" not in df.columns and "speaking_time_per_speaker" in df.columns:

            def compute_aht(row):
                speaking_times = row.get("speaking_time_per_speaker")
                if isinstance(speaking_times, dict):
                    total_seconds = 0
                    for t in speaking_times.values():
                        if isinstance(t, str) and ":" in t:
                            try:
                                parts = t.split(":")
                                if len(parts) == 2:
                                    minutes, seconds = map(int, parts)
                                    total_seconds += minutes * 60 + seconds
                            except (ValueError, IndexError):
                                pass
                    return total_seconds / 60.0 if total_seconds > 0 else None
                return None

            df["AHT"] = df.apply(compute_aht, axis=1)

        # Ensure call_id column exists (for deduplication and validation)
        id_col = None
        for col in df.columns:
            if col.lower() in ("call_id", "call id", "callid"):
                id_col = col
                break
        if id_col:
            df["call_id"] = df[id_col].astype(str)
        elif "_id" in df.columns:
            df["call_id"] = df["_id"].astype(str)
        else:
            df["call_id"] = "row_" + df.index.astype(str)

        # Filter BPO data to start_date onward (cohort: Aug 01, 2025+)
        df["Call Date"] = pd.to_datetime(df["Call Date"], errors="coerce")
        mask = df["Call Date"] >= pd.Timestamp(start_date)
        df = df[mask].copy()

        print("📊 BPO Centers data loaded (filtered to cohort start_date):")
        print(f"   Total calls: {len(df)}")
        if len(df) > 0 and "Call Date" in df.columns:
            print(f"   Date range: {df['Call Date'].min()} to {df['Call Date'].max()}")

        return df

    except Exception as e:
        raise Exception(f"Error loading BPO Centers data: {e}")


def load_previous_center_data(csv_paths: List[Path]) -> pd.DataFrame:
    """
    Load and normalize previous contact center CSV files.

    Args:
        csv_paths: List of CSV file paths (one per agent)
        transition_date: Only include calls before this date

    Returns:
        Combined DataFrame with all previous contact center data
    """
    print(
        f"📂 Loading previous contact center data from {len(csv_paths)} CSV file(s)..."
    )

    all_dataframes = []

    for i, csv_path in enumerate(csv_paths):
        if not csv_path.exists():
            print(f"⚠️  Warning: CSV file not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"   Loaded {len(df)} calls from {csv_path.name}")

            # Normalize column names to lowercase for easier lookup
            df.columns = df.columns.str.lower()

            # Extract date from call_id (case-insensitive lookup)
            call_id_col = None
            for col in df.columns:
                if "call_id" in col.lower() or col.lower() == "callid":
                    call_id_col = col
                    break

            if call_id_col:
                df["Call Date"] = df[call_id_col].apply(extract_date_from_call_id)
            else:
                print(f"⚠️  Warning: No 'call_id' column found in {csv_path.name}")
                print(f"   Available columns: {list(df.columns)[:10]}...")
                continue

            # Use ALL calls (no date filtering)
            valid_dates = df["Call Date"].notna()
            df = df[valid_dates].copy()

            if len(df) == 0:
                print(f"   ⚠️  No calls with valid dates in {csv_path.name}")
                continue

            # Map columns to standard format (case-insensitive)
            column_mapping = {
                "qa_score": "QA Score",
                "label": "Label",
                "handle_time_minutes": "AHT",
            }

            # Create a mapping dict with actual column names
            rename_dict = {}
            for old_col, new_col in column_mapping.items():
                # Find matching column (case-insensitive)
                for col in df.columns:
                    if col.lower() == old_col.lower():
                        rename_dict[col] = new_col
                        break

            if rename_dict:
                df.rename(columns=rename_dict, inplace=True)

            # Handle QA Score - convert "NA" to NaN
            if "QA Score" in df.columns:
                df["QA Score"] = pd.to_numeric(df["QA Score"], errors="coerce")

            # Calculate rubric pass/fail counts from rubric columns
            # Rubric columns are like "1.1.0", "1.2.0", etc. (not the __reason columns)
            rubric_columns = [
                col
                for col in df.columns
                if re.match(r"^\d+\.\d+\.\d+$", str(col))
                and "__reason" not in str(col).lower()
            ]

            if rubric_columns:

                def count_rubric_results(row):
                    pass_count = 0
                    fail_count = 0
                    for col in rubric_columns:
                        if col not in row.index:
                            continue
                        value = row[col]
                        # Handle various formats: True/False (bool), "True"/"False" (str), "N/A"
                        value_str = str(value).strip().upper()
                        if value is True or value_str == "TRUE" or value_str == "T":
                            pass_count += 1
                        elif value is False or value_str == "FALSE" or value_str == "F":
                            fail_count += 1
                        # N/A values are ignored
                    return pd.Series(
                        {
                            "Rubric Pass Count": pass_count,
                            "Rubric Fail Count": fail_count,
                        }
                    )

                rubric_counts = df.apply(count_rubric_results, axis=1)
                df["Rubric Pass Count"] = rubric_counts["Rubric Pass Count"]
                df["Rubric Fail Count"] = rubric_counts["Rubric Fail Count"]
            else:
                # If no rubric columns, set defaults
                df["Rubric Pass Count"] = 0
                df["Rubric Fail Count"] = 0

            # Add agent identifier from filename
            agent_name = (
                csv_path.stem.replace("qa_results_", "").replace("_", " ").title()
            )
            df["Agent"] = f"Previous Agent {i + 1}" if not agent_name else agent_name

            all_dataframes.append(df)

        except Exception as e:
            print(f"⚠️  Error loading {csv_path.name}: {e}")
            continue

    if not all_dataframes:
        raise ValueError("No valid data loaded from CSV files")

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    print(f"✅ Combined {len(combined_df)} calls from previous contact center")
    print(
        f"   Date range: {combined_df['Call Date'].min()} to {combined_df['Call Date'].max()}"
    )

    return combined_df


def calculate_qa_metrics(df: pd.DataFrame) -> Dict:
    """Calculate QA score metrics."""
    metrics = {}

    if "QA Score" in df.columns:
        valid_scores = df["QA Score"].dropna()
        if len(valid_scores) > 0:
            metrics["avg_qa_score"] = valid_scores.mean()
            metrics["qa_score_count"] = len(valid_scores)
        else:
            metrics["avg_qa_score"] = None
            metrics["qa_score_count"] = 0
    else:
        metrics["avg_qa_score"] = None
        metrics["qa_score_count"] = 0

    return metrics


def calculate_pass_rate_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate pass rate metrics.
    pass_flag = qa_score >= PASS_THRESHOLD (70).
    Pass rate = Excellent (90-100) + Good (70-89).
    Falls back to Rubric or Label if QA Score not available.
    """
    metrics = {}

    # Primary: derive from QA Score to match quality distribution buckets
    if "QA Score" in df.columns:
        valid_scores = df["QA Score"].dropna()
        if len(valid_scores) > 0:
            pass_count = (valid_scores >= PASS_THRESHOLD).sum()
            total = len(valid_scores)
            metrics["pass_rate"] = (pass_count / total) * 100
            metrics["pass_count"] = int(pass_count)
            metrics["fail_count"] = int(total - pass_count)
            return metrics

    # Fallback 1: From Rubric Pass/Fail Count
    if "Rubric Pass Count" in df.columns and "Rubric Fail Count" in df.columns:
        total_pass = df["Rubric Pass Count"].sum()
        total_fail = df["Rubric Fail Count"].sum()
        if (total_pass + total_fail) > 0:
            metrics["pass_rate"] = (total_pass / (total_pass + total_fail)) * 100
            metrics["pass_count"] = int(total_pass)
            metrics["fail_count"] = int(total_fail)
            return metrics

    # Fallback 2: From Label column (Positive = Pass)
    if "Label" in df.columns:
        valid_labels = df["Label"].dropna()
        valid_labels = valid_labels[valid_labels.str.lower() != "invalid"]
        if len(valid_labels) > 0:
            pass_count = (valid_labels.str.lower() == "positive").sum()
            fail_count = (valid_labels.str.lower() == "negative").sum()
            total = pass_count + fail_count
            if total > 0:
                metrics["pass_rate"] = (pass_count / total) * 100
                metrics["pass_count"] = int(pass_count)
                metrics["fail_count"] = int(fail_count)
                return metrics

    metrics["pass_rate"] = None
    metrics["pass_count"] = 0
    metrics["fail_count"] = 0
    return metrics


def calculate_aht_metrics(df: pd.DataFrame) -> Dict:
    """Calculate Average Handle Time metrics."""
    metrics = {}

    if "AHT" in df.columns:
        valid_aht = df["AHT"].dropna()
        if len(valid_aht) > 0:
            metrics["avg_aht"] = valid_aht.mean()
            metrics["aht_count"] = len(valid_aht)
        else:
            metrics["avg_aht"] = None
            metrics["aht_count"] = 0
    else:
        metrics["avg_aht"] = None
        metrics["aht_count"] = 0

    return metrics


def compute_cohort_config(
    df: pd.DataFrame,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    cohort_name: str,
) -> Dict[str, Any]:
    """
    Compute cohort config: total_calls, days_in_period, calls_per_day.
    Uses explicit start_date and end_date. days_in_period is INCLUSIVE (calendar days).
    """
    total_calls = len(df)
    if total_calls == 0 or "Call Date" not in df.columns:
        return {
            "total_calls": 0,
            "days_in_period": 0,
            "calls_per_day": None,
            "start_date": start_date,
            "end_date": end_date,
        }

    valid_dates = df["Call Date"].dropna()
    if len(valid_dates) == 0:
        return {
            "total_calls": total_calls,
            "days_in_period": 0,
            "calls_per_day": None,
            "start_date": start_date,
            "end_date": end_date,
        }

    min_dt = pd.Timestamp(valid_dates.min())
    max_dt = pd.Timestamp(valid_dates.max())

    if start_date is None:
        start_date = min_dt.to_pydatetime()
    else:
        start_date = pd.Timestamp(start_date)
    if end_date is None:
        end_date = max_dt.to_pydatetime()
    else:
        end_date = pd.Timestamp(end_date)

    # Clamp to actual data range
    start_date = max(start_date, min_dt)
    end_date = min(end_date, max_dt)
    # days_in_period: inclusive calendar days
    days_in_period = (end_date - start_date).days + 1
    if days_in_period <= 0:
        days_in_period = 1

    calls_per_day = total_calls / days_in_period

    return {
        "total_calls": total_calls,
        "days_in_period": days_in_period,
        "calls_per_day": calls_per_day,
        "start_date": start_date,
        "end_date": end_date,
    }


def calculate_volume_metrics(
    df: pd.DataFrame,
    cohort_config: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Calculate call volume metrics.
    If cohort_config provided, uses its total_calls, days_in_period, calls_per_day.
    Otherwise falls back to inferring from data (for backward compatibility).
    """
    metrics = {}

    # Total calls (exclude Invalid labels if present)
    if "Label" in df.columns:
        valid_calls = (
            df[df["Label"].str.lower() != "invalid"]
            if df["Label"].dtype == "object"
            else df
        )
        metrics["total_calls"] = len(valid_calls)
    else:
        valid_calls = df
        metrics["total_calls"] = len(df)

    if cohort_config:
        metrics["total_calls"] = cohort_config["total_calls"]
        metrics["days_in_period"] = cohort_config["days_in_period"]
        metrics["calls_per_day"] = cohort_config["calls_per_day"]
    else:
        # Fallback: infer from data
        if "Call Date" in valid_calls.columns and metrics["total_calls"] > 0:
            valid_dates = valid_calls["Call Date"].dropna()
            if len(valid_dates) > 0:
                days_in_period = (
                    pd.Timestamp(valid_dates.max())
                    - pd.Timestamp(valid_dates.min())
                ).days + 1
                metrics["days_in_period"] = max(1, days_in_period)
                metrics["calls_per_day"] = (
                    metrics["total_calls"] / metrics["days_in_period"]
                )
            else:
                metrics["days_in_period"] = 0
                metrics["calls_per_day"] = None
        else:
            metrics["days_in_period"] = 0
            metrics["calls_per_day"] = None

    return metrics


def calculate_rubric_metrics(df: pd.DataFrame) -> Dict:
    """Calculate rubric failure metrics."""
    metrics = {}

    # Find rubric columns (exclude __reason columns)
    rubric_columns = [
        col
        for col in df.columns
        if re.match(r"^\d+\.\d+\.\d+$", str(col)) and "__reason" not in str(col).lower()
    ]

    if rubric_columns:
        failure_counts = {}
        for col in rubric_columns:
            if col not in df.columns:
                continue
            # Count failures (False, "False", "F", etc.)
            failures = 0
            for value in df[col]:
                value_str = str(value).strip().upper() if pd.notna(value) else ""
                if value is False or value_str == "FALSE" or value_str == "F":
                    failures += 1
            if failures > 0:
                failure_counts[col] = failures

        # Sort by failure count
        sorted_failures = sorted(
            failure_counts.items(), key=lambda x: x[1], reverse=True
        )
        metrics["top_failure_codes"] = sorted_failures[:10]  # Top 10
        metrics["total_rubric_failures"] = sum(failure_counts.values())
    else:
        # Use Rubric Fail Count if available
        if "Rubric Fail Count" in df.columns:
            metrics["total_rubric_failures"] = df["Rubric Fail Count"].sum()
        else:
            metrics["total_rubric_failures"] = 0
        metrics["top_failure_codes"] = []

    return metrics


def calculate_agent_metrics(df: pd.DataFrame) -> Dict:
    """Calculate agent performance distribution metrics."""
    metrics = {}

    if "Agent" in df.columns and "QA Score" in df.columns:
        agent_perf = (
            df.groupby("Agent")["QA Score"].agg(["mean", "count"]).reset_index()
        )
        agent_perf.columns = ["Agent", "Avg Score", "Call Count"]
        agent_perf = agent_perf.sort_values("Avg Score", ascending=False)
        metrics["agent_performance"] = agent_perf
        metrics["num_agents"] = len(agent_perf)
    else:
        metrics["agent_performance"] = pd.DataFrame()
        metrics["num_agents"] = 0

    return metrics


def calculate_consistency_metrics(df: pd.DataFrame) -> Dict:
    """Calculate consistency metrics (standard deviation, coefficient of variation)."""
    metrics = {}

    if "QA Score" in df.columns:
        valid_scores = df["QA Score"].dropna()
        if len(valid_scores) > 0:
            metrics["score_std"] = valid_scores.std()
            metrics["score_mean"] = valid_scores.mean()
            metrics["score_median"] = valid_scores.median()
            metrics["score_min"] = valid_scores.min()
            metrics["score_max"] = valid_scores.max()
            metrics["score_range"] = valid_scores.max() - valid_scores.min()
            # Coefficient of variation (normalized consistency)
            if metrics["score_mean"] > 0:
                metrics["coefficient_of_variation"] = (
                    metrics["score_std"] / metrics["score_mean"]
                ) * 100
            else:
                metrics["coefficient_of_variation"] = None
        else:
            metrics["score_std"] = None
            metrics["score_mean"] = None
            metrics["score_median"] = None
            metrics["score_min"] = None
            metrics["score_max"] = None
            metrics["score_range"] = None
            metrics["coefficient_of_variation"] = None
    else:
        metrics["score_std"] = None
        metrics["score_mean"] = None
        metrics["score_median"] = None
        metrics["score_min"] = None
        metrics["score_max"] = None
        metrics["score_range"] = None
        metrics["coefficient_of_variation"] = None

    return metrics


def calculate_quality_distribution(df: pd.DataFrame) -> Dict:
    """
    Calculate quality tier distribution.
    Buckets aligned with PASS_THRESHOLD=70:
    - Excellent: 90-100
    - Good: 70-89
    - Fair: 60-69
    - Poor: <60
    Pass = Excellent + Good
    """
    metrics = {}

    if "QA Score" in df.columns:
        valid_scores = df["QA Score"].dropna()
        if len(valid_scores) > 0:
            total = len(valid_scores)
            excellent = ((valid_scores >= 90) & (valid_scores <= 100)).sum()
            good = ((valid_scores >= PASS_THRESHOLD) & (valid_scores < 90)).sum()
            fair = ((valid_scores >= 60) & (valid_scores < PASS_THRESHOLD)).sum()
            poor = (valid_scores < 60).sum()

            metrics["excellent_count"] = excellent
            metrics["good_count"] = good
            metrics["fair_count"] = fair
            metrics["poor_count"] = poor
            metrics["excellent_pct"] = (excellent / total) * 100
            metrics["good_pct"] = (good / total) * 100
            metrics["fair_pct"] = (fair / total) * 100
            metrics["poor_pct"] = (poor / total) * 100
        else:
            metrics["excellent_count"] = 0
            metrics["good_count"] = 0
            metrics["fair_count"] = 0
            metrics["poor_count"] = 0
            metrics["excellent_pct"] = 0
            metrics["good_pct"] = 0
            metrics["fair_pct"] = 0
            metrics["poor_pct"] = 0
    else:
        metrics["excellent_count"] = 0
        metrics["good_count"] = 0
        metrics["fair_count"] = 0
        metrics["poor_count"] = 0
        metrics["excellent_pct"] = 0
        metrics["good_pct"] = 0
        metrics["fair_pct"] = 0
        metrics["poor_pct"] = 0

    return metrics


def calculate_trend_metrics(df: pd.DataFrame) -> Dict:
    """Calculate trend metrics (weekly/monthly aggregations)."""
    metrics = {}

    if "Call Date" in df.columns and "QA Score" in df.columns:
        df_copy = df[["Call Date", "QA Score"]].copy()
        df_copy = df_copy.dropna()

        if len(df_copy) > 0:
            df_copy["Call Date"] = pd.to_datetime(df_copy["Call Date"])
            df_copy["Week"] = df_copy["Call Date"].dt.to_period("W")
            df_copy["Month"] = df_copy["Call Date"].dt.to_period("M")

            # Weekly trends
            weekly_avg = df_copy.groupby("Week")["QA Score"].mean()
            if len(weekly_avg) > 1:
                # Calculate improvement rate (slope)
                weeks = np.arange(len(weekly_avg))
                scores = weekly_avg.values
                if len(weeks) > 1:
                    slope = np.polyfit(weeks, scores, 1)[0]
                    metrics["weekly_improvement_rate"] = slope
                    metrics["weekly_trend_data"] = weekly_avg.to_dict()
                else:
                    metrics["weekly_improvement_rate"] = None
                    metrics["weekly_trend_data"] = {}
            else:
                metrics["weekly_improvement_rate"] = None
                metrics["weekly_trend_data"] = {}

            # Monthly trends
            monthly_avg = df_copy.groupby("Month")["QA Score"].mean()
            if len(monthly_avg) > 0:
                metrics["monthly_trend_data"] = monthly_avg.to_dict()
            else:
                metrics["monthly_trend_data"] = {}
        else:
            metrics["weekly_improvement_rate"] = None
            metrics["weekly_trend_data"] = {}
            metrics["monthly_trend_data"] = {}
    else:
        metrics["weekly_improvement_rate"] = None
        metrics["weekly_trend_data"] = {}
        metrics["monthly_trend_data"] = {}

    return metrics


def calculate_rubric_improvements(
    previous_df: pd.DataFrame, bpo_df: pd.DataFrame
) -> Dict:
    """Calculate rubric code improvements by comparing failure rates."""
    metrics = {}

    def get_rubric_failure_rates(df):
        """Get failure rates for each rubric code."""
        failure_rates = {}

        # Find rubric columns
        rubric_columns = [
            col
            for col in df.columns
            if re.match(r"^\d+\.\d+\.\d+$", str(col))
            and "__reason" not in str(col).lower()
        ]

        if rubric_columns:
            for col in rubric_columns:
                if col not in df.columns:
                    continue
                total = df[col].notna().sum()
                if total > 0:
                    failures = 0
                    for value in df[col]:
                        value_str = (
                            str(value).strip().upper() if pd.notna(value) else ""
                        )
                        if value is False or value_str == "FALSE" or value_str == "F":
                            failures += 1
                    failure_rate = (failures / total) * 100
                    failure_rates[col] = {
                        "failures": failures,
                        "total": total,
                        "rate": failure_rate,
                    }

        return failure_rates

    previous_rates = get_rubric_failure_rates(previous_df)
    bpo_rates = get_rubric_failure_rates(bpo_df)

    # Find improvements (codes where BPO has lower failure rate)
    improvements = []
    for code, bpo_data in bpo_rates.items():
        if code in previous_rates:
            prev_rate = previous_rates[code]["rate"]
            bpo_rate = bpo_data["rate"]
            improvement = prev_rate - bpo_rate  # Positive = improvement
            if improvement > 0:  # Only show improvements
                improvements.append(
                    {
                        "code": code,
                        "previous_rate": prev_rate,
                        "bpo_rate": bpo_rate,
                        "improvement": improvement,
                        "improvement_pct": (improvement / prev_rate * 100)
                        if prev_rate > 0
                        else 0,
                    }
                )

    # Sort by improvement amount
    improvements.sort(key=lambda x: x["improvement"], reverse=True)
    metrics["top_improvements"] = improvements[:10]  # Top 10
    metrics["total_improvements"] = len(improvements)

    return metrics


def calculate_reason_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate call reason distribution metrics.
    Canonicalizes labels before aggregation (strip, lowercase, collapse spaces).
    Display labels use Title Case.
    Percentages use calls_with_reason as denominator (% of calls that have reason data).
    """
    metrics = {}

    reason_col = None
    for col in df.columns:
        if col.lower() == "reason":
            reason_col = col
            break
    if reason_col is None:
        return metrics

    total_calls = len(df)
    # Canonicalize before aggregation
    canonical = df[reason_col].fillna("").apply(canonicalize_label)
    # Aggregate by canonical key
    reason_counts = canonical[canonical != ""].value_counts()
    calls_with_reason = int(reason_counts.sum()) if len(reason_counts) > 0 else 0

    if calls_with_reason == 0 or total_calls == 0:
        metrics["reason_distribution"] = {}
        metrics["top_reason"] = None
        metrics["top_reason_count"] = 0
        metrics["top_reason_pct"] = 0
        metrics["unique_reasons"] = 0
        metrics["reason_calls_with_data"] = 0
        return metrics

    # Build distribution with display labels (Title Case)
    dist = {}
    for canonical_key, count in reason_counts.items():
        display_label = label_to_display(canonical_key)
        if display_label in dist:
            dist[display_label] += count
        else:
            dist[display_label] = count

    metrics["reason_distribution"] = dist
    top_canonical = reason_counts.index[0]
    metrics["top_reason"] = label_to_display(top_canonical)
    metrics["top_reason_count"] = int(reason_counts.iloc[0])
    # Percentages use calls_with_reason as denominator (% of calls that have reason data)
    metrics["top_reason_pct"] = (reason_counts.iloc[0] / calls_with_reason) * 100
    metrics["unique_reasons"] = len(dist)
    metrics["reason_calls_with_data"] = calls_with_reason

    return metrics


def calculate_outcome_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate call outcome distribution metrics.
    Canonicalizes labels before aggregation.
    Percentages use calls_with_outcome as denominator (% of calls that have outcome data).
    """
    metrics = {}

    outcome_col = None
    for col in df.columns:
        if col.lower() == "outcome":
            outcome_col = col
            break
    if outcome_col is None:
        return metrics

    total_calls = len(df)
    canonical = df[outcome_col].fillna("").apply(canonicalize_label)
    outcome_counts = canonical[canonical != ""].value_counts()
    calls_with_outcome = int(outcome_counts.sum()) if len(outcome_counts) > 0 else 0

    if calls_with_outcome == 0 or total_calls == 0:
        metrics["outcome_distribution"] = {}
        metrics["top_outcome"] = None
        metrics["top_outcome_count"] = 0
        metrics["top_outcome_pct"] = 0
        metrics["unique_outcomes"] = 0
        metrics["outcome_calls_with_data"] = 0
        return metrics

    dist = {}
    for canonical_key, count in outcome_counts.items():
        display_label = label_to_display(canonical_key)
        if display_label in dist:
            dist[display_label] += count
        else:
            dist[display_label] = count

    metrics["outcome_distribution"] = dist
    top_canonical = outcome_counts.index[0]
    metrics["top_outcome"] = label_to_display(top_canonical)
    metrics["top_outcome_count"] = int(outcome_counts.iloc[0])
    # Percentages use calls_with_outcome as denominator (% of calls that have outcome data)
    metrics["top_outcome_pct"] = (outcome_counts.iloc[0] / calls_with_outcome) * 100
    metrics["unique_outcomes"] = len(dist)
    metrics["outcome_calls_with_data"] = calls_with_outcome

    return metrics


def calculate_product_metrics(df: pd.DataFrame) -> Dict:
    """Calculate product distribution metrics from Summary, Reason, and Outcome fields."""
    metrics = {}

    # Combine text from Summary, Reason, and Outcome fields
    product_data = []
    text_fields = []

    if "Summary" in df.columns:
        text_fields.append("Summary")
    if "Reason" in df.columns:
        text_fields.append("Reason")
    if "Outcome" in df.columns:
        text_fields.append("Outcome")

    if not text_fields:
        return metrics

    for idx, row in df.iterrows():
        combined_text = " ".join(
            [str(row.get(field, "") or "") for field in text_fields]
        )
        products = extract_products_from_text(combined_text)
        product_data.extend(products)

    if len(product_data) == 0:
        return metrics

    product_counts = pd.Series(product_data).value_counts()
    total_mentions = len(product_data)

    metrics["product_distribution"] = product_counts.to_dict()
    metrics["top_product"] = (
        product_counts.index[0] if len(product_counts) > 0 else None
    )
    metrics["top_product_count"] = (
        product_counts.iloc[0] if len(product_counts) > 0 else 0
    )
    metrics["top_product_pct"] = (
        (product_counts.iloc[0] / total_mentions * 100)
        if len(product_counts) > 0
        else 0
    )
    metrics["unique_products"] = len(product_counts)
    metrics["total_product_mentions"] = total_mentions
    # Count calls where ANY of the text fields have non-null data
    # (since product extraction combines all available text fields)
    if text_fields:
        calls_with_any_text = df[df[text_fields].notna().any(axis=1)]
        metrics["calls_with_products"] = len(calls_with_any_text)
    else:
        metrics["calls_with_products"] = 0

    return metrics


def calculate_all_metrics(
    df: pd.DataFrame,
    cohort_config: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Calculate all KPIs for a dataset."""
    metrics = {}

    metrics.update(calculate_qa_metrics(df))
    metrics.update(calculate_pass_rate_metrics(df))
    metrics.update(calculate_aht_metrics(df))
    metrics.update(calculate_volume_metrics(df, cohort_config=cohort_config))
    metrics.update(calculate_rubric_metrics(df))
    metrics.update(calculate_agent_metrics(df))
    metrics.update(calculate_consistency_metrics(df))
    metrics.update(calculate_quality_distribution(df))
    metrics.update(calculate_trend_metrics(df))

    metrics.update(calculate_reason_metrics(df))
    metrics.update(calculate_outcome_metrics(df))
    metrics.update(calculate_product_metrics(df))

    return metrics


def create_comparison_chart(
    metric_name: str,
    previous_value: float,
    bpo_value: float,
    previous_label: str = "Previous Contact Center",
    bpo_label: str = "BPO Centers",
) -> plt.Figure:
    """Create a bar chart comparing a metric between previous and BPO Centers."""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = [previous_label, bpo_label]
    values = [previous_value, bpo_value]

    # Determine colors (green for improvement if BPO > previous for positive metrics,
    # or BPO < previous for negative metrics like AHT)
    colors = ["#e74c3c", "#2ecc71"]  # Red, Green

    # For metrics where lower is better (like AHT), reverse the color logic
    if "AHT" in metric_name or "Handle Time" in metric_name:
        if bpo_value < previous_value:
            colors = ["#e74c3c", "#2ecc71"]  # Improvement (green)
        else:
            colors = ["#2ecc71", "#e74c3c"]  # Worse (red)
    else:
        # For metrics where higher is better
        if bpo_value > previous_value:
            colors = ["#e74c3c", "#2ecc71"]  # Improvement (green)
        else:
            colors = ["#2ecc71", "#e74c3c"]  # Worse (red)

    bars = ax.bar(
        categories, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.2f}" if isinstance(value, (int, float)) else "N/A",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Calculate and display percentage change
    if previous_value and previous_value != 0:
        pct_change = ((bpo_value - previous_value) / previous_value) * 100
        change_text = f"Change: {pct_change:+.1f}%"
        # AHT: increase = bad (red), decrease = good (green)
        is_aht = "AHT" in metric_name or "Handle Time" in metric_name
        change_color = "#e74c3c" if (is_aht and pct_change > 0) or (not is_aht and pct_change < 0) else "#2ecc71"
        ax.text(
            0.5,
            0.95,
            change_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=change_color,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5, edgecolor=change_color),
        )

    ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
    ax.set_title(f"{metric_name} Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig


def create_improvement_summary_chart(improvements: Dict) -> plt.Figure:
    """Create a chart showing percentage improvements for key metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = []
    pct_changes = []
    colors_list = []

    for metric, pct_change in improvements.items():
        if pct_change is not None:
            metrics.append(metric.replace("_", " ").title())
            pct_changes.append(pct_change)
            # AHT: lower is better—increase (pct>0) = bad/red, decrease (pct<0) = good/green
            # Other metrics: higher is better—increase = good/green, decrease = bad/red
            is_aht = "aht" in metric.lower() or "handle time" in metric.lower()
            if is_aht:
                colors_list.append("#e74c3c" if pct_change > 0 else "#2ecc71")
            else:
                colors_list.append("#2ecc71" if pct_change > 0 else "#e74c3c")

    if not metrics:
        ax.text(
            0.5,
            0.5,
            "No improvement data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        return fig

    bars = ax.barh(
        metrics,
        pct_changes,
        color=colors_list,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for bar, value in zip(bars, pct_changes):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:+.1f}%",
            ha="left" if width > 0 else "right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Percentage Change (%)", fontsize=12, fontweight="bold")
    ax.set_title("Key Performance Improvements", fontsize=14, fontweight="bold", pad=20)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig


def create_distribution_comparison(
    previous_scores: pd.Series, bpo_scores: pd.Series
) -> plt.Figure:
    """
    Create a histogram/box plot comparing score distributions.
    Histogram uses normalized % of calls per bin (not raw counts) for fair
    comparison across unequal cohort sizes. Shared bin edges across cohorts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    prev_vals = previous_scores.dropna().values
    bpo_vals = bpo_scores.dropna().values

    # Shared bin edges (0-100 for QA scores)
    bins = np.linspace(0, 100, 21)

    if len(prev_vals) > 0:
        prev_weights = np.ones_like(prev_vals, dtype=float) / len(prev_vals) * 100
        ax1.hist(
            prev_vals,
            bins=bins,
            alpha=0.6,
            label="Previous Center",
            color="#e74c3c",
            edgecolor="black",
            weights=prev_weights,
        )
    if len(bpo_vals) > 0:
        bpo_weights = np.ones_like(bpo_vals, dtype=float) / len(bpo_vals) * 100
        ax1.hist(
            bpo_vals,
            bins=bins,
            alpha=0.6,
            label="BPO Centers",
            color="#2ecc71",
            edgecolor="black",
            weights=bpo_weights,
        )
    ax1.set_xlabel("QA Score", fontsize=12, fontweight="bold")
    ax1.set_ylabel("% of Calls", fontsize=12, fontweight="bold")
    ax1.set_title("Score Distribution Comparison (Normalized)", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Box plot comparison (filter out empty arrays)
    data_to_plot = [prev_vals, bpo_vals]
    data_to_plot = [d for d in data_to_plot if len(d) > 0]
    tick_labels = ["Previous\nCenter", "BPO\nCenters"]
    if len(data_to_plot) == 2:
        labels = tick_labels
    elif len(data_to_plot) == 1:
        labels = [tick_labels[0] if len(prev_vals) > 0 else tick_labels[1]]
    else:
        labels = []
    if data_to_plot:
        bp = ax2.boxplot(
            data_to_plot,
            tick_labels=labels,
            patch_artist=True,
            showmeans=True,
        )
        # Color boxes by cohort
        if "boxes" in bp:
            colors = ["#e74c3c", "#2ecc71"]
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor(colors[i % 2])
    else:
        ax2.text(
            0.5, 0.5, "No score data available",
            ha="center", va="center", transform=ax2.transAxes,
        )
    ax2.set_ylabel("QA Score", fontsize=12, fontweight="bold")
    ax2.set_title("Score Distribution (Box Plot)", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def create_quality_tier_chart(previous_metrics: Dict, bpo_metrics: Dict) -> plt.Figure:
    """Create pie charts comparing quality tier distributions.
    Pass = Excellent + Good (score >= PASS_THRESHOLD).
    Passing is defined as a QA score ≥ 70%."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Quality Distribution | Passing = QA score ≥ {PASS_THRESHOLD}% (Excellent + Good)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    # Previous Center tiers
    prev_labels = [
        "Excellent\n(90-100)",
        "Good\n(70-89)",
        "Fair\n(60-69)",
        "Poor\n(<60)",
    ]
    prev_sizes = [
        previous_metrics.get("excellent_pct", 0),
        previous_metrics.get("good_pct", 0),
        previous_metrics.get("fair_pct", 0),
        previous_metrics.get("poor_pct", 0),
    ]
    prev_colors = ["#27ae60", "#2ecc71", "#f39c12", "#e74c3c"]

    ax1.pie(
        prev_sizes,
        labels=prev_labels,
        colors=prev_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    ax1.set_title(
        "Previous Contact Center\nQuality Distribution", fontsize=14, fontweight="bold"
    )

    # BPO Centers tiers
    bpo_labels = [
        "Excellent\n(90-100)",
        "Good\n(70-89)",
        "Fair\n(60-69)",
        "Poor\n(<60)",
    ]
    bpo_sizes = [
        bpo_metrics.get("excellent_pct", 0),
        bpo_metrics.get("good_pct", 0),
        bpo_metrics.get("fair_pct", 0),
        bpo_metrics.get("poor_pct", 0),
    ]

    ax2.pie(
        bpo_sizes,
        labels=bpo_labels,
        colors=prev_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    ax2.set_title("BPO Centers\nQuality Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def create_trend_chart(bpo_df: pd.DataFrame) -> plt.Figure:
    """Create a time series chart showing Monthly Performance Trends."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if "Call Date" in bpo_df.columns and "QA Score" in bpo_df.columns:
        df_copy = bpo_df[["Call Date", "QA Score"]].copy()
        df_copy = df_copy.dropna()
        df_copy["Call Date"] = pd.to_datetime(df_copy["Call Date"])
        df_copy = df_copy.sort_values("Call Date")

        # Monthly aggregation only
        df_copy["year_month"] = df_copy["Call Date"].dt.to_period("M")
        monthly_avg = df_copy.groupby("year_month")["QA Score"].mean()

        # Convert period to datetime for plotting
        month_dates = [pd.Period.to_timestamp(m) for m in monthly_avg.index]

        ax.plot(
            month_dates,
            monthly_avg.values,
            marker="o",
            linewidth=2,
            markersize=8,
            color="#2ecc71",
            label="Monthly Avg QA Score",
        )
        ax.fill_between(month_dates, monthly_avg.values, alpha=0.3, color="#2ecc71")

        # Add trend line
        if len(monthly_avg) > 1:
            z = np.polyfit(range(len(monthly_avg)), monthly_avg.values, 1)
            p = np.poly1d(z)
            ax.plot(
                month_dates,
                p(range(len(monthly_avg))),
                "--",
                color="#34495e",
                linewidth=2,
                label="Trend Line",
                alpha=0.7,
            )

        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("QA Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Monthly Performance Trends",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
    else:
        ax.text(
            0.5,
            0.5,
            "No trend data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    plt.tight_layout()
    return fig


def create_rubric_improvement_chart(improvements: List[Dict]) -> plt.Figure:
    """Create a horizontal bar chart showing top rubric code improvements."""
    if not improvements:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No rubric improvements to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        return fig

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top 10 improvements
    top_improvements = improvements[:10]

    codes = [imp["code"] for imp in top_improvements]
    improvement_pcts = [imp["improvement_pct"] for imp in top_improvements]

    colors = ["#2ecc71"] * len(codes)
    bars = ax.barh(
        codes,
        improvement_pcts,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, improvement_pcts)):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{pct:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        # Add previous and BPO rates
        prev_rate = top_improvements[i]["previous_rate"]
        bpo_rate = top_improvements[i]["bpo_rate"]
        ax.text(
            0,
            bar.get_y() + bar.get_height() / 2.0,
            f"  {prev_rate:.1f}% → {bpo_rate:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            style="italic",
        )

    ax.set_xlabel("Improvement (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rubric Code", fontsize=12, fontweight="bold")
    ax.set_title(
        "Top Rubric Code Improvements\n(Reduction in Failure Rate)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def create_reason_comparison_chart(
    previous_reasons: Dict, bpo_reasons: Dict
) -> plt.Figure:
    """Create side-by-side comparison chart for call reasons (counts)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Top 10 Call Reasons (counts). Note: Percentages use denominator = calls with reason data only.",
        fontsize=10,
        fontweight="bold",
        y=1.02,
    )

    # Previous center reasons (top 10)
    if previous_reasons:
        prev_series = pd.Series(previous_reasons).head(10)
        ax1.barh(
            range(len(prev_series)), prev_series.values, color="#e74c3c", alpha=0.7
        )
        ax1.set_yticks(range(len(prev_series)))
        ax1.set_yticklabels(prev_series.index, fontsize=9)
        ax1.set_xlabel("Number of Calls", fontsize=11)
        ax1.set_title(
            "Previous Contact Center\nTop 10 Call Reasons",
            fontsize=12,
            fontweight="bold",
        )
        ax1.invert_yaxis()
        ax1.grid(axis="x", alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title(
            "Previous Contact Center\nCall Reasons", fontsize=12, fontweight="bold"
        )

    # BPO Centers reasons (top 10)
    if bpo_reasons:
        bpo_series = pd.Series(bpo_reasons).head(10)
        ax2.barh(range(len(bpo_series)), bpo_series.values, color="#2ecc71", alpha=0.7)
        ax2.set_yticks(range(len(bpo_series)))
        ax2.set_yticklabels(bpo_series.index, fontsize=9)
        ax2.set_xlabel("Number of Calls", fontsize=11)
        ax2.set_title(
            "BPO Centers\nTop 10 Call Reasons", fontsize=12, fontweight="bold"
        )
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("BPO Centers\nCall Reasons", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def create_outcome_comparison_chart(
    previous_outcomes: Dict, bpo_outcomes: Dict
) -> plt.Figure:
    """Create side-by-side comparison chart for call outcomes (counts)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Top 10 Call Outcomes (counts). Note: Percentages use denominator = calls with outcome data only.",
        fontsize=10,
        fontweight="bold",
        y=1.02,
    )

    # Previous center outcomes (top 10)
    if previous_outcomes:
        prev_series = pd.Series(previous_outcomes).head(10)
        ax1.barh(
            range(len(prev_series)), prev_series.values, color="#e74c3c", alpha=0.7
        )
        ax1.set_yticks(range(len(prev_series)))
        ax1.set_yticklabels(prev_series.index, fontsize=9)
        ax1.set_xlabel("Number of Calls", fontsize=11)
        ax1.set_title(
            "Previous Contact Center\nTop 10 Outcomes", fontsize=12, fontweight="bold"
        )
        ax1.invert_yaxis()
        ax1.grid(axis="x", alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title(
            "Previous Contact Center\nOutcomes", fontsize=12, fontweight="bold"
        )

    # BPO Centers outcomes (top 10)
    if bpo_outcomes:
        bpo_series = pd.Series(bpo_outcomes).head(10)
        ax2.barh(range(len(bpo_series)), bpo_series.values, color="#2ecc71", alpha=0.7)
        ax2.set_yticks(range(len(bpo_series)))
        ax2.set_yticklabels(bpo_series.index, fontsize=9)
        ax2.set_xlabel("Number of Calls", fontsize=11)
        ax2.set_title("BPO Centers\nTop 10 Outcomes", fontsize=12, fontweight="bold")
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("BPO Centers\nOutcomes", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def create_product_comparison_chart(
    previous_products: Dict, bpo_products: Dict
) -> plt.Figure:
    """Create side-by-side comparison chart for products discussed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Previous center products (top 10)
    if previous_products:
        prev_series = pd.Series(previous_products).head(10)
        ax1.barh(
            range(len(prev_series)), prev_series.values, color="#e74c3c", alpha=0.7
        )
        ax1.set_yticks(range(len(prev_series)))
        ax1.set_yticklabels(prev_series.index, fontsize=9)
        ax1.set_xlabel("Number of Mentions", fontsize=11)
        ax1.set_title(
            "Previous Contact Center\nTop 10 Products Discussed",
            fontsize=12,
            fontweight="bold",
        )
        ax1.invert_yaxis()
        ax1.grid(axis="x", alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "No products found",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title(
            "Previous Contact Center\nProducts Discussed",
            fontsize=12,
            fontweight="bold",
        )

    # BPO Centers products (top 10)
    if bpo_products:
        bpo_series = pd.Series(bpo_products).head(10)
        ax2.barh(range(len(bpo_series)), bpo_series.values, color="#2ecc71", alpha=0.7)
        ax2.set_yticks(range(len(bpo_series)))
        ax2.set_yticklabels(bpo_series.index, fontsize=9)
        ax2.set_xlabel("Number of Mentions", fontsize=11)
        ax2.set_title(
            "BPO Centers\nTop 10 Products Discussed", fontsize=12, fontweight="bold"
        )
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No products found",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("BPO Centers\nProducts Discussed", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def create_agent_performance_heatmap(
    bpo_df: pd.DataFrame,
    total_calls_bpo: Optional[int] = None,
) -> plt.Figure:
    """
    Create a heatmap showing agent performance across multiple metrics.
    Call_Count = nunique(call_id) per agent - displayed as integer with comma formatting.
    """
    if bpo_df is None or len(bpo_df) == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No agent data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    if "Agent" not in bpo_df.columns or "QA Score" not in bpo_df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "Agent or QA Score column not found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    df_work = bpo_df.copy()
    df_work["Agent"] = df_work["Agent"].fillna("Unknown Agent")
    df_work.loc[df_work["Agent"].astype(str).str.strip() == "", "Agent"] = "Unknown Agent"

    call_id_col = "call_id" if "call_id" in df_work.columns else None
    if call_id_col is None:
        for c in df_work.columns:
            if c.lower() in ("call_id", "call id", "callid"):
                call_id_col = c
                break

    def _agent_pass_rate(group):
        scores = group["QA Score"].dropna()
        if len(scores) == 0:
            return np.nan
        return (scores >= PASS_THRESHOLD).sum() / len(scores) * 100

    if call_id_col:
        call_counts = df_work.groupby("Agent")[call_id_col].nunique()
    else:
        call_counts = df_work.groupby("Agent").size()

    agent_metrics = df_work.groupby("Agent").agg({"QA Score": "mean"}).reset_index()
    agent_metrics = agent_metrics.rename(columns={"QA Score": "Avg_Score"})
    agent_metrics["Call_Count"] = agent_metrics["Agent"].map(call_counts).fillna(0).astype(int)
    agent_metrics["Pass_Rate"] = df_work.groupby("Agent", group_keys=False).apply(
        _agent_pass_rate, include_groups=False
    ).values

    total_calls = total_calls_bpo if total_calls_bpo is not None else int(agent_metrics["Call_Count"].sum())

    agent_metrics = agent_metrics.sort_values("Avg_Score", ascending=False)

    # Data for heatmap: Avg_Score, Pass_Rate (0-100), Call_Count (use % share for color scale)
    heatmap_color = agent_metrics[["Avg_Score", "Pass_Rate"]].copy()
    heatmap_color["Call_Count"] = (
        (agent_metrics["Call_Count"] / total_calls * 100) if total_calls > 0
        else pd.Series([0] * len(agent_metrics), index=agent_metrics.index)
    )
    heatmap_color = heatmap_color.fillna(0)
    heatmap_color.index = agent_metrics["Agent"]

    # Custom annot: display actual Call_Count as integer with comma (not %)
    annot_values = []
    for _, row in agent_metrics.iterrows():
        annot_values.append([
            f"{row['Avg_Score']:.1f}" if pd.notna(row["Avg_Score"]) else "—",
            f"{row['Pass_Rate']:.1f}" if pd.notna(row["Pass_Rate"]) else "—",
            f"{int(row['Call_Count']):,}",
        ])
    annot_df = pd.DataFrame(
        np.array(annot_values).T,
        index=["Avg_Score", "Pass_Rate", "Call_Count"],
        columns=agent_metrics["Agent"].tolist(),
    )

    fig, ax = plt.subplots(figsize=(14, max(8, len(agent_metrics) * 0.5)))
    sns.heatmap(
        heatmap_color.T,
        annot=annot_df,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Score / Share (0-100)"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Agent Performance Heatmap\n(Avg Score, Pass Rate %, Call Count)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Agent", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def create_monthly_trend_analysis(bpo_df: pd.DataFrame) -> plt.Figure:
    """Create a comprehensive monthly trend analysis with multiple metrics."""
    if bpo_df is None or len(bpo_df) == 0 or "Call Date" not in bpo_df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(
            0.5,
            0.5,
            "No date data available for trend analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Check for required QA Score column
    if "QA Score" not in bpo_df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(
            0.5,
            0.5,
            "QA Score column not found for trend analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Ensure Call Date is datetime
    bpo_df = bpo_df.copy()
    bpo_df["Call Date"] = pd.to_datetime(bpo_df["Call Date"], errors="coerce")
    bpo_df = bpo_df[bpo_df["Call Date"].notna()]

    if len(bpo_df) == 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(
            0.5,
            0.5,
            "No valid date data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Monthly aggregation only; pass = QA score >= PASS_THRESHOLD
    bpo_df["Month"] = bpo_df["Call Date"].dt.to_period("M")

    def _monthly_pass_rate(group):
        scores = group["QA Score"].dropna()
        if len(scores) == 0:
            return np.nan
        return (scores >= PASS_THRESHOLD).sum() / len(scores) * 100

    monthly_metrics = bpo_df.groupby("Month").agg(
        {"QA Score": ["mean", "count"]}
    ).reset_index()
    monthly_metrics.columns = ["Month", "Avg_Score", "Call_Count"]
    monthly_metrics["Pass_Rate"] = bpo_df.groupby("Month", group_keys=False).apply(
        _monthly_pass_rate, include_groups=False
    ).values

    # Ensure Avg_Score exists (should always exist if QA Score was validated, but check defensively)
    if "Avg_Score" not in monthly_metrics.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(
            0.5,
            0.5,
            "Average score data not available for trend analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Convert Month to string for plotting
    monthly_metrics["Month_Str"] = monthly_metrics["Month"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Monthly Performance Trends", fontsize=16, fontweight="bold", y=0.995)

    # 1. Average QA Score Trend
    ax1 = axes[0, 0]
    x_indices = range(len(monthly_metrics))
    ax1.plot(
        x_indices,
        monthly_metrics["Avg_Score"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
    )
    ax1.fill_between(
        x_indices,
        monthly_metrics["Avg_Score"],
        alpha=0.3,
        color="#2E86AB",
    )
    ax1.set_title("Average QA Score Trend", fontsize=12, fontweight="bold")
    ax1.set_ylabel("QA Score", fontsize=10)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(monthly_metrics["Month_Str"], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0, top=100)

    # 2. Pass Rate Trend
    ax2 = axes[0, 1]
    x_indices = range(len(monthly_metrics))
    # For plotting, matplotlib handles NaN/NA by skipping those points
    # This is correct - we don't want to plot 0% when there's no data
    pass_rate_plot = monthly_metrics["Pass_Rate"]
    ax2.plot(
        x_indices,
        pass_rate_plot,
        marker="s",
        linewidth=2,
        markersize=8,
        color="#A23B72",
    )
    # Only fill between where we have valid data (non-NA)
    valid_mask = pass_rate_plot.notna()
    if valid_mask.any():
        ax2.fill_between(
            [i for i, valid in zip(x_indices, valid_mask) if valid],
            pass_rate_plot[valid_mask],
            alpha=0.3,
            color="#A23B72",
        )
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(monthly_metrics["Month_Str"], rotation=45, ha="right")
    ax2.set_title("Pass Rate Trend", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Pass Rate (%)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0, top=100)

    # 3. Call Volume Trend
    ax3 = axes[1, 0]
    bars = ax3.bar(
        monthly_metrics["Month_Str"],
        monthly_metrics["Call_Count"],
        color="#F18F01",
        alpha=0.7,
    )
    ax3.set_title("Call Volume Trend", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Number of Calls", fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. Combined Score and Pass Rate
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(
        monthly_metrics["Month_Str"],
        monthly_metrics["Avg_Score"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        label="Avg Score",
    )
    # For plotting, matplotlib handles NaN/NA by skipping those points
    # This is correct - we don't want to plot 0% when there's no data
    pass_rate_combined = monthly_metrics["Pass_Rate"]
    line2 = ax4_twin.plot(
        monthly_metrics["Month_Str"],
        pass_rate_combined,
        marker="s",
        linewidth=2,
        markersize=8,
        color="#A23B72",
        label="Pass Rate",
    )
    ax4.set_title("Score & Pass Rate Combined", fontsize=12, fontweight="bold")
    ax4.set_ylabel("QA Score", fontsize=10, color="#2E86AB")
    ax4_twin.set_ylabel("Pass Rate (%)", fontsize=10, color="#A23B72")
    ax4.tick_params(axis="y", labelcolor="#2E86AB")
    ax4_twin.tick_params(axis="y", labelcolor="#A23B72")
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Combine legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax4.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    return fig


def create_top_failure_reasons_chart(bpo_df: pd.DataFrame) -> plt.Figure:
    """Create a chart showing top failure reasons with impact analysis.
    Supports both wide format (columns 1.1.0, 1.2.0, etc.) and Rubric Details dict column."""
    if bpo_df is None or len(bpo_df) == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Find rubric columns (wide format: 1.1.0, 1.2.0, etc.)
    rubric_columns = [
        col
        for col in bpo_df.columns
        if re.match(r"^\d+\.\d+\.\d+$", str(col)) and "__reason" not in str(col).lower()
    ]

    # Rubric Details column (dict per row: {"1.1.0": {"status": "Pass/Fail/N/A"}, ...})
    rubric_details_col = None
    for col in bpo_df.columns:
        if str(col).strip().lower() == "rubric details":
            rubric_details_col = col
            break

    failure_data = []
    total_calls = len(bpo_df)

    if rubric_columns:
        # Wide format: one column per rubric code
        for col in rubric_columns:
            failures = 0
            for value in bpo_df[col]:
                value_str = str(value).strip().upper() if pd.notna(value) else ""
                if value is False or value_str == "FALSE" or value_str == "F":
                    failures += 1

            if failures > 0:
                failure_pct = (failures / total_calls) * 100
                failure_data.append(
                    {"Code": col, "Failures": failures, "Failure_Pct": failure_pct}
                )
    elif rubric_details_col:
        # Rubric Details format: dict with code -> {status, note}
        failure_counts = {}
        for _, row in bpo_df.iterrows():
            details = row.get(rubric_details_col)
            if not isinstance(details, dict):
                continue
            for code, item in details.items():
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status", "")).strip().lower()
                if status in ("fail", "false"):
                    failure_counts[code] = failure_counts.get(code, 0) + 1

        for code, failures in failure_counts.items():
            if failures > 0 and re.match(r"^\d+\.\d+\.\d+$", str(code)):
                failure_pct = (failures / total_calls) * 100
                failure_data.append(
                    {"Code": code, "Failures": failures, "Failure_Pct": failure_pct}
                )
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        ax.text(
            0.5,
            0.6,
            "Rubric breakdown unavailable for this dataset",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.45,
            "Missing rubric columns (e.g. 1.1.0, 1.2.0) or Rubric Details column.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        return fig

    if not failure_data:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No failures found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    failure_df = pd.DataFrame(failure_data)
    failure_df = failure_df.sort_values("Failures", ascending=False).head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Top 10 Rubric Failure Codes - Impact Analysis", fontsize=14, fontweight="bold"
    )

    # Left: Bar chart of failure counts
    bars = ax1.barh(
        failure_df["Code"], failure_df["Failures"], color="#C73E1D", alpha=0.7
    )
    ax1.set_xlabel("Number of Failures", fontsize=11, fontweight="bold")
    ax1.set_title("Failure Count", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Right: Percentage impact
    bars2 = ax2.barh(
        failure_df["Code"], failure_df["Failure_Pct"], color="#FF6B35", alpha=0.7
    )
    ax2.set_xlabel("Failure Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Impact on Total Calls", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def create_bpo_only_kpi_dashboard(bpo_metrics: Dict) -> plt.Figure:
    """Create a KPI dashboard showing only BPO Centers metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Key Performance Indicators (KPI) Dashboard - BPO Centers",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Prepare data
    metrics_to_show = [
        ("avg_qa_score", "Average QA Score", "Score", 0, 100),
        ("pass_rate", "Pass Rate", "%", 0, 100),
        ("avg_aht", "Average Handle Time", "min", None, None),
        ("total_calls", "Total Call Volume", "calls", None, None),
        ("calls_per_day", "Calls Per Day", "calls/day", None, None),
        ("excellent_pct", "Excellent Scores (90+)", "%", 0, 100),
    ]

    for idx, (metric_key, metric_name, unit, ymin, ymax) in enumerate(metrics_to_show):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        bpo_val = bpo_metrics.get(metric_key)

        if bpo_val is None:
            ax.text(
                0.5,
                0.5,
                f"{metric_name}\nData not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title(metric_name, fontsize=11, fontweight="bold")
            continue

        # Create single bar
        bars = ax.bar(
            ["BPO Centers"],
            [bpo_val],
            color="#28A745",
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value label
        height = bars[0].get_height()
        label = f"{bpo_val:.1f}" if isinstance(bpo_val, float) else f"{int(bpo_val)}"
        ax.text(
            bars[0].get_x() + bars[0].get_width() / 2.0,
            height,
            f"{label} {unit}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3, axis="y")

    fig.text(
        0.5,
        0.01,
        PASS_DEFINITION_TEXT,
        ha="center",
        fontsize=9,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def create_kpi_dashboard(previous_metrics: Dict, bpo_metrics: Dict) -> plt.Figure:
    """Create a KPI dashboard with key metrics comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Key Performance Indicators (KPI) Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Prepare data
    metrics_to_compare = [
        ("avg_qa_score", "Average QA Score", "Score", 0, 100),
        ("pass_rate", "Pass Rate", "%", 0, 100),
        ("avg_aht", "Average Handle Time", "min", None, None),
        ("total_calls", "Total Call Volume", "calls", None, None),
        ("calls_per_day", "Calls Per Day", "calls/day", None, None),
        ("excellent_pct", "Excellent Scores (90+)", "%", 0, 100),
    ]

    for idx, (metric_key, metric_name, unit, ymin, ymax) in enumerate(
        metrics_to_compare
    ):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        prev_val = previous_metrics.get(metric_key)
        bpo_val = bpo_metrics.get(metric_key)

        if prev_val is None or bpo_val is None:
            ax.text(
                0.5,
                0.5,
                f"{metric_name}\nData not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title(metric_name, fontsize=11, fontweight="bold")
            continue

        # Create comparison bars
        categories = ["Previous\nCenter", "BPO\nCenters"]
        values = [prev_val, bpo_val]
        colors = ["#6C757D", "#28A745"]

        bars = ax.bar(
            categories,
            values,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{label} {unit}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Calculate and show improvement
        if prev_val != 0:
            improvement = ((bpo_val - prev_val) / prev_val) * 100
            # AHT: increase = bad (red), decrease = good (green)
            is_aht = metric_key == "avg_aht" or "handle time" in metric_name.lower()
            if is_aht:
                improvement_color = "#DC3545" if improvement > 0 else "#28A745"
            else:
                improvement_color = "#28A745" if improvement > 0 else "#DC3545"
            improvement_text = f"{improvement:+.1f}%"
            ax.text(
                0.5,
                0.95,
                improvement_text,
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                color=improvement_color,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=improvement_color,
                    linewidth=2,
                ),
            )

        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3, axis="y")

    fig.text(
        0.5,
        0.01,
        PASS_DEFINITION_TEXT,
        ha="center",
        fontsize=9,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def create_agent_leaderboard(
    bpo_df: pd.DataFrame,
    total_calls: Optional[int] = None,
) -> plt.Figure:
    """
    Create an agent leaderboard with rankings.
    Uses nunique(call_id) for Call_Count. Adds Other/Unknown row when showing Top N
    so displayed totals reconcile to total_calls_bpo.
    """
    if bpo_df is None or len(bpo_df) == 0 or "Agent" not in bpo_df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No agent data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    if total_calls is None or total_calls <= 0:
        total_calls = len(bpo_df)  # Fallback for legacy callers

    df_work = bpo_df.copy()
    df_work["Agent"] = df_work["Agent"].fillna("Unknown Agent")
    df_work.loc[df_work["Agent"].astype(str).str.strip() == "", "Agent"] = "Unknown Agent"

    call_id_col = "call_id" if "call_id" in df_work.columns else None
    if call_id_col is None:
        for c in df_work.columns:
            if c.lower() in ("call_id", "call id", "callid"):
                call_id_col = c
                break
    if call_id_col is None:
        call_id_col = "Call ID" if "Call ID" in df_work.columns else None

    if "QA Score" not in df_work.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "QA Score column not found for leaderboard",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    def _agent_pass_rate(group):
        scores = group["QA Score"].dropna()
        if len(scores) == 0:
            return np.nan
        return (scores >= PASS_THRESHOLD).sum() / len(scores) * 100

    # Use nunique(call_id) for true call counts
    if call_id_col:
        call_counts = df_work.groupby("Agent")[call_id_col].nunique()
    else:
        call_counts = df_work.groupby("Agent").size()

    agg_df = df_work.groupby("Agent").agg({"QA Score": "mean"}).reset_index()
    agg_df = agg_df.rename(columns={"QA Score": "Avg_Score"})
    agg_df["Call_Count"] = agg_df["Agent"].map(call_counts).fillna(0).astype(int)
    pass_rates = df_work.groupby("Agent", group_keys=False).apply(
        _agent_pass_rate, include_groups=False
    )
    agg_df["Pass_Rate"] = agg_df["Agent"].map(pass_rates).values

    agg_df = agg_df.sort_values("Avg_Score", ascending=False)
    agent_sum_all = int(agg_df["Call_Count"].sum())
    if abs(agent_sum_all - total_calls) >= 1:
        raise ValueError(
            f"Agent leaderboard does not reconcile to cohort totals: "
            f"sum(agent_call_counts)={agent_sum_all} != total_calls_bpo={total_calls}"
        )

    top_agents = agg_df.head(AGENT_LEADERBOARD_TOP_N)
    top_n_sum = int(top_agents["Call_Count"].sum())
    other_count = total_calls - top_n_sum

    # Add Other/Unknown row so totals reconcile visually
    if other_count > 0:
        other_row = pd.DataFrame([{
            "Agent": "Other/Unknown",
            "Avg_Score": np.nan,
            "Call_Count": other_count,
            "Pass_Rate": np.nan,
        }])
        display_df = pd.concat([top_agents, other_row], ignore_index=True)
    else:
        display_df = top_agents.copy()

    display_df["Rank"] = range(1, len(display_df) + 1)

    fig, ax = plt.subplots(figsize=(14, max(8, len(display_df) * 0.5)))

    y_pos = np.arange(len(display_df))
    scores_plot = display_df["Avg_Score"].fillna(0).values
    bar_colors = [
        [0.9, 0.9, 0.9, 0.8] if agent == "Other/Unknown"
        else plt.cm.RdYlGn(score / 100)
        for agent, score in zip(display_df["Agent"], scores_plot)
    ]
    ax.barh(y_pos, scores_plot, color=bar_colors, alpha=0.8)

    for i, (_, row) in enumerate(display_df.iterrows()):
        score = row["Avg_Score"]
        score_val = float(score) if pd.notna(score) else 0.0
        score_str = f"{score:.1f}" if pd.notna(score) else "—"
        ax.text(score_val + 1, i, score_str, va="center", fontsize=9, fontweight="bold")
        count_str = f"{int(row['Call_Count']):,}"
        ax.text(
            score_val + 1,
            i - 0.25,
            f"({count_str} calls)",
            va="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{int(r['Rank'])} {r['Agent']}" for _, r in display_df.iterrows()])
    ax.set_xlabel("Average QA Score", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Agent Performance Leaderboard (Top {AGENT_LEADERBOARD_TOP_N} + Other)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


def create_top_call_reasons_chart(bpo_df: pd.DataFrame) -> plt.Figure:
    """Create a chart showing top call reasons."""
    if bpo_df is None or len(bpo_df) == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Check for Reason column (case-insensitive)
    reason_col = None
    for col in bpo_df.columns:
        if col.lower() == "reason":
            reason_col = col
            break

    if reason_col is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No reason data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Canonicalize and aggregate (same pipeline as calculate_reason_metrics)
    canonical = bpo_df[reason_col].fillna("").apply(canonicalize_label)
    reason_counts_raw = canonical[canonical != ""].value_counts()
    dist = {label_to_display(k): v for k, v in reason_counts_raw.items()}
    reason_counts = pd.Series(dist).sort_values(ascending=False).head(10)
    calls_with_reason = int(reason_counts_raw.sum()) if len(reason_counts_raw) > 0 else 0
    if len(reason_counts) == 0 or calls_with_reason == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No reason data found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Top 10 Call Reasons\n(% of calls with reason data; denominator excludes calls without reason)",
        fontsize=12,
        fontweight="bold",
    )

    # Left: Count chart
    bars1 = ax1.barh(
        reason_counts.index, reason_counts.values, color="#3498DB", alpha=0.7
    )
    ax1.set_xlabel("Number of Calls", fontsize=11, fontweight="bold")
    ax1.set_title("Call Count", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Right: Percentage chart (% of calls with reason data only)
    reason_pct = (reason_counts / calls_with_reason * 100).round(1)
    bars2 = ax2.barh(reason_pct.index, reason_pct.values, color="#9B59B6", alpha=0.7)
    ax2.set_xlabel("% of Calls", fontsize=11, fontweight="bold")
    ax2.set_title("Percentage Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def create_top_call_outcomes_chart(bpo_df: pd.DataFrame) -> plt.Figure:
    """Create a chart showing top call outcomes."""
    if bpo_df is None or len(bpo_df) == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Check for Outcome column (case-insensitive)
    outcome_col = None
    for col in bpo_df.columns:
        if col.lower() == "outcome":
            outcome_col = col
            break

    if outcome_col is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No outcome data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Canonicalize and aggregate (same pipeline as calculate_outcome_metrics)
    canonical = bpo_df[outcome_col].fillna("").apply(canonicalize_label)
    outcome_counts_raw = canonical[canonical != ""].value_counts()
    dist = {label_to_display(k): v for k, v in outcome_counts_raw.items()}
    outcome_counts = pd.Series(dist).sort_values(ascending=False).head(10)
    calls_with_outcome = int(outcome_counts_raw.sum()) if len(outcome_counts_raw) > 0 else 0
    if len(outcome_counts) == 0 or calls_with_outcome == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No outcome data found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Top 10 Call Outcomes\n(% of calls with outcome data; denominator excludes calls without outcome)",
        fontsize=12,
        fontweight="bold",
    )

    # Left: Count chart
    bars1 = ax1.barh(
        outcome_counts.index, outcome_counts.values, color="#E74C3C", alpha=0.7
    )
    ax1.set_xlabel("Number of Calls", fontsize=11, fontweight="bold")
    ax1.set_title("Call Count", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Right: Percentage chart (% of calls with outcome data only)
    outcome_pct = (outcome_counts / calls_with_outcome * 100).round(1)
    bars2 = ax2.barh(outcome_pct.index, outcome_pct.values, color="#F39C12", alpha=0.7)
    ax2.set_xlabel("% of Calls", fontsize=11, fontweight="bold")
    ax2.set_title("Percentage Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def create_top_products_chart(bpo_df: pd.DataFrame) -> plt.Figure:
    """Create a chart showing top products discussed."""
    if bpo_df is None or len(bpo_df) == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Extract products from text fields
    product_data = []
    text_fields = []

    for col in bpo_df.columns:
        col_lower = col.lower()
        if col_lower in ["summary", "reason", "outcome"]:
            text_fields.append(col)

    if not text_fields:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No text fields available for product extraction",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Extract products from all text fields
    for idx, row in bpo_df.iterrows():
        combined_text = " ".join(
            [str(row.get(field, "") or "") for field in text_fields]
        )
        products = extract_products_from_text(combined_text)
        product_data.extend(products)

    if len(product_data) == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "No products found in call data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Get top products
    product_counts = pd.Series(product_data).value_counts().head(10)
    total_mentions = len(product_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Top 10 Products Discussed", fontsize=14, fontweight="bold")

    # Left: Count chart
    bars1 = ax1.barh(
        product_counts.index, product_counts.values, color="#27AE60", alpha=0.7
    )
    ax1.set_xlabel("Number of Mentions", fontsize=11, fontweight="bold")
    ax1.set_title("Mention Count", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Right: Percentage chart
    product_pct = (product_counts / total_mentions * 100).round(1)
    bars2 = ax2.barh(product_pct.index, product_pct.values, color="#16A085", alpha=0.7)
    ax2.set_xlabel("Percentage of Mentions (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Percentage Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def generate_pdf_report(
    previous_metrics: Dict,
    bpo_metrics: Dict,
    output_path: Path,
    bpo_start_date: datetime,
    previous_data: pd.DataFrame = None,
    bpo_data: pd.DataFrame = None,
    cohort_config: Optional[Dict[str, Any]] = None,
    total_calls_bpo: Optional[int] = None,
):
    """Generate multi-page PDF comparison report."""
    print(f"\n📄 Generating PDF report: {output_path}")

    with PdfPages(output_path) as pdf:
        # Page 1: Title Page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")

        has_previous_data = previous_data is not None and len(previous_data) > 0
        title_text = "Contact Center Performance Report"
        subtitle_text = (
            "BPO Centers Performance Analysis"
            if not has_previous_data
            else "Previous Contact Center vs BPO Centers"
        )
        date_text = (
            f"BPO Centers Data: From {bpo_start_date.strftime('%B %d, %Y')} onwards"
        )
        generated_text = (
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        )

        ax.text(
            0.5,
            0.7,
            title_text,
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.6,
            subtitle_text,
            ha="center",
            va="center",
            fontsize=18,
            style="italic",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.4,
            date_text,
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.3,
            generated_text,
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Cohort Definition
        if cohort_config:
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            cohort_lines = ["COHORT DEFINITION", ""]
            cohort_lines.append(
                PASS_DEFINITION_TEXT
            )
            cohort_lines.append("")
            for name, cfg in cohort_config.items():
                cohort_lines.append(f"{name}:")
                cohort_lines.append(f"  Start Date: {cfg.get('start_date', 'N/A')}")
                cohort_lines.append(f"  End Date: {cfg.get('end_date', 'N/A')}")
                cohort_lines.append(f"  Days in Period: {cfg.get('days_in_period', 'N/A')}")
                cohort_lines.append(f"  Total Calls: {cfg.get('total_calls', 'N/A')}")
                cpd = cfg.get("calls_per_day")
                cohort_lines.append(
                    f"  Calls Per Day: {cpd:.2f}" if cpd is not None else "  Calls Per Day: N/A"
                )
                cohort_lines.append("")
            ax.text(0.1, 0.95, "\n".join(cohort_lines), transform=ax.transAxes,
                    fontsize=11, verticalalignment="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 3: Executive Summary
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")

        has_previous_data = previous_data is not None and len(previous_data) > 0
        summary_lines = ["EXECUTIVE SUMMARY", ""]
        summary_lines.append(PASS_DEFINITION_TEXT)
        summary_lines.append("")

        if not has_previous_data:
            summary_lines.append("BPO Centers Performance Overview")
            summary_lines.append("")

        # Calculate improvements (only if we have previous data)
        improvements = {}
        if (
            has_previous_data
            and previous_metrics.get("avg_qa_score") is not None
            and bpo_metrics.get("avg_qa_score") is not None
            and previous_metrics["avg_qa_score"] != 0
        ):
            pct = (
                (bpo_metrics["avg_qa_score"] - previous_metrics["avg_qa_score"])
                / previous_metrics["avg_qa_score"]
            ) * 100
            improvements["QA Score"] = pct
            summary_lines.append(f"• QA Score: {pct:+.1f}% change")

        if (
            previous_metrics.get("pass_rate") is not None
            and bpo_metrics.get("pass_rate") is not None
            and previous_metrics["pass_rate"] != 0
        ):
            pct = (
                (bpo_metrics["pass_rate"] - previous_metrics["pass_rate"])
                / previous_metrics["pass_rate"]
            ) * 100
            improvements["Pass Rate"] = pct
            summary_lines.append(f"• Pass Rate: {pct:+.1f}% change")

        if (
            previous_metrics.get("avg_aht") is not None
            and bpo_metrics.get("avg_aht") is not None
            and previous_metrics["avg_aht"] != 0
        ):
            pct = (
                (bpo_metrics["avg_aht"] - previous_metrics["avg_aht"])
                / previous_metrics["avg_aht"]
            ) * 100
            improvements["AHT"] = pct
            summary_lines.append(f"• Average Handle Time: {pct:+.1f}% change")

        # Consistency: show absolute std change (e.g. 29.0 → 28.9)
        if (
            previous_metrics.get("score_std") is not None
            and bpo_metrics.get("score_std") is not None
        ):
            prev_std = previous_metrics["score_std"]
            bpo_std = bpo_metrics["score_std"]
            delta_std = bpo_std - prev_std
            if abs(delta_std) < CONSISTENCY_NO_CHANGE_THRESHOLD:
                summary_lines.append(
                    "• Consistency: No material change in score consistency"
                )
            else:
                summary_lines.append(
                    f"• Consistency: {prev_std:.1f} → {bpo_std:.1f} (std dev)"
                )

        # Add quality tier improvements
        if (
            previous_metrics.get("excellent_pct") is not None
            and bpo_metrics.get("excellent_pct") is not None
            and previous_metrics["excellent_pct"] != 0
        ):
            excellent_improvement = (
                (bpo_metrics["excellent_pct"] - previous_metrics["excellent_pct"])
                / previous_metrics["excellent_pct"]
            ) * 100
            if excellent_improvement > 0:
                improvements["Excellent Scores (90+)"] = excellent_improvement
                summary_lines.append(
                    f"• Excellent Scores (90+): {excellent_improvement:+.1f}% change"
                )

        summary_lines.extend(["", "KEY METRICS"])
        if has_previous_data:
            summary_lines.append("Previous Contact Center:")
            summary_lines.append(
                f"  • Total Calls: {previous_metrics.get('total_calls', 'N/A')}"
            )
            summary_lines.append(
                f"  • Avg QA Score: {previous_metrics.get('avg_qa_score', 'N/A'):.1f}"
                if previous_metrics.get("avg_qa_score") is not None
                else "  • Avg QA Score: N/A"
            )
            summary_lines.append(
                f"  • Median Score: {previous_metrics.get('score_median', 'N/A'):.1f}"
                if previous_metrics.get("score_median") is not None
                else "  • Median Score: N/A"
            )
            summary_lines.append(
                f"  • Std Dev: {previous_metrics.get('score_std', 'N/A'):.1f}"
                if previous_metrics.get("score_std") is not None
                else "  • Std Dev: N/A"
            )
            summary_lines.append(
                f"  • Pass Rate: {previous_metrics.get('pass_rate', 'N/A'):.1f}%"
                if previous_metrics.get("pass_rate") is not None
                else "  • Pass Rate: N/A"
            )
            summary_lines.append(
                f"  • Avg AHT: {previous_metrics.get('avg_aht', 'N/A'):.1f} min"
                if previous_metrics.get("avg_aht") is not None
                else "  • Avg AHT: N/A"
            )
            summary_lines.append(
                f"  • Excellent (90+): {previous_metrics.get('excellent_pct', 0):.1f}%"
                if previous_metrics.get("excellent_pct") is not None
                else "  • Excellent (90+): N/A"
            )
            summary_lines.append("")

        summary_lines.append("BPO Centers:")
        summary_lines.append(
            f"  • Total Calls: {bpo_metrics.get('total_calls', 'N/A')}"
        )
        summary_lines.append(
            f"  • Avg QA Score: {bpo_metrics.get('avg_qa_score', 'N/A'):.1f}"
            if bpo_metrics.get("avg_qa_score") is not None
            else "  • Avg QA Score: N/A"
        )
        summary_lines.append(
            f"  • Median Score: {bpo_metrics.get('score_median', 'N/A'):.1f}"
            if bpo_metrics.get("score_median") is not None
            else "  • Median Score: N/A"
        )
        summary_lines.append(
            f"  • Std Dev: {bpo_metrics.get('score_std', 'N/A'):.1f}"
            if bpo_metrics.get("score_std") is not None
            else "  • Std Dev: N/A"
        )
        summary_lines.append(
            f"  • Pass Rate: {bpo_metrics.get('pass_rate', 'N/A'):.1f}%"
            if bpo_metrics.get("pass_rate") is not None
            else "  • Pass Rate: N/A"
        )
        summary_lines.append(
            f"  • Avg AHT: {bpo_metrics.get('avg_aht', 'N/A'):.1f} min"
            if bpo_metrics.get("avg_aht") is not None
            else "  • Avg AHT: N/A"
        )
        summary_lines.append(
            f"  • Excellent (90+): {bpo_metrics.get('excellent_pct', 0):.1f}%"
            if bpo_metrics.get("excellent_pct") is not None
            else "  • Excellent (90+): N/A"
        )

        summary_text = "\n".join(summary_lines)
        ax.text(
            0.1,
            0.95,
            summary_text,
            ha="left",
            va="top",
            fontsize=11,
            family="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3+: Metric Comparison Charts (only if we have previous data)
        has_previous_data = previous_data is not None and len(previous_data) > 0
        if (
            has_previous_data
            and previous_metrics.get("avg_qa_score") is not None
            and bpo_metrics.get("avg_qa_score") is not None
        ):
            fig = create_comparison_chart(
                "Average QA Score",
                previous_metrics["avg_qa_score"],
                bpo_metrics["avg_qa_score"],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if (
            previous_metrics.get("pass_rate") is not None
            and bpo_metrics.get("pass_rate") is not None
        ):
            fig = create_comparison_chart(
                "Pass Rate (%)", previous_metrics["pass_rate"], bpo_metrics["pass_rate"]
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if (
            previous_metrics.get("avg_aht") is not None
            and bpo_metrics.get("avg_aht") is not None
        ):
            fig = create_comparison_chart(
                "Average Handle Time (minutes)",
                previous_metrics["avg_aht"],
                bpo_metrics["avg_aht"],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if (
            previous_metrics.get("total_calls") is not None
            and bpo_metrics.get("total_calls") is not None
        ):
            fig = create_comparison_chart(
                "Total Call Volume",
                previous_metrics["total_calls"],
                bpo_metrics["total_calls"],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if (
            previous_metrics.get("calls_per_day") is not None
            and bpo_metrics.get("calls_per_day") is not None
        ):
            fig = create_comparison_chart(
                "Calls Per Day",
                previous_metrics["calls_per_day"],
                bpo_metrics["calls_per_day"],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Improvement Summary Chart (only if we have previous data)
        if improvements and has_previous_data:
            fig = create_improvement_summary_chart(improvements)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Distribution Comparison (only if we have previous data)
        if has_previous_data and previous_data is not None and bpo_data is not None:
            if "QA Score" in previous_data.columns and "QA Score" in bpo_data.columns:
                fig = create_distribution_comparison(
                    previous_data["QA Score"], bpo_data["QA Score"]
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # Quality Tier Comparison (only if we have previous data)
        if (
            has_previous_data
            and previous_metrics.get("excellent_pct") is not None
            and bpo_metrics.get("excellent_pct") is not None
        ):
            fig = create_quality_tier_chart(previous_metrics, bpo_metrics)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Trend Chart
        if bpo_data is not None:
            fig = create_trend_chart(bpo_data)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Rubric Improvements (only if we have previous data)
        if has_previous_data and previous_data is not None and bpo_data is not None:
            rubric_improvements = calculate_rubric_improvements(previous_data, bpo_data)
            if rubric_improvements.get("top_improvements"):
                fig = create_rubric_improvement_chart(
                    rubric_improvements["top_improvements"]
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # Call Reasons Comparison (only if we have previous data)
        if (
            has_previous_data
            and previous_metrics.get("reason_distribution") is not None
            and bpo_metrics.get("reason_distribution") is not None
        ):
            fig = create_reason_comparison_chart(
                previous_metrics["reason_distribution"],
                bpo_metrics["reason_distribution"],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Call Outcomes Comparison (only if we have previous data)
        if (
            has_previous_data
            and previous_metrics.get("outcome_distribution") is not None
            and bpo_metrics.get("outcome_distribution") is not None
        ):
            fig = create_outcome_comparison_chart(
                previous_metrics["outcome_distribution"],
                bpo_metrics["outcome_distribution"],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Products Discussed Comparison (only if we have previous data)
        if (
            has_previous_data
            and previous_metrics.get("product_distribution") is not None
            and bpo_metrics.get("product_distribution") is not None
        ):
            fig = create_product_comparison_chart(
                previous_metrics["product_distribution"],
                bpo_metrics["product_distribution"],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # NEW IMPRESSIVE CHARTS FOR BPO REVIEW

        # KPI Dashboard (show BPO metrics, with comparison if available)
        print("  📊 Generating KPI Dashboard...")
        has_previous_data = previous_data is not None and len(previous_data) > 0
        if has_previous_data:
            fig = create_kpi_dashboard(previous_metrics, bpo_metrics)
        else:
            # Create BPO-only KPI dashboard
            fig = create_bpo_only_kpi_dashboard(bpo_metrics)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Agent Performance Heatmap
        if bpo_data is not None and len(bpo_data) > 0:
            print("  📊 Generating Agent Performance Heatmap...")
            fig = create_agent_performance_heatmap(
                bpo_data,
                total_calls_bpo=total_calls_bpo,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Agent Leaderboard
        if bpo_data is not None and len(bpo_data) > 0:
            print("  📊 Generating Agent Leaderboard...")
            fig = create_agent_leaderboard(
                bpo_data,
                total_calls=total_calls_bpo if total_calls_bpo is not None else bpo_metrics.get("total_calls"),
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Monthly Trend Analysis
        if bpo_data is not None and len(bpo_data) > 0:
            print("  📊 Generating Monthly Trend Analysis...")
            fig = create_monthly_trend_analysis(bpo_data)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Top Failure Reasons with Impact
        if bpo_data is not None and len(bpo_data) > 0:
            print("  📊 Generating Top Failure Reasons Analysis...")
            fig = create_top_failure_reasons_chart(bpo_data)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Top Call Reasons
        if bpo_data is not None and len(bpo_data) > 0:
            print("  📊 Generating Top Call Reasons Chart...")
            fig = create_top_call_reasons_chart(bpo_data)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Top Call Outcomes
        if bpo_data is not None and len(bpo_data) > 0:
            print("  📊 Generating Top Call Outcomes Chart...")
            fig = create_top_call_outcomes_chart(bpo_data)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Top Products Discussed
        if bpo_data is not None and len(bpo_data) > 0:
            print("  📊 Generating Top Products Chart...")
            fig = create_top_products_chart(bpo_data)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"✅ PDF report generated successfully: {output_path}")


def main():
    """Main function to run the comparison report generator."""
    parser = argparse.ArgumentParser(
        description="Generate comparison report between previous contact center and BPO Centers"
    )
    parser.add_argument(
        "--previous-csv",
        nargs="+",
        type=str,
        help="Path(s) to previous contact center CSV file(s)",
    )
    parser.add_argument(
        "--previous-csv-dir",
        type=str,
        help="Directory containing previous contact center CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.pdf",
        help="Output PDF file path (default: comparison_report.pdf)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(CACHE_FILE),
        help=f"Path to BPO Centers cache file (default: {CACHE_FILE})",
    )
    parser.add_argument(
        "--bpo-start-date",
        type=str,
        default="2025-08-01",
        help="BPO Centers start date filter in YYYY-MM-DD format (default: 2025-08-01)",
    )
    parser.add_argument(
        "--download-cache-only",
        action="store_true",
        help="Download BPO call data from S3 month caches to the cache file and exit (no PDF). Uses .streamlit/secrets.toml or env BPO_S3_BUCKET.",
    )

    args = parser.parse_args()

    # Download-only mode: fetch from S3 and write to cache file
    if args.download_cache_only:
        cache_path = Path(args.cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        config = _get_s3_config()
        if not config:
            print("❌ S3 not configured. Set .streamlit/secrets.toml [s3] or env BPO_S3_BUCKET and AWS credentials.")
            sys.exit(1)
        print("📂 Downloading BPO month caches from S3...")
        call_data = _load_bpo_call_data_from_s3()
        if not call_data:
            print("❌ No call data returned from S3.")
            sys.exit(1)
        out = {"call_data": call_data, "errors": [], "timestamp": datetime.now().isoformat(), "count": len(call_data)}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out, f, default=str, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(call_data)} calls to {cache_path}")
        sys.exit(0)

    # Parse BPO start date
    try:
        bpo_start_date = datetime.strptime(args.bpo_start_date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid BPO start date format: {args.bpo_start_date}")
        print("   Use YYYY-MM-DD format (e.g., 2025-08-01)")
        sys.exit(1)

    # Determine CSV file paths (optional - can generate BPO-only report)
    csv_paths = []
    previous_data = None
    previous_metrics = {}

    if args.previous_csv:
        csv_paths = [Path(p) for p in args.previous_csv]
    elif args.previous_csv_dir:
        csv_dir = Path(args.previous_csv_dir)
        if not csv_dir.exists():
            print(f"⚠️  Warning: CSV directory not found: {csv_dir}")
            print("   Generating BPO-only report (no comparison data)")
        else:
            csv_paths = list(csv_dir.glob("*.csv"))
            if not csv_paths:
                print(f"⚠️  Warning: No CSV files found in directory: {csv_dir}")
                print("   Generating BPO-only report (no comparison data)")
    else:
        print("ℹ️  No previous CSV files provided - generating BPO-only report")
        print("   (Use --previous-csv or --previous-csv-dir for comparison)")

    print("=" * 80)
    print("Contact Center Comparison Report Generator")
    print("=" * 80)
    print()

    try:
        # Load data
        print("📊 Loading data...")
        if csv_paths:
            previous_data = load_previous_center_data(csv_paths)
        else:
            previous_data = None
            print("   (Skipping previous center data - BPO-only report)")

        bpo_data = load_bpo_centers_data(Path(args.cache), bpo_start_date)

        if bpo_data is None or len(bpo_data) == 0:
            raise ValueError("No BPO Centers data loaded. Cannot generate report.")

        # Apply global filters to get canonical dataframes (shared across all report sections)
        df_prev = (
            apply_global_filters(previous_data, "previous")
            if previous_data is not None and len(previous_data) > 0
            else None
        )
        df_bpo = apply_global_filters(bpo_data, "bpo")
        total_calls_bpo = len(df_bpo)

        if len(df_bpo) == 0:
            raise ValueError(
                "No BPO data remaining after global filters (call_id, QA Score, Call Date, dedupe)."
            )

        # Compute cohort config from canonical dataframes
        previous_config = None
        if df_prev is not None and len(df_prev) > 0:
            previous_config = compute_cohort_config(
                df_prev,
                start_date=None,
                end_date=None,
                cohort_name="Previous Contact Center",
            )

        bpo_end = None
        if len(df_bpo) > 0 and "Call Date" in df_bpo.columns:
            bpo_end = pd.Timestamp(df_bpo["Call Date"].max())
        bpo_config = compute_cohort_config(
            df_bpo,
            start_date=bpo_start_date,
            end_date=bpo_end,
            cohort_name="BPO Centers",
        )

        print("\n📈 Calculating metrics...")
        previous_metrics = (
            calculate_all_metrics(df_prev, cohort_config=previous_config)
            if df_prev is not None
            else {}
        )
        bpo_metrics = calculate_all_metrics(df_bpo, cohort_config=bpo_config)

        # Master validation block (stops if any check fails)
        cohort_cfg_dict = {}
        if previous_config:
            cohort_cfg_dict["Previous Contact Center"] = previous_config
        cohort_cfg_dict["BPO Centers"] = bpo_config
        validate_report(
            df_prev,
            df_bpo,
            cohort_cfg_dict,
            previous_metrics,
            bpo_metrics,
        )

        # Agent reconciliation stats (for runtime log and validation)
        agent_stats = _compute_agent_reconciliation_stats(df_bpo, total_calls_bpo)

        # Runtime output: COHORT SUMMARY + AGENT RECONCILIATION
        print("\nCOHORT SUMMARY")
        print("--------------")
        if previous_config:
            print("Previous:")
            print(f"  Start: {previous_config.get('start_date')}")
            print(f"  End: {previous_config.get('end_date')}")
            print(f"  Days: {previous_config.get('days_in_period')}")
            print(f"  Total Calls: {previous_config.get('total_calls')}")
            pcpd = previous_config.get("calls_per_day")
            print(f"  Calls/Day: {pcpd:.2f}" if pcpd is not None else "  Calls/Day: N/A")
            print()
        print("BPO:")
        print(f"  BPO TOTAL CALLS: {total_calls_bpo}")
        print(f"  Start: {bpo_config.get('start_date')}")
        print(f"  End: {bpo_config.get('end_date')}")
        print(f"  Days: {bpo_config.get('days_in_period')}")
        cpd = bpo_config.get("calls_per_day")
        print(f"  Calls/Day: {cpd:.2f}" if cpd is not None else "  Calls/Day: N/A")
        print()
        print("AGENT RECONCILIATION")
        print("--------------------")
        print(f"  AGENT SUM (ALL): {agent_stats['agent_sum_all']}")
        if agent_stats.get("top_n_sum") is not None:
            print(f"  TOP N SUM: {agent_stats['top_n_sum']} (top {AGENT_LEADERBOARD_TOP_N})")
            print(f"  OTHER/UNKNOWN: {agent_stats['other_unknown']}")
        print()
        print(f"PASS THRESHOLD: {PASS_THRESHOLD}")
        print("VALIDATION: PASSED")
        print()

        # Generate report (use canonical df_bpo)
        output_path = Path(args.output)
        generate_pdf_report(
            previous_metrics,
            bpo_metrics,
            output_path,
            bpo_start_date,
            df_prev,
            df_bpo,
            cohort_config=cohort_cfg_dict,
            total_calls_bpo=total_calls_bpo,
        )

        print("\n" + "=" * 80)
        print("✅ Report generation complete!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
