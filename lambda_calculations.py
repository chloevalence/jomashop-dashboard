"""
Calculation functions for QA Dashboard metrics.
Lambda-ready module containing all calculation logic without UI dependencies.

This module provides all calculation functions extracted from the Streamlit dashboard
for use in AWS Lambda functions. All functions accept pandas DataFrames and return
JSON-serializable dictionaries.
"""

import pandas as pd
import numpy as np
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union


# ============================================================================
# Helper Functions
# ============================================================================

# Known agent mappings (hardcoded for Lambda - no file I/O)
KNOWN_AGENT_MAPPINGS = {
    # Agent 1: Jesus
    "unknown": "Agent 1",
    "bp016803073": "Agent 1",
    "bp016803074": "Agent 1",
    "bp agent 016803073": "Agent 1",  # Space version
    "bp agent 016803074": "Agent 1",  # Space version
    # Agent 2: Gerardo
    "bpagent024577540": "Agent 2",
    "bp agent 024577540": "Agent 2",  # Space version
    # Agent 3: Edgar
    "bpagent030844482": "Agent 3",
    "bp agent 030844482": "Agent 3",  # Space version
    # Agent 4: Osiris
    "bpagent047779576": "Agent 4",
    "bp agent 047779576": "Agent 4",  # Space version
    # Agent 6: Daniela
    "bpagent065185612": "Agent 6",
    "bp agent 065185612": "Agent 6",  # Space version
    # Agent 7: Yasmin
    "bpagent072229421": "Agent 7",
    "bp agent 072229421": "Agent 7",  # Space version
    # Agent 8: Moises
    "bpagent089724913": "Agent 8",
    "bp agent 089724913": "Agent 8",  # Space version
    # Agent 9: Marcos
    "bpagent093540654": "Agent 9",
    "bp agent 093540654": "Agent 9",  # Space version
    # Agent 10: Angel
    "bpagent102256681": "Agent 10",
    "bp agent 102256681": "Agent 10",  # Space version
    # Agent 5: (Left, but calls need to be accessible to admins)
    "bpagent051705087": "Agent 5",
    "bp agent 051705087": "Agent 5",  # Space version
    # Agent 11: (No agent account, but viewable to admins)
    "bpagent113827380": "Agent 11",
    "bp agent 113827380": "Agent 11",  # Space version
}


def normalize_agent_id(agent_str: Union[str, Any]) -> Union[str, Any]:
    """
    Normalize agent ID by extracting first two digits after 'bpagent'.

    Lambda-compatible version without file I/O dependencies.

    Format: 'bpagent###########' → extract first two digits → 'Agent ##'
    Exceptions: Special cases for Jesus (unknown, bp016803073, bp016803074, bp01*)

    Args:
        agent_str: Agent ID string (may contain extra characters)

    Returns:
        Normalized agent ID in 'Agent ##' format, or original string if no match
    """
    if pd.isna(agent_str) or not agent_str:
        return agent_str

    agent_str_lower = str(agent_str).lower().strip()

    # Check if already in "Agent X" or "BPO Agent X" format - return as-is
    if agent_str_lower.startswith("agent ") or agent_str_lower.startswith("bpo agent "):
        # Extract number and return in consistent format
        agent_str_clean = (
            agent_str_lower.replace("bpo agent ", "").replace("agent ", "").strip()
        )
        try:
            agent_num = int(agent_str_clean)
            return f"Agent {agent_num}"
        except ValueError:
            pass

    # Special cases for Jesus (Agent 1)
    agent_id_normalized = agent_str_lower.replace(" ", "").replace("_", "")
    if agent_id_normalized == "unknown" or agent_str_lower == "unknown":
        return "Agent 1"
    if agent_id_normalized in ["bp016803073", "bp016803074"]:
        return "Agent 1"
    if agent_id_normalized.startswith("bp01"):
        return "Agent 1"

    # Check known mappings first (these take precedence)
    for known_id, known_name in KNOWN_AGENT_MAPPINGS.items():
        known_id_normalized = known_id.replace(" ", "").replace("_", "")
        if (
            agent_id_normalized == known_id_normalized
            or agent_str_lower == known_id.lower()
        ):
            return known_name

    # Extract first two digits after "bpagent"
    # Pattern: bpagent########### → extract first two digits (##)
    match = re.search(r"bpagent(\d{2})", agent_id_normalized)
    if match:
        agent_num = int(match.group(1))
        return f"Agent {agent_num}"

    # For new agent IDs, assign deterministically based on hash
    # This ensures the same agent ID always gets the same number
    hash_value = int(hashlib.md5(agent_id_normalized.encode()).hexdigest(), 16)
    # Use modulo to get a number between 1 and 99
    # Note: Agent 5 and Agent 11 are reserved for known mappings, skip them in hash assignment
    agent_number = (hash_value % 99) + 1
    # Skip 5 and 11 in hash assignment (they're reserved for known mappings)
    if agent_number == 5:
        agent_number = 12  # Skip 5, use 12 instead
    if agent_number == 11:
        agent_number = 12  # Skip 11, use 12 instead

    return f"Agent {agent_number}"


def normalize_category(value: Union[str, Any]) -> Union[str, Any]:
    """
    Normalize call reason/outcome categories to merge duplicates and handle case sensitivity.

    Rules:
    1. Case-insensitive: "order status inquiry" and "Order status inquiry" → "Order status inquiry"
    2. Merge shipping-related: "customer informed about shipping delay", "shipping status",
       "shipping timeline" → "shipping status"
    3. Rename: "refund and return process initiated" → "return process initiated"

    Args:
        value: Category string to normalize

    Returns:
        Normalized category string
    """
    if pd.isna(value) or not value or not str(value).strip():
        return value

    value_str = str(value).strip()

    # Normalize to lowercase for comparison
    value_lower = value_str.lower()

    # Merge shipping-related categories
    shipping_variants = [
        "customer informed about shipping delay",
        "shipping status",
        "shipping timeline",
    ]
    if value_lower in shipping_variants:
        return "shipping status"

    # Rename "refund and return process initiated" to "return process initiated"
    if value_lower == "refund and return process initiated":
        return "return process initiated"

    # Return the value as-is (case normalization handled separately if needed)
    return value_str


def extract_products_from_text(text: Union[str, Any]) -> List[str]:
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

    # Product categories
    product_categories = [
        "watch",
        "watches",
        "timepiece",
        "timepieces",
        "wristwatch",
        "wristwatches",
        "bracelet",
        "bracelets",
        "necklace",
        "necklaces",
        "ring",
        "rings",
        "earrings",
        "pendant",
        "pendants",
        "jewelry",
        "jewellery",
        "accessories",
    ]

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
            category_normalized = (
                category.title() if category != "jewellery" else "Jewelry"
            )
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


def normalize_categories_in_dataframe(
    df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    Normalize categories in a DataFrame column, handling case-insensitive duplicates.

    This function:
    1. Applies category-specific normalization rules
    2. Merges case-insensitive duplicates (keeps the most common capitalization)

    Args:
        df: DataFrame to normalize
        column_name: Name of the column to normalize

    Returns:
        DataFrame with normalized column
    """
    if column_name not in df.columns:
        return df

    df = df.copy()

    # First, apply category-specific normalization rules
    df[column_name] = df[column_name].apply(normalize_category)

    # Then, handle case-insensitive duplicates
    # Group by lowercase version and use the most common capitalization
    if df[column_name].notna().any():
        # Create a mapping: lowercase -> most common capitalization
        category_counts = df[column_name].value_counts()
        lowercase_to_canonical = {}

        for category in category_counts.index:
            if pd.notna(category):
                cat_lower = str(category).lower()
                # If we haven't seen this lowercase version, or this capitalization is more common
                if cat_lower not in lowercase_to_canonical:
                    lowercase_to_canonical[cat_lower] = category
                else:
                    # Use the one with higher count
                    current_count = category_counts.get(category, 0)
                    existing_cat = lowercase_to_canonical[cat_lower]
                    existing_count = category_counts.get(existing_cat, 0)
                    if current_count > existing_count:
                        lowercase_to_canonical[cat_lower] = category

        # Apply the mapping
        def apply_case_normalization(val):
            if pd.isna(val):
                return val
            val_str = str(val)
            val_lower = val_str.lower()
            if val_lower in lowercase_to_canonical:
                return lowercase_to_canonical[val_lower]
            return val

        df[column_name] = df[column_name].apply(apply_case_normalization)

    return df


# ============================================================================
# Core Calculation Functions
# ============================================================================


def calculate_qa_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate QA score metrics.

    Args:
        df: DataFrame with QA data

    Returns:
        Dictionary containing avg_qa_score and qa_score_count
    """
    metrics = {}

    if "QA Score" in df.columns:
        valid_scores = df["QA Score"].dropna()
        if len(valid_scores) > 0:
            metrics["avg_qa_score"] = float(valid_scores.mean())
            metrics["qa_score_count"] = int(len(valid_scores))
        else:
            metrics["avg_qa_score"] = None
            metrics["qa_score_count"] = 0
    else:
        metrics["avg_qa_score"] = None
        metrics["qa_score_count"] = 0

    return metrics


def calculate_pass_rate(df: pd.DataFrame) -> Optional[float]:
    """
    Calculate pass rate from dataframe.

    Args:
        df: DataFrame with rubric pass/fail counts

    Returns:
        Pass rate as percentage (0-100), or None if cannot calculate
    """
    if "Rubric Pass Count" in df.columns and "Rubric Fail Count" in df.columns:
        total_pass = df["Rubric Pass Count"].sum()
        total_fail = df["Rubric Fail Count"].sum()
        if (total_pass + total_fail) > 0:
            return float((total_pass / (total_pass + total_fail)) * 100)
    return None


def calculate_pass_rate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate pass rate metrics.

    Args:
        df: DataFrame with QA data

    Returns:
        Dictionary containing pass_rate, pass_count, and fail_count
    """
    metrics = {}

    # Method 1: From Rubric Pass/Fail Count
    if "Rubric Pass Count" in df.columns and "Rubric Fail Count" in df.columns:
        total_pass = df["Rubric Pass Count"].sum()
        total_fail = df["Rubric Fail Count"].sum()
        if (total_pass + total_fail) > 0:
            metrics["pass_rate"] = float((total_pass / (total_pass + total_fail)) * 100)
            metrics["pass_count"] = int(total_pass)
            metrics["fail_count"] = int(total_fail)
        else:
            metrics["pass_rate"] = None
            metrics["pass_count"] = 0
            metrics["fail_count"] = 0
    # Method 2: From Label column (Positive = Pass)
    elif "Label" in df.columns:
        valid_labels = df["Label"].dropna()
        if len(valid_labels) > 0:
            # Exclude Invalid labels
            valid_labels = valid_labels[valid_labels.str.lower() != "invalid"]
            if len(valid_labels) > 0:
                pass_count = int((valid_labels.str.lower() == "positive").sum())
                fail_count = int((valid_labels.str.lower() == "negative").sum())
                total = pass_count + fail_count
                if total > 0:
                    metrics["pass_rate"] = float((pass_count / total) * 100)
                    metrics["pass_count"] = pass_count
                    metrics["fail_count"] = fail_count
                else:
                    metrics["pass_rate"] = None
                    metrics["pass_count"] = 0
                    metrics["fail_count"] = 0
            else:
                metrics["pass_rate"] = None
                metrics["pass_count"] = 0
                metrics["fail_count"] = 0
        else:
            metrics["pass_rate"] = None
            metrics["pass_count"] = 0
            metrics["fail_count"] = 0
    else:
        metrics["pass_rate"] = None
        metrics["pass_count"] = 0
        metrics["fail_count"] = 0

    return metrics


def calculate_aht_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate Average Handle Time metrics.

    Args:
        df: DataFrame with AHT data

    Returns:
        Dictionary containing avg_aht and aht_count
    """
    metrics = {}

    if "AHT" in df.columns:
        valid_aht = df["AHT"].dropna()
        if len(valid_aht) > 0:
            metrics["avg_aht"] = float(valid_aht.mean())
            metrics["aht_count"] = int(len(valid_aht))
        else:
            metrics["avg_aht"] = None
            metrics["aht_count"] = 0
    else:
        metrics["avg_aht"] = None
        metrics["aht_count"] = 0

    return metrics


def calculate_volume_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate call volume metrics.

    Args:
        df: DataFrame with call data

    Returns:
        Dictionary containing total_calls and calls_per_day
    """
    metrics = {}

    # Total calls (exclude Invalid labels if present)
    if "Label" in df.columns:
        valid_calls = (
            df[df["Label"].str.lower() != "invalid"]
            if df["Label"].dtype == "object"
            else df
        )
        metrics["total_calls"] = int(len(valid_calls))
    else:
        valid_calls = df
        metrics["total_calls"] = int(len(df))

    # Calls per day - use valid_calls for date range calculation
    if "Call Date" in valid_calls.columns and metrics["total_calls"] > 0:
        # Filter out NaT values before calculating date range
        valid_dates = valid_calls["Call Date"].dropna()
        if len(valid_dates) > 0:
            date_range = (valid_dates.max() - valid_dates.min()).days + 1
            if date_range > 0:
                metrics["calls_per_day"] = float(metrics["total_calls"] / date_range)
            else:
                metrics["calls_per_day"] = float(metrics["total_calls"])
        else:
            # All dates are NaT - cannot calculate calls per day
            metrics["calls_per_day"] = None
    else:
        metrics["calls_per_day"] = None

    return metrics


def calculate_rubric_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate rubric failure metrics.

    Args:
        df: DataFrame with rubric data

    Returns:
        Dictionary containing top_failure_codes and total_rubric_failures
    """
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
        # Convert to list of lists for JSON serialization
        metrics["top_failure_codes"] = [
            [code, count] for code, count in sorted_failures[:10]
        ]  # Top 10
        metrics["total_rubric_failures"] = int(sum(failure_counts.values()))
    else:
        # Use Rubric Fail Count if available
        if "Rubric Fail Count" in df.columns:
            metrics["total_rubric_failures"] = int(df["Rubric Fail Count"].sum())
        else:
            metrics["total_rubric_failures"] = 0
        metrics["top_failure_codes"] = []

    return metrics


def calculate_agent_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate agent performance distribution metrics.

    Args:
        df: DataFrame with agent and QA Score data

    Returns:
        Dictionary containing agent_performance (list of dicts) and num_agents
    """
    metrics = {}

    if "Agent" in df.columns and "QA Score" in df.columns:
        agent_perf = (
            df.groupby("Agent")["QA Score"].agg(["mean", "count"]).reset_index()
        )
        agent_perf.columns = ["Agent", "Avg Score", "Call Count"]
        agent_perf = agent_perf.sort_values("Avg Score", ascending=False)

        # Convert DataFrame to list of dicts for JSON serialization
        metrics["agent_performance"] = agent_perf.to_dict("records")
        # Ensure numeric values are native Python types
        for record in metrics["agent_performance"]:
            record["Avg Score"] = float(record["Avg Score"])
            record["Call Count"] = int(record["Call Count"])

        metrics["num_agents"] = int(len(agent_perf))
    else:
        metrics["agent_performance"] = []
        metrics["num_agents"] = 0

    return metrics


def calculate_consistency_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate consistency metrics (standard deviation, coefficient of variation).

    Args:
        df: DataFrame with QA Score data

    Returns:
        Dictionary containing score statistics
    """
    metrics = {}

    if "QA Score" in df.columns:
        valid_scores = df["QA Score"].dropna()
        if len(valid_scores) > 0:
            metrics["score_std"] = float(valid_scores.std())
            metrics["score_mean"] = float(valid_scores.mean())
            metrics["score_median"] = float(valid_scores.median())
            metrics["score_min"] = float(valid_scores.min())
            metrics["score_max"] = float(valid_scores.max())
            metrics["score_range"] = float(valid_scores.max() - valid_scores.min())
            # Coefficient of variation (normalized consistency)
            if metrics["score_mean"] > 0:
                metrics["coefficient_of_variation"] = float(
                    (metrics["score_std"] / metrics["score_mean"]) * 100
                )
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


def calculate_quality_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate quality tier distribution.

    Args:
        df: DataFrame with QA Score data

    Returns:
        Dictionary containing quality tier counts and percentages
    """
    metrics = {}

    if "QA Score" in df.columns:
        valid_scores = df["QA Score"].dropna()
        if len(valid_scores) > 0:
            total = len(valid_scores)
            excellent = int(((valid_scores >= 90) & (valid_scores <= 100)).sum())
            good = int(((valid_scores >= 75) & (valid_scores < 90)).sum())
            fair = int(((valid_scores >= 60) & (valid_scores < 75)).sum())
            poor = int((valid_scores < 60).sum())

            metrics["excellent_count"] = excellent
            metrics["good_count"] = good
            metrics["fair_count"] = fair
            metrics["poor_count"] = poor
            metrics["excellent_pct"] = float((excellent / total) * 100)
            metrics["good_pct"] = float((good / total) * 100)
            metrics["fair_pct"] = float((fair / total) * 100)
            metrics["poor_pct"] = float((poor / total) * 100)
        else:
            metrics["excellent_count"] = 0
            metrics["good_count"] = 0
            metrics["fair_count"] = 0
            metrics["poor_count"] = 0
            metrics["excellent_pct"] = 0.0
            metrics["good_pct"] = 0.0
            metrics["fair_pct"] = 0.0
            metrics["poor_pct"] = 0.0
    else:
        metrics["excellent_count"] = 0
        metrics["good_count"] = 0
        metrics["fair_count"] = 0
        metrics["poor_count"] = 0
        metrics["excellent_pct"] = 0.0
        metrics["good_pct"] = 0.0
        metrics["fair_pct"] = 0.0
        metrics["poor_pct"] = 0.0

    return metrics


def calculate_trend_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate trend metrics (weekly/monthly aggregations).

    Args:
        df: DataFrame with Call Date and QA Score data

    Returns:
        Dictionary containing weekly and monthly trend data
    """
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
                    metrics["weekly_improvement_rate"] = float(slope)
                    # Convert PeriodIndex to string for JSON serialization
                    metrics["weekly_trend_data"] = {
                        str(k): float(v) for k, v in weekly_avg.to_dict().items()
                    }
                else:
                    metrics["weekly_improvement_rate"] = None
                    metrics["weekly_trend_data"] = {}
            else:
                metrics["weekly_improvement_rate"] = None
                metrics["weekly_trend_data"] = {}

            # Monthly trends
            monthly_avg = df_copy.groupby("Month")["QA Score"].mean()
            if len(monthly_avg) > 0:
                # Convert PeriodIndex to string for JSON serialization
                metrics["monthly_trend_data"] = {
                    str(k): float(v) for k, v in monthly_avg.to_dict().items()
                }
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
) -> Dict[str, Any]:
    """
    Calculate rubric code improvements by comparing failure rates.

    Args:
        previous_df: DataFrame with previous period data
        bpo_df: DataFrame with BPO period data

    Returns:
        Dictionary containing top_improvements and total_improvements
    """
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
                    failure_rate = float((failures / total) * 100)
                    failure_rates[col] = {
                        "failures": int(failures),
                        "total": int(total),
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
                        "improvement_pct": float(
                            (improvement / prev_rate * 100) if prev_rate > 0 else 0
                        ),
                    }
                )

    # Sort by improvement amount
    improvements.sort(key=lambda x: x["improvement"], reverse=True)
    metrics["top_improvements"] = improvements[:10]  # Top 10
    metrics["total_improvements"] = int(len(improvements))

    return metrics


def calculate_reason_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate call reason distribution metrics.

    Args:
        df: DataFrame with Reason column

    Returns:
        Dictionary containing reason distribution statistics
    """
    metrics = {}

    if "Reason" not in df.columns:
        return metrics

    reason_counts = df["Reason"].value_counts()
    total_calls = int(len(df[df["Reason"].notna()]))

    if total_calls == 0:
        return metrics

    # Convert counts to int for JSON serialization
    metrics["reason_distribution"] = {
        k: int(v) for k, v in reason_counts.to_dict().items()
    }
    metrics["top_reason"] = reason_counts.index[0] if len(reason_counts) > 0 else None
    metrics["top_reason_count"] = (
        int(reason_counts.iloc[0]) if len(reason_counts) > 0 else 0
    )
    metrics["top_reason_pct"] = (
        float((reason_counts.iloc[0] / total_calls * 100))
        if len(reason_counts) > 0
        else 0.0
    )
    metrics["unique_reasons"] = int(len(reason_counts))
    metrics["reason_calls_with_data"] = total_calls

    return metrics


def calculate_outcome_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate call outcome distribution metrics.

    Args:
        df: DataFrame with Outcome column

    Returns:
        Dictionary containing outcome distribution statistics
    """
    metrics = {}

    if "Outcome" not in df.columns:
        return metrics

    outcome_counts = df["Outcome"].value_counts()
    total_calls = int(len(df[df["Outcome"].notna()]))

    if total_calls == 0:
        return metrics

    # Convert counts to int for JSON serialization
    metrics["outcome_distribution"] = {
        k: int(v) for k, v in outcome_counts.to_dict().items()
    }
    metrics["top_outcome"] = (
        outcome_counts.index[0] if len(outcome_counts) > 0 else None
    )
    metrics["top_outcome_count"] = (
        int(outcome_counts.iloc[0]) if len(outcome_counts) > 0 else 0
    )
    metrics["top_outcome_pct"] = (
        float((outcome_counts.iloc[0] / total_calls * 100))
        if len(outcome_counts) > 0
        else 0.0
    )
    metrics["unique_outcomes"] = int(len(outcome_counts))
    metrics["outcome_calls_with_data"] = total_calls

    return metrics


def calculate_product_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate product distribution metrics from Summary, Reason, and Outcome fields.

    Args:
        df: DataFrame with Summary, Reason, and/or Outcome columns

    Returns:
        Dictionary containing product distribution statistics
    """
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

    # Convert counts to int for JSON serialization
    metrics["product_distribution"] = {
        k: int(v) for k, v in product_counts.to_dict().items()
    }
    metrics["top_product"] = (
        product_counts.index[0] if len(product_counts) > 0 else None
    )
    metrics["top_product_count"] = (
        int(product_counts.iloc[0]) if len(product_counts) > 0 else 0
    )
    metrics["top_product_pct"] = (
        float((product_counts.iloc[0] / total_mentions * 100))
        if len(product_counts) > 0
        else 0.0
    )
    metrics["unique_products"] = int(len(product_counts))
    metrics["total_product_mentions"] = int(total_mentions)
    # Count calls where ANY of the text fields have non-null data
    if text_fields:
        calls_with_any_text = df[df[text_fields].notna().any(axis=1)]
        metrics["calls_with_products"] = int(len(calls_with_any_text))
    else:
        metrics["calls_with_products"] = 0

    return metrics


def calculate_all_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all KPIs for a dataset.

    Args:
        df: DataFrame with QA call data

    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}

    metrics.update(calculate_qa_metrics(df))
    metrics.update(calculate_pass_rate_metrics(df))
    metrics.update(calculate_aht_metrics(df))
    metrics.update(calculate_volume_metrics(df))
    metrics.update(calculate_rubric_metrics(df))
    metrics.update(calculate_agent_metrics(df))
    metrics.update(calculate_consistency_metrics(df))
    metrics.update(calculate_quality_distribution(df))
    metrics.update(calculate_trend_metrics(df))
    metrics.update(calculate_reason_metrics(df))
    metrics.update(calculate_outcome_metrics(df))
    metrics.update(calculate_product_metrics(df))

    return metrics


# ============================================================================
# Advanced Analytics Functions
# ============================================================================


def calculate_historical_baselines(
    df: pd.DataFrame,
    current_start_date: Union[datetime, str],
    current_end_date: Union[datetime, str],
) -> Dict[str, Any]:
    """
    Calculate historical baselines for comparison.

    Args:
        df: DataFrame with call data
        current_start_date: Start date of current period (date or datetime)
        current_end_date: End date of current period (date or datetime)

    Returns:
        Dictionary with baseline metrics for last_30_days, last_90_days, and year_over_year
    """
    # Convert to pandas Timestamp for consistent comparison
    if not isinstance(current_start_date, pd.Timestamp):
        current_start_date = pd.Timestamp(current_start_date)
    if not isinstance(current_end_date, pd.Timestamp):
        current_end_date = pd.Timestamp(current_end_date)

    baselines = {}

    # Get the earliest and latest dates in the dataset
    if df.empty or "Call Date" not in df.columns:
        return baselines

    df_min_date = df["Call Date"].min()
    df_max_date = df["Call Date"].max()

    # Last 30 days baseline
    # First try: data before current period
    last_30_start = current_end_date - timedelta(days=30)
    last_30_data = df[
        (df["Call Date"] >= last_30_start) & (df["Call Date"] < current_start_date)
    ]

    # Fallback: if no data before current period, use earlier 30 days from within dataset
    if last_30_data.empty:
        # Use 30 days ending just before current period, or if that's not possible,
        # use the earliest 30 days in the dataset
        if current_start_date > df_min_date:
            # Try to get 30 days before current_start_date
            fallback_end = current_start_date - timedelta(days=1)
            fallback_start = fallback_end - timedelta(days=30)
            if fallback_start < df_min_date:
                fallback_start = df_min_date
            last_30_data = df[
                (df["Call Date"] >= fallback_start) & (df["Call Date"] <= fallback_end)
            ]
        else:
            # Current period starts at beginning, use earliest 30 days
            last_30_data = df[
                (df["Call Date"] >= df_min_date)
                & (
                    df["Call Date"]
                    <= min(df_min_date + timedelta(days=30), df_max_date)
                )
            ]

    if not last_30_data.empty:
        avg_score = None
        if "QA Score" in last_30_data.columns:
            avg_score = float(last_30_data["QA Score"].mean())

        baselines["last_30_days"] = {
            "avg_score": avg_score,
            "pass_rate": calculate_pass_rate(last_30_data),
            "total_calls": int(len(last_30_data)),
            "period": [
                last_30_data["Call Date"].min().isoformat()
                if pd.notna(last_30_data["Call Date"].min())
                else None,
                last_30_data["Call Date"].max().isoformat()
                if pd.notna(last_30_data["Call Date"].max())
                else None,
            ],
        }

    # Last 90 days baseline
    # First try: data before current period
    last_90_start = current_end_date - timedelta(days=90)
    last_90_data = df[
        (df["Call Date"] >= last_90_start) & (df["Call Date"] < current_start_date)
    ]

    # Fallback: if no data before current period, use earlier 90 days from within dataset
    if last_90_data.empty:
        if current_start_date > df_min_date:
            # Try to get 90 days before current_start_date
            fallback_end = current_start_date - timedelta(days=1)
            fallback_start = fallback_end - timedelta(days=90)
            if fallback_start < df_min_date:
                fallback_start = df_min_date
            last_90_data = df[
                (df["Call Date"] >= fallback_start) & (df["Call Date"] <= fallback_end)
            ]
        else:
            # Current period starts at beginning, use earliest 90 days
            last_90_data = df[
                (df["Call Date"] >= df_min_date)
                & (
                    df["Call Date"]
                    <= min(df_min_date + timedelta(days=90), df_max_date)
                )
            ]

    if not last_90_data.empty:
        avg_score = None
        if "QA Score" in last_90_data.columns:
            avg_score = float(last_90_data["QA Score"].mean())

        baselines["last_90_days"] = {
            "avg_score": avg_score,
            "pass_rate": calculate_pass_rate(last_90_data),
            "total_calls": int(len(last_90_data)),
            "period": [
                last_90_data["Call Date"].min().isoformat()
                if pd.notna(last_90_data["Call Date"].min())
                else None,
                last_90_data["Call Date"].max().isoformat()
                if pd.notna(last_90_data["Call Date"].max())
                else None,
            ],
        }

    # Year-over-year (if data available)
    if current_start_date.year > df["Call Date"].min().year:
        yoy_start = current_start_date - timedelta(days=365)
        yoy_end = current_end_date - timedelta(days=365)
        yoy_data = df[(df["Call Date"] >= yoy_start) & (df["Call Date"] <= yoy_end)]
        if not yoy_data.empty:
            avg_score = None
            if "QA Score" in yoy_data.columns:
                avg_score = float(yoy_data["QA Score"].mean())

            baselines["year_over_year"] = {
                "avg_score": avg_score,
                "pass_rate": calculate_pass_rate(yoy_data),
                "total_calls": int(len(yoy_data)),
                "period": [
                    yoy_start.isoformat(),
                    yoy_end.isoformat(),
                ],
            }

    return baselines


def calculate_percentile_rankings(
    df: pd.DataFrame, metric_col: str = "QA Score"
) -> List[Dict[str, Any]]:
    """
    Calculate percentile rankings for agents.

    Args:
        df: DataFrame with agent performance data
        metric_col: Column name for the metric to rank

    Returns:
        List of dictionaries with agent, metric value, and percentile ranking
    """
    if metric_col not in df.columns:
        return []

    agent_perf = df.groupby("Agent")[metric_col].mean().reset_index()
    agent_perf["percentile"] = agent_perf[metric_col].rank(pct=True) * 100
    agent_perf = agent_perf.sort_values("percentile", ascending=False)

    # Convert to list of dicts for JSON serialization
    result = []
    for _, row in agent_perf.iterrows():
        result.append(
            {
                "Agent": str(row["Agent"]),
                metric_col: float(row[metric_col]),
                "percentile": float(row["percentile"]),
            }
        )

    return result


def calculate_trend_slope(dates: pd.Series, scores: pd.Series) -> float:
    """
    Calculate linear trend slope.

    Args:
        dates: Series of dates
        scores: Series of scores

    Returns:
        Slope value (positive = improving, negative = declining)
    """
    from numpy.linalg import LinAlgError

    if len(dates) < 2:
        return 0.0

    # Check if all dates are the same (would cause date_nums to all be 0)
    if dates.nunique() == 1:
        return 0.0  # No trend if all dates are the same

    # Check if all scores are the same (no variation)
    if scores.nunique() == 1:
        return 0.0  # No trend if all scores are the same

    try:
        date_nums = [(d - dates.min()).days for d in dates]

        # Check if date_nums are all the same (shouldn't happen after above check, but defensive)
        if len(set(date_nums)) == 1:
            return 0.0

        coeffs = np.polyfit(date_nums, scores, 1)
        return float(coeffs[0])
    except (LinAlgError, ValueError, np.linalg.LinAlgError):
        # Handle numerical errors gracefully (SVD convergence issues, etc.)
        return 0.0


def predict_future_scores_simple(
    df: pd.DataFrame, days_ahead: int = 7
) -> Optional[Dict[str, Any]]:
    """
    Simple linear trend forecasting as fallback.

    Args:
        df: DataFrame with historical QA scores and dates
        days_ahead: Number of days to forecast

    Returns:
        Dictionary with forecast data (dates, forecast, lower_bound, upper_bound, method)
    """
    if "Call Date" not in df.columns or "QA Score" not in df.columns:
        return None

    daily_scores = df.groupby("Call Date")["QA Score"].mean().reset_index()
    daily_scores = daily_scores.sort_values("Call Date")

    if len(daily_scores) < 2:
        return None

    # Simple linear regression
    dates = daily_scores["Call Date"]
    scores = daily_scores["QA Score"]

    # Check if all dates are the same
    if dates.nunique() == 1:
        # All dates are the same - return flat forecast
        avg_score = float(scores.mean())
        return {
            "dates": [
                (dates.max() + timedelta(days=i + 1)).isoformat()[:10]
                for i in range(days_ahead)
            ],
            "forecast": [avg_score] * days_ahead,
            "lower_bound": [avg_score - 5] * days_ahead,  # Simple confidence interval
            "upper_bound": [avg_score + 5] * days_ahead,
            "method": "flat",
        }

    # Check if all scores are the same
    if scores.nunique() == 1:
        # All scores are the same - return flat forecast
        avg_score = float(scores.iloc[0])
        return {
            "dates": [
                (dates.max() + timedelta(days=i + 1)).isoformat()[:10]
                for i in range(days_ahead)
            ],
            "forecast": [avg_score] * days_ahead,
            "lower_bound": [avg_score - 5] * days_ahead,
            "upper_bound": [avg_score + 5] * days_ahead,
            "method": "flat",
        }

    # Convert dates to numeric for regression
    date_nums = [(d - dates.min()).days for d in dates]

    # Check if date_nums are all the same (defensive)
    if len(set(date_nums)) == 1:
        avg_score = float(scores.mean())
        return {
            "dates": [
                (dates.max() + timedelta(days=i + 1)).isoformat()[:10]
                for i in range(days_ahead)
            ],
            "forecast": [avg_score] * days_ahead,
            "lower_bound": [avg_score - 5] * days_ahead,
            "upper_bound": [avg_score + 5] * days_ahead,
            "method": "flat",
        }

    try:
        from numpy.linalg import LinAlgError

        # Linear regression
        coeffs = np.polyfit(date_nums, scores, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Predict future dates
        last_date = dates.max()
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(days_ahead)]
        forecast_nums = [(d - dates.min()).days for d in forecast_dates]
        forecast_scores = [float(slope * n + intercept) for n in forecast_nums]

        # Calculate confidence interval (simple std-based)
        residuals = scores - (slope * np.array(date_nums) + intercept)
        std_error = float(np.std(residuals))

        return {
            "dates": [d.isoformat()[:10] for d in forecast_dates],
            "forecast": forecast_scores,
            "lower_bound": [float(f - 1.96 * std_error) for f in forecast_scores],
            "upper_bound": [float(f + 1.96 * std_error) for f in forecast_scores],
            "method": "linear",
        }
    except (LinAlgError, ValueError, np.linalg.LinAlgError):
        # Handle numerical errors gracefully (SVD convergence issues, etc.)
        avg_score = float(scores.mean())
        return {
            "dates": [
                (dates.max() + timedelta(days=i + 1)).isoformat()[:10]
                for i in range(days_ahead)
            ],
            "forecast": [avg_score] * days_ahead,
            "lower_bound": [avg_score - 5] * days_ahead,
            "upper_bound": [avg_score + 5] * days_ahead,
            "method": "flat_fallback",
        }


def predict_future_scores(
    df: pd.DataFrame, days_ahead: int = 7
) -> Optional[Dict[str, Any]]:
    """
    Predict future QA scores using time series forecasting.

    Uses Prophet if available, otherwise falls back to simple linear trend.

    Args:
        df: DataFrame with historical QA scores and dates
        days_ahead: Number of days to forecast

    Returns:
        Dictionary with forecast data and confidence intervals, or None if insufficient data
    """
    try:
        from prophet import Prophet
    except ImportError:
        # Fallback to simple linear trend if Prophet not available
        return predict_future_scores_simple(df, days_ahead)

    if "Call Date" not in df.columns or "QA Score" not in df.columns:
        return None

    # Prepare data for Prophet
    daily_scores = df.groupby("Call Date")["QA Score"].mean().reset_index()
    daily_scores.columns = ["ds", "y"]
    daily_scores = daily_scores.sort_values("ds")

    if len(daily_scores) < 7:  # Need at least 7 days of data
        return predict_future_scores_simple(df, days_ahead)

    try:
        model = Prophet(
            interval_width=0.95, daily_seasonality=False, weekly_seasonality=True
        )
        model.fit(daily_scores)

        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)

        # Extract forecast for future dates only
        forecast_dates = forecast.tail(days_ahead)

        # Convert dates to ISO format strings for JSON serialization
        return {
            "dates": [d.isoformat()[:10] for d in forecast_dates["ds"].dt.date],
            "forecast": [float(x) for x in forecast_dates["yhat"].tolist()],
            "lower_bound": [float(x) for x in forecast_dates["yhat_lower"].tolist()],
            "upper_bound": [float(x) for x in forecast_dates["yhat_upper"].tolist()],
            "method": "prophet",
        }
    except Exception:
        # Fallback to simple method on any error
        return predict_future_scores_simple(df, days_ahead)


def identify_at_risk_agents(
    df: pd.DataFrame, threshold: float = 70.0, lookback_days: int = 14
) -> List[Dict[str, Any]]:
    """
    Identify agents at risk of dropping below threshold.

    Args:
        df: DataFrame with agent performance data
        threshold: QA score threshold
        lookback_days: Number of days to analyze for trend

    Returns:
        List of dictionaries with at-risk agent information
    """
    if (
        "Call Date" not in df.columns
        or "Agent" not in df.columns
        or "QA Score" not in df.columns
    ):
        return []

    cutoff_date = df["Call Date"].max() - timedelta(days=lookback_days)
    recent_data = df[df["Call Date"] >= cutoff_date]

    if recent_data.empty:
        return []

    at_risk = []

    for agent in recent_data["Agent"].unique():
        agent_data = recent_data[recent_data["Agent"] == agent].sort_values("Call Date")

        if len(agent_data) < 3:  # Need at least 3 data points
            continue

        # Calculate metrics
        recent_avg = float(agent_data["QA Score"].mean())
        trend_slope = calculate_trend_slope(
            agent_data["Call Date"], agent_data["QA Score"]
        )
        volatility = float(agent_data["QA Score"].std())
        proximity_to_threshold = threshold - recent_avg

        # Calculate risk score (0-100)
        # Made less sensitive: requires stronger signals or multiple risk factors
        risk_score = 0

        # Trend component (0-40 points) - stricter thresholds
        if trend_slope < -2:  # Strong declining trend (>2 points/day)
            risk_score += 40
        elif trend_slope < -1.5:  # Moderate-strong decline
            risk_score += 25
        elif trend_slope < -1:  # Moderate decline
            risk_score += 15

        # Volatility component (0-30 points) - higher thresholds
        if volatility > 20:  # Very high volatility
            risk_score += 30
        elif volatility > 15:  # High volatility
            risk_score += 18

        # Proximity component (0-30 points) - stricter thresholds
        if proximity_to_threshold <= 3:  # Very close to threshold (<3 points away)
            risk_score += 30
        elif proximity_to_threshold <= 5:  # Close to threshold
            risk_score += 18
        elif proximity_to_threshold <= 8:  # Moderately close
            risk_score += 10

        # Require higher risk score AND at least one strong signal
        # This prevents flagging agents with only moderate risk factors
        has_strong_trend = trend_slope < -1.5
        has_high_volatility = volatility > 15
        is_very_close = proximity_to_threshold <= 5

        if risk_score >= 65 and (
            has_strong_trend or has_high_volatility or is_very_close
        ):  # Higher threshold + requires strong signal
            at_risk.append(
                {
                    "agent": str(agent),
                    "risk_score": int(risk_score),
                    "recent_avg": recent_avg,
                    "trend_slope": trend_slope,
                    "volatility": volatility,
                    "proximity_to_threshold": proximity_to_threshold,
                    "recent_calls": int(len(agent_data)),
                }
            )

    # Sort by risk score
    at_risk.sort(key=lambda x: x["risk_score"], reverse=True)
    return at_risk


def classify_trajectory(
    df: pd.DataFrame, agent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify agent trajectory as improving, declining, stable, or volatile.

    Args:
        df: DataFrame with performance data
        agent: Optional agent ID to filter

    Returns:
        Dictionary with trajectory classification
    """
    if agent:
        df = df[df["Agent"] == agent]

    if "Call Date" not in df.columns or "QA Score" not in df.columns or len(df) < 3:
        return {"trajectory": "insufficient_data", "slope": 0.0, "volatility": 0.0}

    df_sorted = df.sort_values("Call Date")
    dates = df_sorted["Call Date"]
    scores = df_sorted["QA Score"]

    slope = calculate_trend_slope(dates, scores)
    volatility = float(scores.std())

    # Classify trajectory
    if volatility > 15:
        trajectory = "volatile"
    elif slope > 0.5:
        trajectory = "improving"
    elif slope < -0.5:
        trajectory = "declining"
    else:
        trajectory = "stable"

    # Projected score if trend continues
    last_score = float(scores.iloc[-1])
    projected_score = last_score + (slope * 7)  # Project 7 days ahead

    return {
        "trajectory": trajectory,
        "slope": slope,
        "volatility": volatility,
        "current_score": last_score,
        "projected_score": projected_score,
    }
