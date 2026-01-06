#!/usr/bin/env python3
"""
Script to find calls with unknown/missing agent IDs.
Outputs results to terminal and saves to a CSV file.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()
log_dir = PROJECT_ROOT / "logs"
CACHE_FILE = log_dir / "cached_calls_data.json"


def normalize_agent_id(agent_str):
    """Normalize agent ID to bpagent## format (e.g., bpagent01, bpagent02)"""
    if pd.isna(agent_str) or not agent_str:
        return agent_str

    agent_str = str(agent_str).lower().strip()

    # Special case: "unknown" -> Agent 01 (Jesus)
    if agent_str == "unknown":
        return "bpagent01"

    # Special case: bp016803073 and bp016803074 -> Agent 01 (Jesus)
    if agent_str in ["bp016803073", "bp016803074"]:
        return "bpagent01"

    # Special case: Any string starting with "bp01" (first 4 chars) -> Agent 01 (Jesus)
    if agent_str.startswith("bp01"):
        return "bpagent01"

    # Extract number from bpagent### pattern
    match = re.search(r"bpagent(\d+)", agent_str)
    if match:
        agent_num = match.group(1)
        if len(agent_num) >= 2:
            agent_num = agent_num[:2]
        else:
            agent_num = agent_num.zfill(2)
        return f"bpagent{agent_num}"

    return agent_str


def main():
    print("=" * 80)
    print("Unknown Agent Calls Analysis")
    print("=" * 80)
    print()

    # Check if cache file exists
    if not CACHE_FILE.exists():
        print(f"❌ Cache file not found: {CACHE_FILE}")
        print(
            "   Make sure the Streamlit app has been run at least once to create the cache."
        )
        return

    print(f"📂 Loading cache from: {CACHE_FILE}")

    try:
        # Load cached data
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached_data = json.load(f)

        # Extract call data (could be in 'calls', 'call_data', 'data' key or be the list itself)
        if isinstance(cached_data, dict):
            call_data = cached_data.get(
                "call_data", cached_data.get("calls", cached_data.get("data", []))
            )
        else:
            call_data = cached_data

        if not call_data:
            print("❌ No call data found in cache file")
            return

        print(f"✅ Loaded {len(call_data)} calls from cache")
        print()

        # Convert to DataFrame
        df = pd.DataFrame(call_data)

        # Normalize column names (in case they're lowercase)
        if "agent" in df.columns and "Agent" not in df.columns:
            df.rename(columns={"agent": "Agent"}, inplace=True)

        if "Agent" not in df.columns:
            print("❌ 'Agent' column not found in data")
            print(f"   Available columns: {list(df.columns)}")
            return

        # Find calls with unknown/missing agents BEFORE normalization
        # (normalization converts "unknown" to "bpagent01", making this check unreachable)
        print("🔍 Searching for unknown agent calls...")
        unknown_agent_calls = df[
            df["Agent"].isna()
            | (df["Agent"].astype(str).str.strip() == "")
            | (df["Agent"].astype(str).str.strip().str.lower() == "nan")
            | (df["Agent"].astype(str).str.strip().str.lower() == "none")
            | (df["Agent"].astype(str).str.strip().str.lower() == "unknown")
        ].copy()

        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()

        if len(unknown_agent_calls) == 0:
            print("✅ All calls have agent IDs assigned!")
            return

        print(
            f"⚠️  Found {len(unknown_agent_calls)} call(s) with unknown/missing agent IDs"
        )
        print()

        # Show unique agent values
        unique_agent_values = unknown_agent_calls["Agent"].unique()
        print(f"📊 Unique Agent Values in Unknown Calls: {len(unique_agent_values)}")
        for val in unique_agent_values:
            print(f"   - {val}")
        print()

        # Prepare output columns - normalize column names first
        column_mapping = {}
        if (
            "call_id" in unknown_agent_calls.columns
            and "Call ID" not in unknown_agent_calls.columns
        ):
            column_mapping["call_id"] = "Call ID"
        if (
            "filename" in unknown_agent_calls.columns
            and "Filename" not in unknown_agent_calls.columns
        ):
            column_mapping["filename"] = "Filename"
        if (
            "call_date" in unknown_agent_calls.columns
            and "Call Date" not in unknown_agent_calls.columns
        ):
            column_mapping["call_date"] = "Call Date"

        if column_mapping:
            unknown_agent_calls.rename(columns=column_mapping, inplace=True)

        # Build output columns list
        output_cols = []
        for col in ["Call ID", "Filename", "Call Date", "_s3_key", "Agent"]:
            if col in unknown_agent_calls.columns:
                output_cols.append(col)

        # Display summary table
        display_df = (
            unknown_agent_calls[output_cols].copy()
            if output_cols
            else unknown_agent_calls.copy()
        )

        print("📋 Unknown Agent Calls:")
        print()
        print(display_df.to_string(index=False))
        print()

        # Save to CSV
        output_file = (
            PROJECT_ROOT
            / f"unknown_agent_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        display_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        print()

        # Additional statistics
        print("=" * 80)
        print("STATISTICS")
        print("=" * 80)
        print(f"Total calls in dataset: {len(df)}")
        print(f"Unknown agent calls: {len(unknown_agent_calls)}")
        print(f"Percentage: {len(unknown_agent_calls) / len(df) * 100:.2f}%")
        print()

        # Show date range if available
        if "Call Date" in unknown_agent_calls.columns:
            try:
                unknown_agent_calls["Call Date"] = pd.to_datetime(
                    unknown_agent_calls["Call Date"], errors="coerce"
                )
                date_range = unknown_agent_calls["Call Date"].dropna()
                if len(date_range) > 0:
                    print(f"Date range of unknown calls:")
                    print(f"   Earliest: {date_range.min()}")
                    print(f"   Latest: {date_range.max()}")
                    print()
            except Exception:
                # Skip date range if parsing fails
                pass

    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON cache file: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
