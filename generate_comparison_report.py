#!/usr/bin/env python3
"""
Contact Center Comparison Report Generator

Generates a PDF report comparing Jomashop's previous contact center performance
with BPO Centers performance after the transition date (July 7, 2025).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from pathlib import Path
import json
import argparse
import sys
import re
from typing import List, Dict, Tuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure matplotlib
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()
log_dir = PROJECT_ROOT / "logs"
CACHE_FILE = log_dir / "cached_calls_data.json"

# Transition date: July 7, 2025
TRANSITION_DATE = datetime(2025, 7, 7)

# Set style for professional-looking charts
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


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


def load_bpo_centers_data(cache_path: Path, transition_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load BPO Centers data from cache, split by transition date.
    
    Returns:
        Tuple of (before_data, after_data) DataFrames
    """
    print(f"📂 Loading BPO Centers data from: {cache_path}")
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        
        # Extract call data
        if isinstance(cached_data, dict):
            call_data = cached_data.get("call_data", cached_data.get("calls", cached_data.get("data", [])))
        else:
            call_data = cached_data
        
        if not call_data:
            raise ValueError("No call data found in cache file")
        
        print(f"✅ Loaded {len(call_data)} calls from cache")
        
        # Convert to DataFrame
        df = pd.DataFrame(call_data)
        
        # Normalize column names
        column_mapping = {
            "call_date": "Call Date",
            "agent": "Agent",
            "qa_score": "QA Score",
            "label": "Label",
            "rubric_pass_count": "Rubric Pass Count",
            "rubric_fail_count": "Rubric Fail Count",
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Ensure Call Date is datetime
        if "Call Date" in df.columns:
            if df["Call Date"].dtype == "object":
                df["Call Date"] = pd.to_datetime(df["Call Date"], errors="coerce")
        else:
            # Try to extract from other date fields
            if "date_raw" in df.columns:
                df["Call Date"] = pd.to_datetime(df["date_raw"], errors="coerce")
            else:
                raise ValueError("No date column found in BPO Centers data")
        
        # Normalize agent IDs
        if "Agent" in df.columns:
            df["Agent"] = df["Agent"].apply(normalize_agent_id)
        
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
                            except:
                                pass
                    return total_seconds / 60.0 if total_seconds > 0 else None
                return None
            
            df["AHT"] = df.apply(compute_aht, axis=1)
        
        # Split by transition date
        before_mask = df["Call Date"] < transition_date
        after_mask = df["Call Date"] >= transition_date
        
        before_data = df[before_mask].copy()
        after_data = df[after_mask].copy()
        
        print(f"📊 BPO Centers data split:")
        print(f"   Before {transition_date.strftime('%Y-%m-%d')}: {len(before_data)} calls")
        print(f"   After {transition_date.strftime('%Y-%m-%d')}: {len(after_data)} calls")
        
        return before_data, after_data
        
    except Exception as e:
        raise Exception(f"Error loading BPO Centers data: {e}")


def load_previous_center_data(csv_paths: List[Path], transition_date: datetime) -> pd.DataFrame:
    """
    Load and normalize previous contact center CSV files.
    
    Args:
        csv_paths: List of CSV file paths (one per agent)
        transition_date: Only include calls before this date
    
    Returns:
        Combined DataFrame with all previous contact center data
    """
    print(f"📂 Loading previous contact center data from {len(csv_paths)} CSV file(s)...")
    
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
            
            # Filter to dates before transition date
            valid_dates = df["Call Date"].notna()
            before_transition = df["Call Date"] < transition_date
            df = df[valid_dates & before_transition].copy()
            
            if len(df) == 0:
                print(f"   ⚠️  No calls before {transition_date.strftime('%Y-%m-%d')} in {csv_path.name}")
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
            rubric_columns = [col for col in df.columns 
                            if re.match(r"^\d+\.\d+\.\d+$", str(col)) 
                            and "__reason" not in str(col).lower()]
            
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
                    return pd.Series({"Rubric Pass Count": pass_count, "Rubric Fail Count": fail_count})
                
                rubric_counts = df.apply(count_rubric_results, axis=1)
                df["Rubric Pass Count"] = rubric_counts["Rubric Pass Count"]
                df["Rubric Fail Count"] = rubric_counts["Rubric Fail Count"]
            else:
                # If no rubric columns, set defaults
                df["Rubric Pass Count"] = 0
                df["Rubric Fail Count"] = 0
            
            # Add agent identifier from filename
            agent_name = csv_path.stem.replace("qa_results_", "").replace("_", " ").title()
            df["Agent"] = f"Previous Agent {i+1}" if not agent_name else agent_name
            
            all_dataframes.append(df)
            
        except Exception as e:
            print(f"⚠️  Error loading {csv_path.name}: {e}")
            continue
    
    if not all_dataframes:
        raise ValueError("No valid data loaded from CSV files")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"✅ Combined {len(combined_df)} calls from previous contact center")
    print(f"   Date range: {combined_df['Call Date'].min()} to {combined_df['Call Date'].max()}")
    
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
    """Calculate pass rate metrics."""
    metrics = {}
    
    # Method 1: From Rubric Pass/Fail Count
    if "Rubric Pass Count" in df.columns and "Rubric Fail Count" in df.columns:
        total_pass = df["Rubric Pass Count"].sum()
        total_fail = df["Rubric Fail Count"].sum()
        if (total_pass + total_fail) > 0:
            metrics["pass_rate"] = (total_pass / (total_pass + total_fail)) * 100
            metrics["pass_count"] = total_pass
            metrics["fail_count"] = total_fail
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
                pass_count = (valid_labels.str.lower() == "positive").sum()
                fail_count = (valid_labels.str.lower() == "negative").sum()
                total = pass_count + fail_count
                if total > 0:
                    metrics["pass_rate"] = (pass_count / total) * 100
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


def calculate_volume_metrics(df: pd.DataFrame) -> Dict:
    """Calculate call volume metrics."""
    metrics = {}
    
    # Total calls (exclude Invalid labels if present)
    if "Label" in df.columns:
        valid_calls = df[df["Label"].str.lower() != "invalid"] if df["Label"].dtype == "object" else df
        metrics["total_calls"] = len(valid_calls)
    else:
        metrics["total_calls"] = len(df)
    
    # Calls per day
    if "Call Date" in df.columns and metrics["total_calls"] > 0:
        date_range = (df["Call Date"].max() - df["Call Date"].min()).days + 1
        if date_range > 0:
            metrics["calls_per_day"] = metrics["total_calls"] / date_range
        else:
            metrics["calls_per_day"] = metrics["total_calls"]
    else:
        metrics["calls_per_day"] = None
    
    return metrics


def calculate_rubric_metrics(df: pd.DataFrame) -> Dict:
    """Calculate rubric failure metrics."""
    metrics = {}
    
    # Find rubric columns (exclude __reason columns)
    rubric_columns = [col for col in df.columns 
                     if re.match(r"^\d+\.\d+\.\d+$", str(col))
                     and "__reason" not in str(col).lower()]
    
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
        sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
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
        agent_perf = df.groupby("Agent")["QA Score"].agg(["mean", "count"]).reset_index()
        agent_perf.columns = ["Agent", "Avg Score", "Call Count"]
        agent_perf = agent_perf.sort_values("Avg Score", ascending=False)
        metrics["agent_performance"] = agent_perf
        metrics["num_agents"] = len(agent_perf)
    else:
        metrics["agent_performance"] = pd.DataFrame()
        metrics["num_agents"] = 0
    
    return metrics


def calculate_all_metrics(df: pd.DataFrame) -> Dict:
    """Calculate all KPIs for a dataset."""
    metrics = {}
    
    metrics.update(calculate_qa_metrics(df))
    metrics.update(calculate_pass_rate_metrics(df))
    metrics.update(calculate_aht_metrics(df))
    metrics.update(calculate_volume_metrics(df))
    metrics.update(calculate_rubric_metrics(df))
    metrics.update(calculate_agent_metrics(df))
    
    return metrics


def create_comparison_chart(metric_name: str, previous_value: float, bpo_value: float, 
                           previous_label: str = "Previous Contact Center",
                           bpo_label: str = "BPO Centers") -> plt.Figure:
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
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}' if isinstance(value, (int, float)) else 'N/A',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Calculate and display percentage change
    if previous_value and previous_value != 0:
        pct_change = ((bpo_value - previous_value) / previous_value) * 100
        change_text = f"{pct_change:+.1f}%"
        ax.text(0.5, 0.95, f"Change: {change_text}", 
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f"{metric_name} Comparison", fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
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
            # Green for positive improvement, red for negative
            colors_list.append("#2ecc71" if pct_change > 0 else "#e74c3c")
    
    if not metrics:
        ax.text(0.5, 0.5, "No improvement data available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig
    
    bars = ax.barh(metrics, pct_changes, color=colors_list, alpha=0.7, edgecolor="black", linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, pct_changes):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{value:+.1f}%',
                ha='left' if width > 0 else 'right', va='center', 
                fontsize=10, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel("Percentage Change (%)", fontsize=12, fontweight='bold')
    ax.set_title("Key Performance Improvements", fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def generate_pdf_report(previous_metrics: Dict, bpo_metrics: Dict, 
                       output_path: Path, transition_date: datetime):
    """Generate multi-page PDF comparison report."""
    print(f"\n📄 Generating PDF report: {output_path}")
    
    with PdfPages(output_path) as pdf:
        # Page 1: Title Page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title_text = "Contact Center Performance Comparison Report"
        subtitle_text = "Previous Contact Center vs BPO Centers"
        date_text = f"Transition Date: {transition_date.strftime('%B %d, %Y')}"
        generated_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        
        ax.text(0.5, 0.7, title_text, ha='center', va='center', 
                fontsize=24, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.6, subtitle_text, ha='center', va='center',
                fontsize=18, style='italic', transform=ax.transAxes)
        ax.text(0.5, 0.4, date_text, ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.3, generated_text, ha='center', va='center',
                fontsize=12, transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Executive Summary
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary_lines = ["EXECUTIVE SUMMARY", ""]
        
        # Calculate improvements
        improvements = {}
        if previous_metrics.get("avg_qa_score") and bpo_metrics.get("avg_qa_score"):
            pct = ((bpo_metrics["avg_qa_score"] - previous_metrics["avg_qa_score"]) / 
                   previous_metrics["avg_qa_score"]) * 100
            improvements["QA Score"] = pct
            summary_lines.append(f"• QA Score: {pct:+.1f}% change")
        
        if previous_metrics.get("pass_rate") and bpo_metrics.get("pass_rate"):
            pct = ((bpo_metrics["pass_rate"] - previous_metrics["pass_rate"]) / 
                   previous_metrics["pass_rate"]) * 100
            improvements["Pass Rate"] = pct
            summary_lines.append(f"• Pass Rate: {pct:+.1f}% change")
        
        if previous_metrics.get("avg_aht") and bpo_metrics.get("avg_aht"):
            pct = ((bpo_metrics["avg_aht"] - previous_metrics["avg_aht"]) / 
                   previous_metrics["avg_aht"]) * 100
            improvements["AHT"] = pct
            summary_lines.append(f"• Average Handle Time: {pct:+.1f}% change")
        
        summary_lines.extend(["", "KEY METRICS"])
        summary_lines.append(f"Previous Contact Center:")
        summary_lines.append(f"  • Total Calls: {previous_metrics.get('total_calls', 'N/A')}")
        summary_lines.append(f"  • Avg QA Score: {previous_metrics.get('avg_qa_score', 'N/A'):.1f}" if previous_metrics.get('avg_qa_score') else "  • Avg QA Score: N/A")
        summary_lines.append(f"  • Pass Rate: {previous_metrics.get('pass_rate', 'N/A'):.1f}%" if previous_metrics.get('pass_rate') else "  • Pass Rate: N/A")
        summary_lines.append(f"  • Avg AHT: {previous_metrics.get('avg_aht', 'N/A'):.1f} min" if previous_metrics.get('avg_aht') else "  • Avg AHT: N/A")
        
        summary_lines.append("")
        summary_lines.append(f"BPO Centers:")
        summary_lines.append(f"  • Total Calls: {bpo_metrics.get('total_calls', 'N/A')}")
        summary_lines.append(f"  • Avg QA Score: {bpo_metrics.get('avg_qa_score', 'N/A'):.1f}" if bpo_metrics.get('avg_qa_score') else "  • Avg QA Score: N/A")
        summary_lines.append(f"  • Pass Rate: {bpo_metrics.get('pass_rate', 'N/A'):.1f}%" if bpo_metrics.get('pass_rate') else "  • Pass Rate: N/A")
        summary_lines.append(f"  • Avg AHT: {bpo_metrics.get('avg_aht', 'N/A'):.1f} min" if bpo_metrics.get('avg_aht') else "  • Avg AHT: N/A")
        
        summary_text = "\n".join(summary_lines)
        ax.text(0.1, 0.95, summary_text, ha='left', va='top',
                fontsize=11, family='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3+: Metric Comparison Charts
        if previous_metrics.get("avg_qa_score") and bpo_metrics.get("avg_qa_score"):
            fig = create_comparison_chart("Average QA Score", 
                                         previous_metrics["avg_qa_score"],
                                         bpo_metrics["avg_qa_score"])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        if previous_metrics.get("pass_rate") and bpo_metrics.get("pass_rate"):
            fig = create_comparison_chart("Pass Rate (%)",
                                         previous_metrics["pass_rate"],
                                         bpo_metrics["pass_rate"])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        if previous_metrics.get("avg_aht") and bpo_metrics.get("avg_aht"):
            fig = create_comparison_chart("Average Handle Time (minutes)",
                                         previous_metrics["avg_aht"],
                                         bpo_metrics["avg_aht"])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        if previous_metrics.get("total_calls") and bpo_metrics.get("total_calls"):
            fig = create_comparison_chart("Total Call Volume",
                                         previous_metrics["total_calls"],
                                         bpo_metrics["total_calls"])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Improvement Summary Chart
        if improvements:
            fig = create_improvement_summary_chart(improvements)
            pdf.savefig(fig, bbox_inches='tight')
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
        help="Path(s) to previous contact center CSV file(s)"
    )
    parser.add_argument(
        "--previous-csv-dir",
        type=str,
        help="Directory containing previous contact center CSV files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.pdf",
        help="Output PDF file path (default: comparison_report.pdf)"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(CACHE_FILE),
        help=f"Path to BPO Centers cache file (default: {CACHE_FILE})"
    )
    parser.add_argument(
        "--transition-date",
        type=str,
        default="2025-07-07",
        help="Transition date in YYYY-MM-DD format (default: 2025-07-07)"
    )
    
    args = parser.parse_args()
    
    # Parse transition date
    try:
        transition_date = datetime.strptime(args.transition_date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid transition date format: {args.transition_date}")
        print("   Use YYYY-MM-DD format (e.g., 2025-07-07)")
        sys.exit(1)
    
    # Determine CSV file paths
    csv_paths = []
    if args.previous_csv:
        csv_paths = [Path(p) for p in args.previous_csv]
    elif args.previous_csv_dir:
        csv_dir = Path(args.previous_csv_dir)
        if not csv_dir.exists():
            print(f"❌ CSV directory not found: {csv_dir}")
            sys.exit(1)
        csv_paths = list(csv_dir.glob("*.csv"))
        if not csv_paths:
            print(f"❌ No CSV files found in directory: {csv_dir}")
            sys.exit(1)
    else:
        print("❌ Error: Must provide either --previous-csv or --previous-csv-dir")
        parser.print_help()
        sys.exit(1)
    
    print("=" * 80)
    print("Contact Center Comparison Report Generator")
    print("=" * 80)
    print()
    
    try:
        # Load data
        print("📊 Loading data...")
        previous_data = load_previous_center_data(csv_paths, transition_date)
        bpo_before, bpo_after = load_bpo_centers_data(Path(args.cache), transition_date)
        
        # Calculate metrics
        print("\n📈 Calculating metrics...")
        previous_metrics = calculate_all_metrics(previous_data)
        bpo_metrics = calculate_all_metrics(bpo_after)  # Compare with "after" period
        
        # Generate report
        output_path = Path(args.output)
        generate_pdf_report(previous_metrics, bpo_metrics, output_path, transition_date)
        
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

