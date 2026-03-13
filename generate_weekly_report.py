#!/usr/bin/env python3
"""
Weekly QA Report Generator

Generates an exportable PDF report for a given week, including:
- Agent leaderboard
- Pass rates and QA scores over time (all historical data for progress comparison)
- Call Reason / Outcome / Product distributions for the week
- Coaching suggestions per agent (based on most common rubric failures)
- Failing calls per agent (Reason, Outcome, Product, QA Score)

Built to run locally (not in Streamlit). Uses cache or S3 month caches for data.
Identical structure to monthly report, adapted for weekly granularity.
"""

import os
import sys
import json
import re
import argparse
import textwrap
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import io
import tempfile

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_FILE = LOG_DIR / "cached_calls_data.json"
PASS_THRESHOLD = 70

# Report layout: consistent page size (letter) and table font
PAGE_WIDTH, PAGE_HEIGHT = 8.5, 11
TABLE_FONTSIZE = 12

# Month options (must match streamlit month_options for S3 cache keys)
MONTHS_AVAILABLE = [
    (2026, 3), (2026, 2), (2026, 1),
    (2025, 12), (2025, 11), (2025, 10), (2025, 9), (2025, 8), (2025, 7),
]


def _get_s3_config() -> Optional[Dict]:
    """Get S3 config from .streamlit/secrets.toml or env."""
    config = {}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if tomllib and secrets_path.exists():
        try:
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
            s3 = secrets.get("s3", {})
            if s3.get("bucket_name"):
                config["bucket"] = s3["bucket_name"]
                config["prefix"] = s3.get("prefix", "cache").rstrip("/")
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


def _load_from_disk_cache() -> List[Dict]:
    """Load call data from disk cache."""
    if not CACHE_FILE.exists():
        return []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("call_data", data.get("calls", data.get("data", [])))
    return data if isinstance(data, list) else []


def _load_from_s3_month(year: int, month: int) -> List[Dict]:
    """Load call data for a specific month from S3."""
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
        key = f"{prefix}cached_calls_data_{year}_{month:02d}.json"
        resp = client.get_object(Bucket=config["bucket"], Key=key)
        data = json.loads(resp["Body"].read().decode("utf-8"))
        if isinstance(data, dict):
            return data.get("call_data", data.get("calls", data.get("data", [])))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_all_data() -> List[Dict]:
    """Load all call data: from disk cache first, or by merging S3 month caches."""
    calls = _load_from_disk_cache()
    if calls:
        return calls
    # Fallback: load from S3 month caches
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
        all_calls = []
        seen_ids = set()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config["bucket"], Prefix=prefix_full):
            for obj in (page.get("Contents") or []):
                key = obj["Key"]
                if not key.endswith(".json") or "cached_calls_data_" not in key:
                    continue
                try:
                    resp = client.get_object(Bucket=config["bucket"], Key=key)
                    data = json.loads(resp["Body"].read().decode("utf-8"))
                    if isinstance(data, dict):
                        c = data.get("call_data", data.get("calls", data.get("data", [])))
                    else:
                        c = data if isinstance(data, list) else []
                    for call in c:
                        cid = call.get("Call ID") or call.get("call_id") or str(call.get("_id", ""))
                        if cid and cid not in seen_ids:
                            seen_ids.add(cid)
                            all_calls.append(call)
                        elif not cid:
                            all_calls.append(call)
                except Exception:
                    continue
        return all_calls
    except Exception:
        return []


def normalize_agent_id(agent_str) -> str:
    """Normalize agent ID to consistent format."""
    if pd.isna(agent_str) or not agent_str:
        return str(agent_str) if agent_str else "Unknown"
    s = str(agent_str).lower().strip()
    if s.startswith("agent "):
        num = s.replace("agent ", "").strip()
        try:
            return f"Agent {int(num)}"
        except ValueError:
            pass
    match = re.search(r"bpagent(\d{2})", s.replace(" ", "").replace("_", ""))
    if match:
        return f"Agent {int(match.group(1))}"
    return str(agent_str).strip() or "Unknown"


def _agent_sort_key(agent: str) -> Tuple[int, str]:
    """Sort key for agents: Agent 1, Agent 2, ... Agent 10, Agent 11."""
    m = re.search(r"agent\s*(\d+)", str(agent).lower())
    return (int(m.group(1)) if m else 999, str(agent))


def _prepare_dataframe(calls: List[Dict]) -> pd.DataFrame:
    """Convert call list to DataFrame with normalized columns."""
    if not calls:
        return pd.DataFrame()
    df = pd.DataFrame(calls)
    rename = {
        "call_date": "Call Date", "agent": "Agent", "qa_score": "QA Score",
        "reason": "Reason", "outcome": "Outcome", "summary": "Summary",
        "rubric_details": "Rubric Details", "coaching_suggestions": "Coaching Suggestions",
        "challenges": "Challenges", "time": "Time",
    }
    cols = {c.lower(): c for c in df.columns}
    for old_lower, new in rename.items():
        if old_lower in cols and (new not in df.columns or cols[old_lower] != new):
            df = df.rename(columns={cols[old_lower]: new})
    if "Call Date" in df.columns:
        df["Call Date"] = pd.to_datetime(df["Call Date"], errors="coerce")
    elif "date_raw" in df.columns:
        df["Call Date"] = pd.to_datetime(df["date_raw"], format="%m%d%Y", errors="coerce")
    if "Agent" in df.columns:
        df["Agent"] = df["Agent"].apply(normalize_agent_id)
    if "QA Score" in df.columns:
        df["QA Score"] = pd.to_numeric(df["QA Score"], errors="coerce")
    # Compute Call Duration (min) from speaking_time_per_speaker
    if "speaking_time_per_speaker" in df.columns:
        def to_minutes(x):
            if isinstance(x, dict) and "total" in x:
                t = x["total"]
                if ":" in str(t):
                    parts = str(t).split(":")
                    try:
                        return int(parts[0]) + int(parts[1]) / 60 if len(parts) >= 2 else int(parts[0])
                    except (ValueError, IndexError):
                        return np.nan
            return np.nan
        df["Call Duration (min)"] = df["speaking_time_per_speaker"].apply(to_minutes)
    # Extract hour for time-of-day analysis (from Time "HH:MM:SS" or Call ID "YYYYMMDD_HHMMSS_...")
    def _get_hour(row):
        t = row.get("Time") or row.get("time")
        if pd.notna(t) and t:
            s = str(t).strip()
            if len(s) >= 2 and ":" in s:
                try:
                    return int(s.split(":")[0])
                except ValueError:
                    pass
        cid = str(row.get("Call ID") or row.get("call_id") or "")
        if "_" in cid and len(cid) >= 15:
            try:
                return int(cid[9:11])
            except (ValueError, IndexError):
                pass
        return np.nan
    df["Hour"] = df.apply(_get_hour, axis=1)
    return df.dropna(subset=["Call Date"], how="all")


def extract_products(row: pd.Series) -> str:
    """Extract product mentions from Reason, Summary, Outcome."""
    text = " ".join([
        str(row.get("Reason", "") or ""),
        str(row.get("Summary", "") or ""),
        str(row.get("Outcome", "") or ""),
    ]).lower()
    products = []
    for kw in ["watch", "jewelry", "rolex", "omega", "cartier", "galaxy s24", "galaxy watch"]:
        if kw in text:
            products.append(kw.title())
    return ", ".join(products[:5]) if products else "—"


def extract_products_list(row: pd.Series) -> List[str]:
    """Extract product mentions as list for distribution counting."""
    text = " ".join([
        str(row.get("Reason", "") or ""),
        str(row.get("Summary", "") or ""),
        str(row.get("Outcome", "") or ""),
    ]).lower()
    products = []
    for kw in ["watch", "jewelry", "rolex", "omega", "cartier", "galaxy s24", "galaxy watch"]:
        if kw in text:
            products.append(kw.title())
    return products


def categorize_coaching(text: str) -> str:
    """Categorize coaching/challenge text into root cause themes."""
    if not text or pd.isna(text):
        return "Other"
    t = str(text).lower()
    scores = {
        "Product Knowledge": sum(1 for kw in ["product", "knowledge", "unfamiliar", "specification"] if kw in t),
        "Communication": sum(1 for kw in ["communication", "clarity", "explain", "empathy", "tone"] if kw in t),
        "Hold Time": sum(1 for kw in ["hold", "transfer", "escalate", "wait"] if kw in t),
        "System/Technical": sum(1 for kw in ["system", "technical", "error", "software", "slow"] if kw in t),
        "Process/Procedure": sum(1 for kw in ["process", "procedure", "policy", "workflow"] if kw in t),
        "Efficiency": sum(1 for kw in ["efficient", "streamline", "reduce time", "faster"] if kw in t),
    }
    m = max(scores.values())
    return max(scores, key=scores.get) if m > 0 else "Other"


def _load_rubric_labels() -> Dict[str, str]:
    """Load rubric code -> human-readable item mapping from Rubric_v33.json."""
    rubric_path = PROJECT_ROOT / "Rubric_v33.json"
    labels = {}
    if rubric_path.exists():
        try:
            with open(rubric_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in (data if isinstance(data, list) else []):
                code = entry.get("code")
                item = entry.get("item")
                if code and item:
                    labels[str(code).strip()] = str(item).strip()
        except Exception:
            pass
    return labels


def _set_table_fontsize(table, fontsize: int = TABLE_FONTSIZE):
    """Apply font size to all table cells."""
    for cell in table.get_celld().values():
        cell.set_fontsize(fontsize)


def _para(text: str, style) -> Paragraph:
    """Escape and wrap text for ReportLab Paragraph (preserves <br/>)."""
    if text is None or not str(text).strip():
        return Paragraph("—", style)
    s = str(text).strip()
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(s, style)


def _para_html(html: str, style) -> Paragraph:
    """Paragraph from HTML (ReportLab supports <br/>, <b>, etc.)."""
    s = (html or "").replace("&", "&amp;")
    return Paragraph(s, style)


def _build_agent_pages_reportlab(
    week_df: pd.DataFrame,
    rubric_labels: Dict[str, str],
    agent_failures: Dict,
    failing: pd.DataFrame,
    all_agents: List[str],
    week_label: str,
    output_buffer: io.BytesIO,
):
    """Build agent coaching + failing calls pages with ReportLab (proper table wrapping)."""
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        leftIndent=0,
        rightIndent=0,
        spaceBefore=2,
        spaceAfter=2,
        alignment=TA_LEFT,
    )
    header_style = ParagraphStyle(
        "Header",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        fontName="Helvetica-Bold",
    )
    col_widths = [1.4 * inch, 0.55 * inch, 0.5 * inch, 2.1 * inch, 2.1 * inch, 0.6 * inch]

    doc = SimpleDocTemplate(
        output_buffer,
        pagesize=letter,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.75 * inch,
    )
    story = []
    id_col = "Call ID" if "Call ID" in failing.columns else "call_id"
    dur_col = "Call Duration (min)" if "Call Duration (min)" in failing.columns else None

    for agent_idx, agent in enumerate(all_agents):
        fails = agent_failures.get(agent, {})
        top_coaching = sorted(fails.items(), key=lambda x: -x[1])[:5]
        agent_fails_df = failing[failing["Agent"] == agent]

        if len(agent_fails_df) == 0 and not top_coaching:
            continue

        is_last_agent = agent_idx == len(all_agents) - 1

        title = f"{agent} — Coaching & Failing Calls ({week_label})"
        story.append(Paragraph(f"<b>{title}</b>", ParagraphStyle("Title", fontSize=14, spaceAfter=6)))
        story.append(Spacer(1, 4))

        if top_coaching:
            coach_text = "Coaching focus (top rubric failures):<br/>"
            for code, cnt in top_coaching:
                label = rubric_labels.get(str(code).strip(), code)
                lab = str(label).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                coach_text += f"  • {lab}: {cnt} failures<br/>"
            story.append(_para_html(coach_text, cell_style))
            story.append(Spacer(1, 8))

        if len(agent_fails_df) > 0:
            headers = ["Call ID", "Handle (min)", "QA Score", "Reason", "Outcome", "Product"]
            data = [[_para(h, header_style) for h in headers]]
            for _, row in agent_fails_df.iterrows():
                cid = str(row.get(id_col, row.get("call_id", "—")))
                dur = row.get(dur_col)
                handle = f"{dur:.1f}" if isinstance(dur, (int, float)) and not pd.isna(dur) else "—"
                score = row.get("QA Score", "—")
                if isinstance(score, (int, float)) and not pd.isna(score):
                    score = f"{score:.0f}%"
                else:
                    score = str(score)
                reason = str(row.get("Reason", "") or "")
                outcome = str(row.get("Outcome", "") or "")
                product = str(row.get("Product", "") or "—")
                data.append([
                    _para(cid, cell_style),
                    _para(handle, cell_style),
                    _para(score, cell_style),
                    _para(reason, cell_style),
                    _para(outcome, cell_style),
                    _para(product, cell_style),
                ])
            tbl = Table(data, colWidths=col_widths, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(tbl)

        if not is_last_agent:
            story.append(PageBreak())

    doc.build(story)


def generate_report(week_df: pd.DataFrame, all_df: pd.DataFrame, week_start: datetime, output_path: Path):
    """Generate the weekly report PDF."""
    week_end = week_start + timedelta(days=6)
    week_label = f"{week_start.strftime('%b %d')}–{week_end.strftime('%d, %Y')}"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        charts_path = Path(tmp.name)
    try:
        with PdfPages(charts_path, metadata={"Creator": "jomashop-dashboard"}) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, f"Weekly QA Report — {week_label}", fontsize=24, ha="center", va="center")
            fig.subplots_adjust(bottom=0.06)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

            # 1. Agent Leaderboard (week)
            if len(week_df) > 0 and "Agent" in week_df.columns:
                call_col = "Call ID" if "Call ID" in week_df.columns else "call_id"
                id_col = call_col if call_col in week_df.columns else None
                if id_col:
                    agg = week_df.groupby("Agent").agg(
                        Total_Calls=(id_col, "nunique"),
                        Avg_QA_Score=("QA Score", "mean"),
                        Pass_Rate=("QA Score", lambda s: (s >= PASS_THRESHOLD).mean() * 100),
                    ).reset_index()
                else:
                    agg = week_df.groupby("Agent").agg(
                        Total_Calls=("QA Score", "count"),
                        Avg_QA_Score=("QA Score", "mean"),
                        Pass_Rate=("QA Score", lambda s: (s >= PASS_THRESHOLD).mean() * 100),
                    ).reset_index()
                agg = agg.sort_values("Avg_QA_Score", ascending=False)
                cell_data = []
                for _, row in agg.iterrows():
                    r = []
                    for c in agg.columns:
                        v = row[c]
                        if pd.isna(v):
                            r.append("—")
                        elif c == "Total_Calls" and isinstance(v, (int, float)):
                            r.append(str(int(v)))
                        elif c in ("Avg_QA_Score", "Pass_Rate") and isinstance(v, (int, float)):
                            r.append(f"{v:.1f}")
                        else:
                            r.append(str(v))
                    cell_data.append(r)
                fig, ax = plt.subplots(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
                ax.axis("off")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                tbl = ax.table(
                    cellText=cell_data,
                    colLabels=agg.columns.tolist(),
                    loc="center",
                    cellLoc="center",
                )
                _set_table_fontsize(tbl)
                ax.set_title(f"Agent Leaderboard — {week_label}", fontsize=16)
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.06)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

            # 2. Pass rates and QA over time (all data, by week)
            if len(all_df) > 0 and "Call Date" in all_df.columns:
                all_df = all_df.copy()
                all_df["Week"] = all_df["Call Date"].dt.to_period("W-SUN").astype(str)
                weekly = all_df.groupby("Week").agg(
                    Pass_Rate=("QA Score", lambda s: (s >= PASS_THRESHOLD).mean() * 100),
                    Avg_QA=("QA Score", "mean"),
                    Calls=("QA Score", "count"),
                ).reset_index()
                weekly = weekly.sort_values("Week")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PAGE_WIDTH, PAGE_HEIGHT), sharex=True)
                for ax in (ax1, ax2):
                    ax.tick_params(axis="both", labelsize=TABLE_FONTSIZE)
                ax1.plot(weekly["Week"], weekly["Pass_Rate"], marker="o", label="Pass Rate %", linewidth=2, markersize=8)
                ax1.set_ylabel("Pass Rate %", fontsize=TABLE_FONTSIZE)
                ax1.set_title("Pass Rate Over Time (All Data)", fontsize=14)
                ax1.legend(fontsize=TABLE_FONTSIZE)
                ax1.grid(True, alpha=0.3)
                ax2.plot(weekly["Week"], weekly["Avg_QA"], marker="o", color="green", label="Avg QA Score", linewidth=2, markersize=8)
                ax2.set_ylabel("Avg QA Score", fontsize=TABLE_FONTSIZE)
                ax2.set_xlabel("Week", fontsize=TABLE_FONTSIZE)
                ax2.set_title("QA Score Over Time (All Data)", fontsize=14)
                ax2.legend(fontsize=TABLE_FONTSIZE)
                ax2.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.06)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

            # 3. Call Reason / Outcome / Product distributions (week)
            if len(week_df) > 0:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(PAGE_WIDTH, PAGE_HEIGHT), sharex=False)

                if "Reason" in week_df.columns:
                    reasons = week_df["Reason"].dropna().astype(str).str.strip().str.lower()
                    reasons = reasons[reasons != ""]
                    if len(reasons) > 0:
                        reason_counts = reasons.value_counts().head(12)
                        ax1.barh(range(len(reason_counts)), reason_counts.values, color="steelblue", alpha=0.8)
                        ax1.set_yticks(range(len(reason_counts)))
                        ax1.set_yticklabels([str(s)[:50] + ("…" if len(str(s)) > 50 else "") for s in reason_counts.index], fontsize=TABLE_FONTSIZE)
                        ax1.invert_yaxis()
                ax1.set_xlabel("Count", fontsize=TABLE_FONTSIZE)
                ax1.set_title(f"Call Reason Distribution — {week_label}", fontsize=14)
                ax1.tick_params(axis="both", labelsize=TABLE_FONTSIZE)

                if "Outcome" in week_df.columns:
                    outcomes = week_df["Outcome"].dropna().astype(str).str.strip().str.lower()
                    outcomes = outcomes[outcomes != ""]
                    if len(outcomes) > 0:
                        outcome_counts = outcomes.value_counts().head(12)
                        ax2.barh(range(len(outcome_counts)), outcome_counts.values, color="seagreen", alpha=0.8)
                        ax2.set_yticks(range(len(outcome_counts)))
                        ax2.set_yticklabels([str(s)[:50] + ("…" if len(str(s)) > 50 else "") for s in outcome_counts.index], fontsize=TABLE_FONTSIZE)
                        ax2.invert_yaxis()
                ax2.set_xlabel("Count", fontsize=TABLE_FONTSIZE)
                ax2.set_title(f"Call Outcome Distribution — {week_label}", fontsize=14)
                ax2.tick_params(axis="both", labelsize=TABLE_FONTSIZE)

                all_products = []
                for _, row in week_df.iterrows():
                    all_products.extend(extract_products_list(row))
                if all_products:
                    product_counts = Counter(all_products)
                    top_products = product_counts.most_common(12)
                    ax3.barh(range(len(top_products)), [c for _, c in top_products],
                             color="coral", alpha=0.8)
                    ax3.set_yticks(range(len(top_products)))
                    ax3.set_yticklabels([p for p, _ in top_products], fontsize=TABLE_FONTSIZE)
                    ax3.invert_yaxis()
                ax3.set_xlabel("Count (mentions)", fontsize=TABLE_FONTSIZE)
                ax3.set_title(f"Product Distribution — {week_label}", fontsize=14)
                ax3.tick_params(axis="both", labelsize=TABLE_FONTSIZE)

                fig.tight_layout()
                fig.subplots_adjust(bottom=0.06)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

            # 4. QA scores and pass rates by time of day (week)
            if len(week_df) > 0 and "Hour" in week_df.columns and "QA Score" in week_df.columns:
                by_hour = week_df[week_df["Hour"].notna()].copy()
                if len(by_hour) > 0:
                    hourly = by_hour.groupby("Hour").agg(
                        Avg_QA=("QA Score", "mean"),
                        Pass_Rate=("QA Score", lambda s: (s >= PASS_THRESHOLD).mean() * 100),
                        Count=("QA Score", "count"),
                    )
                    hourly = hourly[hourly["Count"] > 0]
                    if len(hourly) > 0:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PAGE_WIDTH, PAGE_HEIGHT), sharex=True)
                        hours = hourly.index
                        ax1.bar(hours, hourly["Avg_QA"], color="steelblue", alpha=0.8)
                        ax1.axhline(y=PASS_THRESHOLD, color="gray", linestyle="--", alpha=0.7)
                        ax1.set_ylabel("Avg QA Score", fontsize=TABLE_FONTSIZE)
                        ax1.set_title(f"QA Score by Time of Day — {week_label}", fontsize=14)
                        ax1.set_ylim(0, 100)
                        ax1.tick_params(axis="both", labelsize=TABLE_FONTSIZE)
                        ax2.bar(hours, hourly["Pass_Rate"], color="seagreen", alpha=0.8)
                        ax2.axhline(y=PASS_THRESHOLD, color="gray", linestyle="--", alpha=0.7)
                        ax2.set_ylabel("Pass Rate %", fontsize=TABLE_FONTSIZE)
                        ax2.set_xlabel("Hour of Day", fontsize=TABLE_FONTSIZE)
                        ax2.set_title(f"Pass Rate by Time of Day — {week_label}", fontsize=14)
                        ax2.set_ylim(0, 100)
                        ax2.tick_params(axis="both", labelsize=TABLE_FONTSIZE)
                        fig.tight_layout()
                        fig.subplots_adjust(bottom=0.06)
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close()

        rubric_labels = _load_rubric_labels() if (PROJECT_ROOT / "Rubric_v33.json").exists() else {}
        agent_failures = defaultdict(lambda: defaultdict(int))
        if "Rubric Details" in week_df.columns:
            for _, row in week_df.iterrows():
                agent = row.get("Agent", "Unknown")
                rd = row.get("Rubric Details", {})
                if isinstance(rd, dict):
                    for code, det in rd.items():
                        if isinstance(det, dict) and det.get("status", "").lower() == "fail":
                            agent_failures[agent][code] += 1
        failing = week_df[week_df["QA Score"] < PASS_THRESHOLD] if "QA Score" in week_df.columns else pd.DataFrame()
        if len(failing) > 0:
            failing = failing.copy()
            failing["Product"] = failing.apply(extract_products, axis=1)
        all_agents = sorted(
            set(agent_failures.keys()) | (set(failing["Agent"].unique()) if len(failing) > 0 else set()),
            key=_agent_sort_key,
        )

        agents_buffer = io.BytesIO()
        _build_agent_pages_reportlab(
            week_df, rubric_labels, agent_failures, failing, all_agents, week_label, agents_buffer
        )

        from pypdf import PdfWriter, PdfReader
        writer = PdfWriter()
        for reader in [PdfReader(charts_path), PdfReader(agents_buffer)]:
            for page in reader.pages:
                writer.add_page(page)
        with open(output_path, "wb") as f:
            writer.write(f)
    finally:
        charts_path.unlink(missing_ok=True)

    print(f"✅ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate weekly QA report (PDF)")
    parser.add_argument("--week", type=str, required=True, help="Week start in YYYY-MM-DD format (e.g. 2026-03-01 for first week of March)")
    parser.add_argument("--output", type=str, default=None, help="Output PDF path (default: weekly_report_YYYY-MM-DD.pdf)")
    parser.add_argument("--cache", type=str, default=str(CACHE_FILE), help="Path to cache file (for disk cache)")
    args = parser.parse_args()

    try:
        week_start = datetime.strptime(args.week, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid --week format. Use YYYY-MM-DD (e.g. 2026-03-01)")
        sys.exit(1)

    year, month = week_start.year, week_start.month
    if (year, month) not in MONTHS_AVAILABLE:
        print(f"⚠️  Week {args.week} may not have cached data. Available months: {MONTHS_AVAILABLE}")

    output_path = Path(args.output) if args.output else PROJECT_ROOT / f"weekly_report_{args.week}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data...")
    all_calls = _load_all_data()
    if not all_calls:
        all_calls = _load_from_s3_month(year, month)
        if not all_calls:
            print("❌ No data found. Ensure cache exists or S3 is configured (.streamlit/secrets.toml)")
            sys.exit(1)
        print(f"   Loaded {len(all_calls)} calls for {year}-{month:02d} from S3")
    else:
        print(f"   Loaded {len(all_calls)} calls from cache")

    df = _prepare_dataframe(all_calls)
    if len(df) == 0:
        print("❌ No valid data after preparation")
        sys.exit(1)

    week_end = week_start + timedelta(days=6)
    week_mask = (df["Call Date"] >= pd.Timestamp(week_start)) & (df["Call Date"] <= pd.Timestamp(week_end))
    week_df = df[week_mask].copy()

    if len(week_df) == 0:
        month_calls = _load_from_s3_month(year, month)
        if month_calls:
            month_prep = _prepare_dataframe(month_calls)
            if len(month_prep) > 0:
                df = pd.concat([df, month_prep], ignore_index=True)
                week_mask = (df["Call Date"] >= pd.Timestamp(week_start)) & (df["Call Date"] <= pd.Timestamp(week_end))
                week_df = df[week_mask].copy()
                print(f"   Supplemented with {len(week_df)} calls for week of {args.week} from S3")

    all_df = df.copy()

    week_label = f"{week_start.strftime('%b %d')}–{week_end.strftime('%d, %Y')}"
    print(f"📊 Report week: {week_label} ({len(week_df)} calls)")
    print(f"   Historical: {len(all_df)} total calls for trend analysis")

    generate_report(week_df, all_df, week_start, output_path)


if __name__ == "__main__":
    main()
