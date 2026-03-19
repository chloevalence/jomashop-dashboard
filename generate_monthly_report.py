#!/usr/bin/env python3
"""
Monthly QA Report Generator

Generates an exportable PDF report for a given month with Valence branding, including:
- Agent leaderboard
- Pass rates and QA scores over time (all historical data for progress comparison)
- Call Reason / Outcome / Product distributions for the month
- Coaching suggestions per agent (based on most common rubric failures)
- Failing calls per agent (Reason, Outcome, Product, QA Score)

Built to run locally (not in Streamlit). Uses cache or S3 month caches for data.
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

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak,
    Image as RLImage, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_FILE = LOG_DIR / "cached_calls_data.json"
PASS_THRESHOLD = 70

# Valence brand colors
V_PURPLE = colors.HexColor("#4900A7")
V_BLUE = colors.HexColor("#0974FF")
V_TEAL = colors.HexColor("#38BDF8")
V_LTPURP = colors.HexColor("#F5F0FF")
V_DARK = colors.HexColor("#0A0A0A")
V_GRAY = colors.HexColor("#64748B")
V_LGRAY = colors.HexColor("#F1F5F9")
V_BORDER = colors.HexColor("#E2E8F0")
V_WHITE = colors.white
P_PURPLE = "#4900A7"
P_BLUE = "#0974FF"
P_TEAL = "#38BDF8"
P_GRAY = "#64748B"
P_LGRAY = "#F1F5F9"
P_GREEN = "#10B981"
P_AMBER = "#F59E0B"

# Report layout
PAGE_WIDTH, PAGE_HEIGHT = 8.5, 11
PAGE_W, PAGE_H = letter
MARGIN = 0.5 * inch
TABLE_FONTSIZE = 12
LOGO_ASPECT = 2000 / 649

# Logo paths
ASSETS = PROJECT_ROOT / "assets"
CURSOR_ASSETS = Path.home() / ".cursor" / "projects" / "Users-Chloe-Downloads-jomashop-dashboard" / "assets"
LOGO_COLORED_SRC = next(
    (p for p in [
        ASSETS / "Valence_-_Text_to_right_-_Colored-23dcb8b0-9e37-4af5-ad4e-672f2cb98533.png",
        CURSOR_ASSETS / "Valence_-_Text_to_right_-_Colored-23dcb8b0-9e37-4af5-ad4e-672f2cb98533.png",
        ASSETS / "Valence_-_Text_to_right_-_Colored.png",
    ] if p.exists()),
    ASSETS / "Valence_-_Text_to_right_-_Colored.png",
)
LOGO_WHITE_SRC = next(
    (p for p in [
        ASSETS / "Valence_-_Text_to_right_-_White-4a5c4619-1570-4e2f-be11-04224c3cfc8e.png",
        CURSOR_ASSETS / "Valence_-_Text_to_right_-_White-4a5c4619-1570-4e2f-be11-04224c3cfc8e.png",
        ASSETS / "Valence_-_Text_to_right_-_White.png",
    ] if p.exists()),
    ASSETS / "Valence_-_Text_to_right_-_White.png",
)
LOGO_WHITE = str(PROJECT_ROOT / "logo_white_transparent.png")
LOGO_COLORED = str(PROJECT_ROOT / "logo_colored_transparent.png")

# Month options (must match streamlit month_options for S3 cache keys)
MONTHS_AVAILABLE = [
    (2026, 3), (2026, 2), (2026, 1),
    (2025, 12), (2025, 11), (2025, 10), (2025, 9), (2025, 8), (2025, 7),
]


def _make_logos_transparent():
    """Convert black-background logos to transparent PNGs (Valence template)."""
    try:
        from PIL import Image
        for src, dest, scale in [
            (LOGO_WHITE_SRC, LOGO_WHITE, 1),
            (LOGO_COLORED_SRC, LOGO_COLORED, 5),
        ]:
            if src.exists():
                img = Image.open(src).convert("RGBA")
                arr = np.array(img, dtype=np.uint8)
                alpha = np.clip(np.max(arr[:, :, :3], axis=2).astype(int) * scale, 0, 255).astype(np.uint8)
                arr[:, :, 3] = alpha
                Image.fromarray(arr).save(dest)
    except Exception:
        pass


def _make_on_page(subtitle: str):
    """Valence header bar (purple) + white logo; gray footer with page number."""
    def on_page(cv, doc):
        cv.saveState()
        cv.setFillColor(V_PURPLE)
        cv.rect(0, PAGE_H - 46, PAGE_W, 46, fill=1, stroke=0)
        if os.path.exists(LOGO_WHITE):
            logo_h, logo_w = 26, 26 * LOGO_ASPECT
            logo_y = PAGE_H - 46 + (46 - logo_h) / 2
            cv.drawImage(LOGO_WHITE, MARGIN, logo_y, width=logo_w, height=logo_h, mask="auto")
        cv.setFont("Helvetica", 8.5)
        cv.setFillColorRGB(0.85, 0.75, 1.0)
        cv.drawRightString(PAGE_W - MARGIN, PAGE_H - 26, subtitle)
        cv.setFillColor(V_LGRAY)
        cv.rect(0, 0, PAGE_W, 26, fill=1, stroke=0)
        cv.setFillColor(V_GRAY)
        cv.setFont("Helvetica", 7.5)
        cv.drawString(MARGIN, 8, f"Monthly QA Report  ·  {subtitle}  ·  Confidential  ·  Valence AI")
        cv.drawRightString(PAGE_W - MARGIN, 8, f"Page {doc.page}")
        cv.restoreState()
    return on_page


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply Valence chart styling."""
    ax.set_facecolor(P_LGRAY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=P_GRAY, labelsize=7.5)
    ax.grid(axis="x", color="white", linewidth=0.8, zorder=0)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", color="#0A0A0A", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5, color=P_GRAY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5, color=P_GRAY)


def _fig_buf(fig, dpi=150):
    """Save matplotlib figure to BytesIO."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


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


def _append_agent_pages(
    story: List,
    rubric_labels: Dict[str, str],
    agent_failures: Dict,
    failing: pd.DataFrame,
    all_agents: List[str],
    month_name: str,
    id_col: str,
    dur_col: Optional[str],
    TW: float,
):
    """Append agent coaching + failing calls pages to story (Valence styling, text wrap)."""
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
        wordWrap="CJK",
    )
    header_style = ParagraphStyle(
        "Header",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        fontName="Helvetica-Bold",
        wordWrap="CJK",
    )
    col_widths = [TW * 0.18, TW * 0.055, TW * 0.065, TW * 0.28, TW * 0.28, TW * 0.085]

    for agent_idx, agent in enumerate(all_agents):
        fails = agent_failures.get(agent, {})
        top_coaching = sorted(fails.items(), key=lambda x: -x[1])[:5]
        agent_fails_df = failing[failing["Agent"] == agent]

        if len(agent_fails_df) == 0 and not top_coaching:
            continue

        is_last_agent = agent_idx == len(all_agents) - 1

        title = f"{agent} — Coaching & Failing Calls ({month_name})"
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
            score_ints = []
            for _, row in agent_fails_df.iterrows():
                cid = str(row.get(id_col, row.get("call_id", "—")))
                dur = row.get(dur_col)
                handle = f"{dur:.1f}" if isinstance(dur, (int, float)) and not pd.isna(dur) else "—"
                score = row.get("QA Score", "—")
                score_int = int(score) if isinstance(score, (int, float)) and not pd.isna(score) else 0
                score_ints.append(score_int)
                if isinstance(score, (int, float)) and not pd.isna(score):
                    score_str = f"{score:.0f}%"
                else:
                    score_str = str(score)
                reason = str(row.get("Reason", "") or "")
                outcome = str(row.get("Outcome", "") or "")
                product = str(row.get("Product", "") or "—")
                data.append([
                    _para(cid, cell_style),
                    _para(handle, cell_style),
                    _para(score_str, cell_style),
                    _para(reason, cell_style),
                    _para(outcome, cell_style),
                    _para(product, cell_style),
                ])
            tbl = Table(data, colWidths=col_widths, repeatRows=1)
            tbl_ts = [
                ("BACKGROUND", (0, 0), (-1, 0), V_PURPLE),
                ("TEXTCOLOR", (0, 0), (-1, 0), V_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("GRID", (0, 0), (-1, -1), 0.3, V_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [V_WHITE, V_LGRAY]),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
            ]
            for i, s in enumerate(score_ints):
                if s < 50 and s > 0:
                    tbl_ts += [("BACKGROUND", (2, i + 1), (2, i + 1), colors.HexColor("#FEE2E2")), ("TEXTCOLOR", (2, i + 1), (2, i + 1), colors.HexColor("#991B1B"))]
                elif s < 70 and s > 0:
                    tbl_ts += [("BACKGROUND", (2, i + 1), (2, i + 1), colors.HexColor("#FEF3C7")), ("TEXTCOLOR", (2, i + 1), (2, i + 1), colors.HexColor("#92400E"))]
                elif s > 80:
                    tbl_ts += [("BACKGROUND", (2, i + 1), (2, i + 1), colors.HexColor("#D1FAE5")), ("TEXTCOLOR", (2, i + 1), (2, i + 1), colors.HexColor("#065F46"))]
            tbl.setStyle(TableStyle(tbl_ts))
            story.append(tbl)

        if not is_last_agent:
            story.append(PageBreak())


def generate_report(month_df: pd.DataFrame, all_df: pd.DataFrame, year: int, month: int, output_path: Path):
    """Generate the monthly report PDF with Valence branding."""
    month_name = datetime(year, month, 1).strftime("%B %Y")
    subtitle = month_name

    _make_logos_transparent()

    rubric_labels = _load_rubric_labels() if (PROJECT_ROOT / "Rubric_v33.json").exists() else {}
    agent_failures = defaultdict(lambda: defaultdict(int))
    if "Rubric Details" in month_df.columns:
        for _, row in month_df.iterrows():
            agent = row.get("Agent", "Unknown")
            rd = row.get("Rubric Details", {})
            if isinstance(rd, dict):
                for code, det in rd.items():
                    if isinstance(det, dict) and det.get("status", "").lower() == "fail":
                        agent_failures[agent][code] += 1
    failing = month_df[month_df["QA Score"] < PASS_THRESHOLD] if "QA Score" in month_df.columns else pd.DataFrame()
    if len(failing) > 0:
        failing = failing.copy()
        failing["Product"] = failing.apply(extract_products, axis=1)
    all_agents = sorted(
        set(agent_failures.keys()) | (set(failing["Agent"].unique()) if len(failing) > 0 else set()),
        key=_agent_sort_key,
    )

    id_col = "Call ID" if "Call ID" in month_df.columns else "call_id"
    dur_col = "Call Duration (min)" if "Call Duration (min)" in failing.columns else None
    TW = PAGE_W - 2 * MARGIN

    sTitleBig = ParagraphStyle("sTitleBig", fontName="Helvetica-Bold", fontSize=34, textColor=V_DARK, leading=40, spaceAfter=4)
    sSubtitle = ParagraphStyle("sSubtitle", fontName="Helvetica", fontSize=13, textColor=V_GRAY, leading=18)
    sH1 = ParagraphStyle("sH1", fontName="Helvetica-Bold", fontSize=14, textColor=V_DARK, leading=18, spaceBefore=10, spaceAfter=5)
    sH2 = ParagraphStyle("sH2", fontName="Helvetica-Bold", fontSize=10.5, textColor=V_DARK, leading=14, spaceBefore=8, spaceAfter=3)
    sCenter = ParagraphStyle("sCenter", fontName="Helvetica", fontSize=9, textColor=V_DARK, alignment=TA_CENTER, leading=12)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.72 * inch, bottomMargin=0.5 * inch,
    )
    story = []

    # ——— COVER PAGE ———
    story.append(Spacer(1, 1.6 * inch))
    if os.path.exists(LOGO_COLORED):
        cover_logo_w, cover_logo_h = 280, 280 / LOGO_ASPECT
        story.append(RLImage(LOGO_COLORED, width=cover_logo_w, height=cover_logo_h))
    story.append(Spacer(1, 0.22 * inch))
    story.append(Paragraph("Monthly QA Report", sTitleBig))
    story.append(Paragraph(month_name, sSubtitle))
    story.append(Spacer(1, 0.12 * inch))
    story.append(HRFlowable(width=TW, thickness=2.5, color=V_PURPLE, spaceAfter=16))

    total_calls = int(month_df[id_col].nunique()) if len(month_df) > 0 and id_col in month_df.columns else 0
    avg_qa = month_df["QA Score"].mean() if len(month_df) > 0 and "QA Score" in month_df.columns else 0
    pass_rate = (month_df["QA Score"] >= PASS_THRESHOLD).mean() * 100 if len(month_df) > 0 and "QA Score" in month_df.columns else 0
    n_agents = month_df["Agent"].nunique() if len(month_df) > 0 and "Agent" in month_df.columns else 0
    kpis = [
        (f"{total_calls:,}", "Total Calls"),
        (f"{avg_qa:.1f}", "Avg QA Score"),
        (f"{pass_rate:.1f}%", "Pass Rate"),
        (str(n_agents), "Agents"),
    ]
    kpi_cells = [[
        Paragraph(f'<font size="22" color="#4900A7"><b>{k}</b></font><br/><font size="8" color="#64748B">{lbl}</font>', sCenter)
        for k, lbl in kpis
    ]]
    kpi_tbl = Table(kpi_cells, colWidths=[TW / 4] * 4)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), V_LTPURP),
        ("BOX", (0, 0), (-1, -1), 0.5, V_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, V_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(kpi_tbl)
    story.append(PageBreak())

    # ——— AGENT LEADERBOARD ———
    story.append(Paragraph("Agent Leaderboard", sH1))
    story.append(HRFlowable(width=TW, thickness=1.5, color=V_PURPLE, spaceAfter=7))
    if len(month_df) > 0 and "Agent" in month_df.columns:
        call_col = "Call ID" if "Call ID" in month_df.columns else "call_id"
        id_c = call_col if call_col in month_df.columns else None
        if id_c:
            agg = month_df.groupby("Agent").agg(
                Total_Calls=(id_c, "nunique"),
                Avg_QA_Score=("QA Score", "mean"),
                Pass_Rate=("QA Score", lambda s: (s >= PASS_THRESHOLD).mean() * 100),
            ).reset_index()
        else:
            agg = month_df.groupby("Agent").agg(
                Total_Calls=("QA Score", "count"),
                Avg_QA_Score=("QA Score", "mean"),
                Pass_Rate=("QA Score", lambda s: (s >= PASS_THRESHOLD).mean() * 100),
            ).reset_index()
        agg = agg.sort_values("Avg_QA_Score", ascending=False)
        names, scores = list(agg["Agent"]), list(agg["Avg_QA_Score"])
        fig, ax = plt.subplots(figsize=(7.4, 3.4), facecolor=P_LGRAY)
        pal = plt.cm.RdYlGn
        norm = [(s - 48) / 38 for s in scores]
        ax.barh(names[::-1], scores[::-1], color=[pal(n) for n in norm[::-1]], height=0.62, zorder=3)
        for bar, s in zip(ax.patches, scores[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{s:.1f}", va="center", ha="left", fontsize=7.5, color="#0A0A0A", fontweight="bold")
        ax.axvline(70, color="#888", lw=1, ls="--", zorder=4, alpha=0.7)
        ax.set_xlim(40, 96)
        _style_ax(ax, title=f"Agent Leaderboard — Avg QA Score ({month_name})", xlabel="Avg QA Score")
        fig.tight_layout(pad=0.4)
        story.append(RLImage(_fig_buf(fig), width=TW, height=TW * 0.43))
        story.append(Spacer(1, 8))
        hdr = ["Agent", "Total Calls", "Avg QA Score", "Pass Rate %"]
        rows = [hdr] + [[a, str(int(b)), f"{c:.1f}", f"{d:.1f}%"] for a, b, c, d in agg.values]
        cw = [TW * 0.32, TW * 0.22, TW * 0.24, TW * 0.22]
        tbl = Table(rows, colWidths=cw, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), V_PURPLE),
            ("TEXTCOLOR", (0, 0), (-1, 0), V_WHITE),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [V_WHITE, V_LGRAY]),
            ("GRID", (0, 0), (-1, -1), 0.3, V_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)
    story.append(PageBreak())

    # ——— PERFORMANCE TRENDS ———
    story.append(Paragraph("Performance Trends", sH1))
    story.append(HRFlowable(width=TW, thickness=1.5, color=V_PURPLE, spaceAfter=7))
    if len(all_df) > 0 and "Call Date" in all_df.columns:
        adf = all_df.copy()
        adf["Month"] = adf["Call Date"].dt.to_period("M").astype(str)
        monthly = adf.groupby("Month").agg(
            Pass_Rate=("QA Score", lambda s: (s >= PASS_THRESHOLD).mean() * 100),
            Avg_QA=("QA Score", "mean"),
        ).reset_index().sort_values("Month")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.2), facecolor=P_LGRAY)
        x = range(len(monthly))
        ax1.plot(x, monthly["Pass_Rate"], color=P_BLUE, lw=2, marker="o", ms=5, zorder=3)
        ax1.fill_between(x, monthly["Pass_Rate"], alpha=0.12, color=P_BLUE)
        _style_ax(ax1, title="Pass Rate % — Monthly Trend", ylabel="Pass Rate %")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(monthly["Month"].tolist(), fontsize=7)
        ax1.grid(axis="y", color="white", lw=0.8)
        ax2.plot(x, monthly["Avg_QA"], color=P_GREEN, lw=2, marker="o", ms=5, zorder=3)
        ax2.fill_between(x, monthly["Avg_QA"], alpha=0.12, color=P_GREEN)
        ax2.axhline(70, color=P_AMBER, lw=1.2, ls="--", alpha=0.8)
        _style_ax(ax2, title="Avg QA Score — Monthly Trend", ylabel="Avg QA Score")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(monthly["Month"].tolist(), fontsize=7)
        ax2.grid(axis="y", color="white", lw=0.8)
        fig.tight_layout(pad=0.5)
        story.append(RLImage(_fig_buf(fig), width=TW, height=TW * 0.68))
    story.append(PageBreak())

    # ——— CALL ANALYTICS ———
    story.append(Paragraph("Call Analytics", sH1))
    story.append(HRFlowable(width=TW, thickness=1.5, color=V_PURPLE, spaceAfter=7))
    if len(month_df) > 0:
        reasons = month_df["Reason"].dropna().astype(str).str.strip().str.lower()
        reasons = reasons[reasons != ""]
        outcomes = month_df["Outcome"].dropna().astype(str).str.strip().str.lower()
        outcomes = outcomes[outcomes != ""]
        all_prods = []
        for _, row in month_df.iterrows():
            all_prods.extend(extract_products_list(row))
        rc = list(reasons.value_counts().head(8).items()) if len(reasons) > 0 else []
        oc = list(outcomes.value_counts().head(8).items()) if len(outcomes) > 0 else []
        pc = Counter(all_prods).most_common(8) if all_prods else []
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.2), facecolor=P_LGRAY)
        for ax, data, color, label in [
            (axes[0], rc, P_PURPLE, "Call Reasons"),
            (axes[1], oc, P_BLUE, "Call Outcomes"),
            (axes[2], [(p, c) for p, c in pc], P_TEAL, "Products"),
        ]:
            if data:
                labs = [d[0][:30] + "…" if len(str(d[0])) > 30 else str(d[0]) for d in data]
                vals = [d[1] for d in data]
                ax.barh(labs[::-1], vals[::-1], color=color, height=0.65, zorder=3)
            _style_ax(ax, title=label, xlabel="Count")
        fig.tight_layout(pad=0.7)
        story.append(RLImage(_fig_buf(fig), width=TW, height=TW * 0.52))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Performance by Time of Day", sH2))
    if len(month_df) > 0 and "Hour" in month_df.columns and "QA Score" in month_df.columns:
        by_hour = month_df[month_df["Hour"].notna()]
        if len(by_hour) > 0:
            hourly = by_hour.groupby("Hour").agg(
                Avg_QA=("QA Score", "mean"),
                Pass_Rate=("QA Score", lambda s: (s >= PASS_THRESHOLD).mean() * 100),
                Count=("QA Score", "count"),
            )
            hourly = hourly[hourly["Count"] > 0]
            if len(hourly) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0), facecolor=P_LGRAY)
                x = np.arange(len(hourly))
                ax1.bar(x, hourly["Avg_QA"], color=[P_PURPLE if v >= 70 else "#C4B5FD" for v in hourly["Avg_QA"]], width=0.65, zorder=3)
                ax1.axhline(70, color="#888", lw=1, ls="--", zorder=4)
                ax1.set_xticks(x)
                ax1.set_xticklabels([f"{h}:00" for h in hourly.index], fontsize=6.5)
                _style_ax(ax1, title="QA Score by Hour", xlabel="Hour", ylabel="Avg QA Score")
                ax1.set_ylim(0, 95)
                ax2.bar(x, hourly["Pass_Rate"], color=[P_BLUE if v >= 60 else "#FCA5A5" for v in hourly["Pass_Rate"]], width=0.65, zorder=3)
                ax2.axhline(60, color="#888", lw=1, ls="--", zorder=4)
                ax2.set_xticks(x)
                ax2.set_xticklabels([f"{h}:00" for h in hourly.index], fontsize=6.5)
                _style_ax(ax2, title="Pass Rate by Hour", xlabel="Hour", ylabel="Pass Rate %")
                ax2.set_ylim(0, 95)
                fig.tight_layout(pad=0.5)
                story.append(RLImage(_fig_buf(fig), width=TW, height=TW * 0.38))
    story.append(PageBreak())

    # ——— AGENT PAGES ———
    _append_agent_pages(story, rubric_labels, agent_failures, failing, all_agents, month_name, id_col, dur_col, TW)

    on_page = _make_on_page(subtitle)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"✅ Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate monthly QA report (PDF)")
    parser.add_argument("--month", type=str, required=True, help="Month in YYYY-MM format (e.g. 2026-03)")
    parser.add_argument("--output", type=str, default=None, help="Output PDF path (default: monthly_report_YYYY-MM.pdf)")
    parser.add_argument("--cache", type=str, default=str(CACHE_FILE), help="Path to cache file (for disk cache)")
    args = parser.parse_args()

    try:
        year, month = map(int, args.month.split("-"))
    except ValueError:
        print(f"❌ Invalid --month format. Use YYYY-MM (e.g. 2026-03)")
        sys.exit(1)

    if (year, month) not in MONTHS_AVAILABLE:
        print(f"⚠️  Month {year}-{month:02d} may not have cached data. Available: {MONTHS_AVAILABLE}")

    output_path = Path(args.output) if args.output else PROJECT_ROOT / f"monthly_report_{year}-{month:02d}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data...")
    all_calls = _load_all_data()
    if not all_calls:
        # Try loading just the requested month from S3
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

    month_start = datetime(year, month, 1)
    if month == 12:
        month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = datetime(year, month + 1, 1) - timedelta(days=1)
    month_mask = (df["Call Date"] >= pd.Timestamp(month_start)) & (df["Call Date"] <= pd.Timestamp(month_end))
    month_df = df[month_mask].copy()

    # If report month has no calls in main cache, try loading that month from S3
    if len(month_df) == 0:
        month_calls = _load_from_s3_month(year, month)
        if month_calls:
            month_prep = _prepare_dataframe(month_calls)
            if len(month_prep) > 0:
                df = pd.concat([df, month_prep], ignore_index=True)
                month_mask = (df["Call Date"] >= pd.Timestamp(month_start)) & (df["Call Date"] <= pd.Timestamp(month_end))
                month_df = df[month_mask].copy()
                print(f"   Supplemented with {len(month_df)} calls for {year}-{month:02d} from S3 monthly cache")

    all_df = df.copy()

    print(f"📊 Report month: {month_start.strftime('%B %Y')} ({len(month_df)} calls)")
    print(f"   Historical: {len(all_df)} total calls for trend analysis")

    generate_report(month_df, all_df, year, month, output_path)


if __name__ == "__main__":
    main()
