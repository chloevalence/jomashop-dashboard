import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime, timedelta, date
from pandas import ExcelWriter
from matplotlib.backends.backend_pdf import PdfPages
import streamlit_authenticator as stauth
import firebase_admin
from firebase_admin import credentials, firestore
import json
from xlsxwriter.utility import xl_rowcol_to_cell
import time

# Build Firebase credentials from secrets
firebase_creds = {
    "type": st.secrets["firebase"]["type"],
    "project_id": st.secrets["firebase"]["project_id"],
    "private_key_id": st.secrets["firebase"]["private_key_id"],
    "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
    "client_email": st.secrets["firebase"]["client_email"],
    "client_id": st.secrets["firebase"]["client_id"],
    "auth_uri": st.secrets["firebase"]["auth_uri"],
    "token_uri": st.secrets["firebase"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
}

cred = credentials.Certificate(firebase_creds)

# Initialize Firebase app
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

@st.cache_data(ttl=3600)
def load_all_calls(page_size: int = 1000):
    """
    Paginate through 'calls' in Firestore 1,000 docs at a time,
    assemble into a single list, and cache the result for 1h.
    """
    client = firestore.client()            # uses your default credentials
    coll = client.collection("calls")
    all_calls = []
    last_doc = None

    while True:
        query = coll.order_by("call_date")
        if last_doc:
            query = query.start_after(last_doc)
        batch = list(query.limit(page_size).stream())
        if not batch:
            break
        # accumulate and advance cursor
        all_calls.extend(d.to_dict() for d in batch)
        last_doc = batch[-1]

    return all_calls

# Connect to Firestore
db = firestore.client()

st.set_page_config(page_title="Emotion Dashboard", layout="wide")

credentials = st.secrets["credentials"].to_dict()
cookie = st.secrets["cookie"]
auto_hash = st.secrets.get("auto_hash", False)

# --- Fetch Call Metadata ---
with st.spinner("⏳ Loading call data…"):
    t0 = time.time()
    call_data = load_all_calls(page_size=1000)
    elapsed = time.time() - t0

st.success(f"✅ Loaded {len(call_data)} calls in {elapsed:.2f}s")
meta_df = pd.DataFrame(call_data)

# --- Normalize raw fields into your canonical names: ---

meta_df.rename(columns={
    "company":        "Company",
    "agent":          "Agent",
    "call_date":      "Call Date",
    "date_raw":       "Date Raw",   # MMDDYYYY
    "time":           "Call Time",
    "call_id":        "Call ID",
    "low_confidences":"Low Confidences"
}, inplace=True)

# First, if all Call Dates are missing, try parsing Date Raw
if ("Call Date" not in meta_df.columns) or meta_df["Call Date"].isna().all():
    if "Date Raw" in meta_df.columns:
        meta_df["Call Date"] = pd.to_datetime(
            meta_df["Date Raw"],
            format="%m%d%Y",
            errors="coerce"
        )
    else:
        st.sidebar.error("❌ Neither Call Date nor Date Raw found—cannot parse any dates.")
        st.stop()

# Now, drop anything still unparseable in one go
before = len(meta_df)

# — identify any calls still missing a date —
missing = meta_df[meta_df["Call Date"].isna()]
meta_df.dropna(subset=["Call Date"], inplace=True)
dropped = before - len(meta_df)

#if dropped:
#    st.sidebar.warning(f"⚠️ Dropped {dropped} calls with no valid date.")

# Fix: Create Avg Happiness % directly from average_happiness_value
meta_df["Avg Happiness %"] = meta_df["average_happiness_value"]

# --- Recompute "Call Duration (s)" if missing ---
if "Call Duration (s)" not in meta_df.columns:
    if "speaking_time_per_speaker" in meta_df.columns:
        def compute_speaking_time(row):
            speaking_times = row["speaking_time_per_speaker"]
            if isinstance(speaking_times, dict):
                total = 0
                for t in speaking_times.values():
                    if isinstance(t, str) and ":" in t:
                        try:
                            minutes, seconds = map(int, t.split(":"))
                            total += minutes * 60 + seconds
                        except:
                            pass
                return total
            return None

        meta_df["Call Duration (s)"] = meta_df.apply(compute_speaking_time, axis=1)
    else:
        meta_df["Call Duration (s)"] = None


authenticator = stauth.Authenticate(
    credentials,
    cookie["name"],
    cookie["key"],
    cookie["expiry_days"],
    auto_hash=auto_hash,
)

# --- LOGIN GUARD ---
auth_status = st.session_state.get("authentication_status")

# If they’ve never submitted the form, show it
if auth_status is None:
    authenticator.login("main", "Login")
    st.stop()

# If they submitted bad creds, show error and stay on login
if auth_status is False:
    st.error("❌ Username or password is incorrect")
    st.stop()

st.sidebar.success(f"Welcome, {st.session_state.get('name')} 👋")

# --- Sidebar Section Toggles ---
st.sidebar.header("Display Options")
show_summary = st.sidebar.checkbox("📋 Show Summary Table", value=True)
show_leaderboard = st.sidebar.checkbox("🏆 Show Agent Leaderboard", value=True)
show_agent = st.sidebar.checkbox("📊 Show Happiness by Agent", value=True)
show_rolling = st.sidebar.checkbox("📈 Show Rolling Happiness", value=True)
show_emotion_by_company = st.sidebar.checkbox("🎯 Show Emotion Distribution by Company", value=True)
show_avg_by_time = st.sidebar.checkbox("📌 Show Happiness by Time of Day", value=True)
show_duration_vs_happiness = st.sidebar.checkbox("📍 Show Call Duration vs. Avg Happiness", value=True)
show_confidence = st.sidebar.checkbox("📈 Show Happiness vs. Confidence", value=True)
show_emotion_by_agent = st.sidebar.checkbox("🧊 Emotion Proportion by Agent", value=True)
show_duration_by_company = st.sidebar.checkbox("📶 Show Call Duration by Company", value=True)
show_volume = st.sidebar.checkbox("📞 Show Call Volume by Agent", value=True)

company_colors = {
    "Quantum": "#1f77b4",
    "StVincent": "#ff7f0e",
    "ABCMotors": "#2ca02c"
}

for emotion in ["happy", "angry", "sad", "neutral"]:
    if emotion not in meta_df.columns:
        meta_df[emotion] = 0

meta_df["Total Emotions"] = meta_df[["happy", "angry", "sad", "neutral"]].sum(axis=1)

# --- Prepare Metadata ---
meta_df["Call Duration (min)"] = meta_df["Call Duration (s)"] / 60
meta_df["Total Emotions"] = meta_df[["happy", "angry", "sad", "neutral"]].sum(axis=1)
meta_df["Avg Happiness %"] = (meta_df["happy"] / meta_df["Total Emotions"]) * 100

# --- Sidebar Filters ---
st.sidebar.header("📊 Filter Data")
min_date = meta_df["Call Date"].min()
max_date = meta_df["Call Date"].max()

companies = meta_df["Company"].dropna().unique().tolist()
available_agents = meta_df[meta_df["Company"].isin(companies)]["Agent"].dropna().unique().tolist()
dates = meta_df["Call Date"].dropna().sort_values().dt.date.unique().tolist()

if not dates:
    st.warning("⚠️ No calls with valid dates to display. Check your data or filters.")
    st.stop()

preset_option = st.sidebar.selectbox(
    "📆 Date Range Presets",
    options=["All Time", "This Week", "Last 7 Days", "Last 30 Days", "Custom"],
    index=0
)

selected_dates = (min(dates), max(dates))

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

    # Normalize into exactly (start_date, end_date)
    if isinstance(custom_input, tuple):
        if len(custom_input) == 2 and all(isinstance(d, date) for d in custom_input):
            selected_dates = custom_input
        else:
            st.warning("⚠️ Please select both a valid start and end date.")
            st.stop()
    elif isinstance(custom_input, date):
        selected_dates = (custom_input, custom_input)
    else:
        st.warning("⚠️ Please select a valid date range.")
        st.stop()

# Multiselect filters
selected_companies = st.sidebar.multiselect("Select Companies", companies, default=companies)
available_agents = meta_df[meta_df["Company"].isin(selected_companies)]["Agent"].dropna().unique().tolist()
selected_agents = st.sidebar.multiselect("Select Agents", available_agents, default=available_agents)

# Ensure both values in selected_dates are proper `date` objects
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        st.error("❌ Invalid date range selected.")
        st.stop()
else:
    st.error("❌ selected_dates is not a valid tuple of dates.")
    st.stop()

start, end = selected_dates

# Filter the DataFrame
filtered_df = meta_df[
    (meta_df["Company"].isin(selected_companies)) &
    (meta_df["Agent"].isin(selected_agents)) &
    (meta_df["Call Date"].dt.date >= start) &
    (meta_df["Call Date"].dt.date <= end)
].copy()

if filtered_df.empty:
    st.warning("⚠️ No data matches the current filter selection. Please adjust your filters.")
    st.stop()

# --- Create Figures ---
figures = []
summary_df = pd.DataFrame()

# --- Summary Table ---
if show_summary:
    st.subheader("📋 Summary Metrics")
    summary_data = {
        "Total Calls": [len(filtered_df)],
        "Unique Agents": [filtered_df["Agent"].nunique()],
        "Unique Companies": [filtered_df["Company"].nunique()],
        "Avg Call Duration (min)": [filtered_df["Call Duration (min)"].mean()],
        "Avg Happiness %": [filtered_df["Avg Happiness %"].mean()],
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

# --- Leaderboard ---
if show_leaderboard:
    st.subheader("🏆 Agent Leaderboard")
    agent_summary = (
        filtered_df
        .groupby("Agent")
        .agg(
            Total_Calls=("Call ID", "count"),
            Avg_Happiness_Percent=("Avg Happiness %", "mean"),
            Avg_Call_Duration_Min=("Call Duration (min)", "mean")
        )
        .reset_index()
        .sort_values(by="Avg_Happiness_Percent", ascending=False)
    )

    agent_summary = agent_summary.copy()

    st.dataframe(agent_summary, use_container_width=True)

    fig_leaderboard, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table_data = agent_summary.round(1)
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(agent_summary.columns))))
    figures.append((fig_leaderboard, "Agent Leaderboard"))
    plt.close(fig_leaderboard)

col4, col5 = st.columns(2)
col6, col7 = st.columns(2)
col8, col9 = st.columns(2)
col10, col11 = st.columns(2)

with col4:
    if show_agent:
        st.subheader("📊 Average Happiness by Agent")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        filtered_df.groupby("Agent")["Avg Happiness %"].mean().sort_values().plot(kind="barh", ax=ax1, color="teal")
        ax1.set_xlabel("Avg Happiness (%)")
        st.pyplot(fig1)
        figures.append((fig1, "Average Happiness by Agent"))

with col5:
    if show_rolling:
        st.subheader("📈 Rolling Happiness per Company (7-day)")
        filtered_df["Total Emotions"] = filtered_df[["happy", "angry", "sad", "neutral"]].sum(axis=1)
        filtered_df["Happy %"] = filtered_df["happy"] / filtered_df["Total Emotions"] * 100
        company_roll = filtered_df.groupby(["Call Date", "Company"])["Happy %"].mean().unstack()
        company_roll = company_roll.rolling(window=7, min_periods=1).mean()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        for company in company_roll.columns:
            ax2.plot(company_roll.index, company_roll[company], label=company, color=company_colors.get(company))
        ax2.set_ylabel("Rolling Happiness %")
        ax2.set_xlabel("Date")
        ax2.legend()
        st.pyplot(fig2)
        figures.append((fig2, "Rolling Happiness per Company"))

with col6:
    if show_emotion_by_company:
        st.subheader("🎯 Emotion Distribution by Company")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        emo_totals = filtered_df.groupby("Company")[["happy", "angry", "sad", "neutral"]].sum()
        emo_totals.plot(kind="bar", ax=ax3, color=["green", "red", "blue", "orange"])
        ax3.set_ylabel("Emotion Count")
        ax3.set_xlabel("Company")
        ax3.legend(title="Emotion")
        st.pyplot(fig3)
        figures.append((fig3, "Emotion Distribution by Company"))

with col7:
    if show_avg_by_time:
        st.subheader("📌 Average Happiness by Time of Day")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        filtered_df.groupby("Call Time")["Avg Happiness %"].mean().plot(kind="bar", ax=ax4, color="orange")
        ax4.set_xlabel("Call Time")
        ax4.set_ylabel("Avg Happiness (%)")
        st.pyplot(fig4)
        figures.append((fig4, "Happiness by Time of Day"))

with col8:
    if show_duration_vs_happiness:
        st.subheader("📍 Call Duration vs. Avg Happiness")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=filtered_df, x="Call Duration (min)", y="Avg Happiness %", hue="Company",
                        palette=company_colors, ax=ax5)
        ax5.set_xlabel("Call Duration (min)")
        ax5.set_ylabel("Avg Happiness (%)")
        st.pyplot(fig5)
        figures.append((fig5, "Duration vs Happiness"))

with col9:
    if show_confidence:
        st.subheader("📈 Happiness vs. Low Confidence")
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=filtered_df, x="Low Confidences", y="Avg Happiness %", hue="Company",
                        palette=company_colors, ax=ax6)
        ax6.set_xlabel("Low Confidence (%)")
        ax6.set_ylabel("Avg Happiness (%)")
        st.pyplot(fig6)
        figures.append((fig6, "Happiness vs Low Confidence"))

with col10:
    if show_emotion_by_agent:
        st.subheader("🧊 Emotion Proportion by Agent")
        emotion_pivot = filtered_df.groupby("Agent")[["happy", "angry", "sad", "neutral"]].sum()
        emotion_pivot_percent = emotion_pivot.div(emotion_pivot.sum(axis=1), axis=0) * 100
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        emotion_pivot_percent.plot(kind='bar', stacked=True, ax=ax7, color=["green", "red", "blue", "orange"])
        ax7.set_ylabel("Emotion %")
        ax7.set_xlabel("Agent")
        st.pyplot(fig7)
        figures.append((fig7, "Emotion Proportion by Agent"))

with col11:
    if show_duration_by_company:
        st.subheader("📶 Avg Call Duration by Company")
        fig8, ax8 = plt.subplots(figsize=(8, 4))
        avg_dur = filtered_df.groupby("Company")["Call Duration (min)"].mean()
        colors = [company_colors.get(c, "gray") for c in avg_dur.index]
        avg_dur.plot(kind="bar", ax=ax8, color=colors)
        ax8.set_ylabel("Avg Duration (min)")
        ax8.set_xlabel("Company")
        st.pyplot(fig8)
        figures.append((fig8, "Avg Call Duration by Company"))

if show_volume:
    st.subheader("📞 Call Volume per Agent per Day")
    call_volume = filtered_df.groupby(["Call Date", "Agent", "Company"]).size().reset_index(name="Call Count")
    facet = sns.FacetGrid(call_volume, col="Company", col_wrap=2, height=4, sharey=False)
    facet.map_dataframe(sns.lineplot, x="Call Date", y="Call Count", hue="Agent")
    facet.add_legend()
    facet.set_titles("{col_name}")

    # Rotate x-axis labels to avoid overlap
    for ax in facet.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    st.pyplot(facet.fig)
    figures.append((facet.fig, "Call Volume per Agent per Day"))

# --- Create Excel in memory ---
excel_buffer = io.BytesIO()
with ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    # === Prepare a copy for export, converting the dict column to JSON text ===
    export_df = filtered_df.copy()

    # Clean every cell: JSON-encode dicts/lists, strip tz from datetimes
    from datetime import datetime, timezone
    def _clean(val):
        # nested objects → JSON text
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        # timezone-aware datetimes → drop tzinfo
        if isinstance(val, datetime) and val.tzinfo is not None:
            # convert to UTC then drop tz
            val = val.astimezone(timezone.utc).replace(tzinfo=None)
            return val
        return val

    for col in export_df.columns:
        export_df[col] = export_df[col].map(_clean)

    # Now write export_df instead of filtered_df
    export_df.to_excel(writer, sheet_name="Call Metadata", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    # Charts sheet
    workbook = writer.book
    worksheet = workbook.add_worksheet("Charts")
    writer.sheets["Charts"] = worksheet

    def insert_plot(fig, cell):
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format='png', dpi=150, bbox_inches="tight")
        imgdata.seek(0)
        worksheet.insert_image(cell, "", {"image_data": imgdata})

    # Insert all figures in 2-column layout
    for idx, (fig, _) in enumerate(figures):
        row = (idx % 2) * 25  # 0 or 25 depending on odd/even
        col = (idx // 2) * 8  # every two figures, move right
        if col >= 16384:
            raise ValueError("Too many columns for Excel")  # Excel supports up to XFD (~16K)
        cell = xl_rowcol_to_cell(row + 1, col)
        insert_plot(fig, cell)

# --- Download Buttons ---
st.download_button(
    label="📥 Download Raw Data (Excel)",
    data=excel_buffer.getvalue(),
    file_name=f"call_report_{selected_dates[0]}_to_{selected_dates[1]}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Generate the PDF in memory and offer immediate download
pdf_buffer = io.BytesIO()
with PdfPages(pdf_buffer) as pdf:
    for idx, (fig, title) in enumerate(figures, start=1):
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])  # Shrink slightly more to leave room for footer

        # --- Add footer text manually ---
        footer_text = f"Generated by Valence Dashboard • © 2025"
        page_number = f"Page {idx}"

        fig.text(0.5, 0.02, footer_text, ha='center', va='center', fontsize=8, color='gray')
        fig.text(0.98, 0.02, page_number, ha='right', va='center', fontsize=8, color='gray')

        pdf.savefig(fig, bbox_inches='tight')
pdf_buffer.seek(0)

st.download_button(
    label="📄 Export All Graphs as PDF",
    data=pdf_buffer,
    file_name="Charts.pdf",
    mime="application/pdf"
)

st.markdown("---")
st.markdown("Built with ❤️ by [Valence](https://www.getvalenceai.com) | Pilot Dashboard © 2025")
