# Valence ACSI Dashboard

This Streamlit dashboard visualizes emotion AI insights across customer support calls processed by Valence. It enables dynamic filtering, KPI tracking, and exportable summaries of real-time voice emotion analytics.

## What It Does

- Processes `.json` call logs exported from Valence's ACSI analysis pipeline
- Aggregates emotional data by agent, company, and time of day
- Enables flexible, cross-week filtering and drilldown by:
  - Date range (with presets and custom selection)
  - Company
  - Agent
- Displays key metrics:
  - Average happiness percentage
  - Call durations
  - Emotion proportions
  - Call volume
  - Low-confidence detection rates
- Allows export to:
  - Excel summary sheets
  - PDF chart reports

## Folder Structure

Your input should be a `.zip` file structured as:
Uploaded ZIP ┣ 04152025/ ┃ ┣ call_001.json ┃ ┗ call_002.json ┣ 04162025/ ┃ ┣ call_003.json ┃ ┗ call_004.json


Each subfolder name (e.g., `04152025`) should represent the date of that day's batch, formatted as `MMDDYYYY`.

## JSON Structure

Each `.json` file must contain:
- `"metadata"` with fields like `agent`, `company`, `time`, `date`, and `low_confidences`
- `"emotion_graph"` array with x/y values representing time vs. happiness score
- `"emotion_counts"` dictionary (e.g., happy, sad, angry, neutral)
- `"speaking_time_per_speaker"` (optional, for calculating call duration)
- `"average_happiness_value"`

## Features

- **Robust Encoding Support**: Automatically handles UTF-8 and fallback to Latin-1 for legacy JSONs.
- **Cross-Week Filtering**: Unlike older versions, filters now span all dates and are not restricted to folder-level grouping.
- **Summary Metrics**: Aggregates and displays call-level KPIs in real-time.
- **Agent leaderboards**: Analyzes call volume, happiness scores, and handle time for each agent.
- **Date Presets**: Users can easily view stats for this week, last 7 days, or any custom range.
- **Sidebar Filters**: Fast filtering by company and agent.

## Running the Dashboard

Make sure Streamlit and required dependencies are installed:

```markdown
pip install streamlit pandas plotly
streamlit run streamlit_app_1_3_4.py
```

## Authentication

This dashboard supports user-level authentication using streamlit-authenticator. User credentials are managed via a secrets.toml file and should not be pushed to GitHub.

## Export Options

Filtered results can be exported as:
- Excel spreadsheets (per agent, per company)
- PDF summary reports with all charts
- All-in-one ZIP containing both

## Known Limitations

- Visualizations are static (no scroll/zoom/hover — upcoming feature).
- Date parsing requires "Call Date" in MMDDYYYY format.
- Emotion graph assumes presence of "x" and "y" in each emotion_graph entry.

## Coming Soon

- Generative AI insights
- Persistent data storage
- Interactive charts (zoom, hover, scroll)
- Custom PDF/Excel export for filtered views
- Saved filter presets

Made with ❤️ by the Valence team.
