# Jomashop QA Rubric Dashboard

A comprehensive Streamlit dashboard for visualizing and analyzing QA rubric performance across customer support calls. This dashboard enables dynamic filtering, KPI tracking, agent-specific views, and exportable summaries of QA call evaluations.

## What It Does

- Processes PDF QA reports from S3 bucket
- Extracts structured QA data including scores, rubric details, feedback, and coaching suggestions
- Enables flexible filtering and drilldown by:
  - Date range (with presets and custom selection)
  - Agent (with agent-specific views)
  - QA Score range
  - Labels (Pass/Fail/Neutral)
  - Failed rubric codes
  - Text search across call content
- Displays key QA metrics:
  - Average QA scores
  - Pass/Fail rates
  - Rubric code analysis
  - Agent performance comparisons
  - Trends over time
- Provides complete rubric reference with search and browse functionality
- Allows export to:
  - Excel spreadsheets (QA data, agent performance, rubric)
  - PDF chart reports
  - Individual call reports

## Features

### Core Functionality
- **S3 Integration**: Loads QA PDF reports directly from AWS S3 bucket
- **PDF Parsing**: Extracts structured data from QA report PDFs
- **Agent-Specific Views**: Agents see only their own calls with team average comparisons
- **Admin Views**: Full access to all data with comprehensive filtering
- **Rubric Reference**: Complete searchable rubric with 1,500+ items

### Analytics & Insights
- **Performance Alerts**: Highlights calls/agents below threshold
- **Rubric Code Analysis**: Identifies most common failure points
- **Agent Trends**: Individual agent performance over time
- **Team Comparisons**: Compare agent performance side-by-side
- **Pass/Fail Trends**: Track quality improvement over time
- **Coaching Insights**: Most common coaching suggestions
- **Time of Day Analysis**: Performance patterns by hour
- **Call Volume Analysis**: Volume trends by agent and over time

### Filtering & Search
- **Date Range**: All Time, This Week, Last 7 Days, Last 30 Days, Custom
- **Agent Filter**: Multiselect agents (admin) or auto-filtered (agent view)
- **QA Score Range**: Slider to filter by score range
- **Label Filter**: Filter by Pass/Fail/Neutral labels
- **Failed Rubric Codes**: Filter calls that failed specific rubric codes
- **Text Search**: Search across Reason, Summary, and Outcome fields

### Export Options
- **Excel Export**: Complete QA data with multiple sheets (QA Data, Agent Performance, Summary, Charts)
- **PDF Export**: All charts and visualizations as PDF
- **Individual Call Reports**: Export single call details as text file
- **Rubric Export**: Download complete rubric as Excel file

## Setup

### Prerequisites

```bash
pip install streamlit pandas matplotlib seaborn streamlit-authenticator boto3 pdfplumber openpyxl xlsxwriter
```

### Configuration

1. **S3 Credentials**: Add to `.streamlit/secrets.toml`:
```toml
[s3]
aws_access_key_id = "YOUR_ACCESS_KEY"
aws_secret_access_key = "YOUR_SECRET_KEY"
region_name = "us-west-2"
bucket_name = "your-bucket-name"
prefix = ""  # Optional: folder path in S3
```

2. **User Authentication**: Configure users in `.streamlit/secrets.toml`:
```toml
[credentials.usernames.username]
email = "user@example.com"
name = "User Name"
password = "$2y$12$..."  # Hashed password

[cookie]
name = "dashboard_session"
key = "your-secret-key"
expiry_days = 7
```

3. **Agent Mapping** (for agent-specific views): Add to `.streamlit/secrets.toml`:
```toml
[user_mapping]
[user_mapping.agent_username]
agent_id = "bpagent01"  # Must match Agent field in PDF data

# Admins (no agent_id = can see all data)
[user_mapping.admin_username]
# No agent_id needed
```

4. **Rubric File**: Place `Rubric_v33.json` in the dashboard directory

## Running the Dashboard

```bash
streamlit run streamlit_app_1_3_4.py
```

The dashboard will be available at `http://localhost:8501`

## Data Format

### PDF Reports
PDFs should be stored in the S3 bucket with the following naming convention:
- Format: `YYYYMMDD_HHMMSS_TYIO-bpagent##@nextiva.com-phone-IN.pdf`
- Example: `20250715_060053_TYIO-bpagent01@nextiva.com-+14178141239-IN.pdf`

### PDF Content Structure
PDFs should contain:
- **Call ID**: Unique identifier
- **QA Score**: Percentage score
- **Label**: Pass/Fail/Neutral
- **Reason, Outcome, Summary**: Call details
- **Strengths, Challenges**: Performance feedback
- **Coaching Suggestions**: List of coaching points
- **Rubric Details**: Code, status (Pass/Fail/N/A), and notes
- **BPO Agent**: Agent identifier (e.g., "BPO Agent 01")

## Dashboard Sections

### 1. Performance Alerts
- Highlights calls/agents below configurable threshold
- Shows summary of low-performing agents

### 2. Summary Metrics
- Total Calls
- Average QA Score
- Pass Rate
- Unique Agents

### 3. Agent Performance
- Leaderboard with key metrics
- Comparison tables (agent view shows vs team average)

### 4. QA Score Trends
- Daily average QA scores over time
- Pass/Fail rate trends
- Agent-specific trends (for agent view)

### 5. Rubric Code Analysis
- Statistics for each rubric code
- Top failing codes
- Category heatmaps
- Most common failure reasons

### 6. Agent-Specific Trends
- Individual agent performance over time
- Comparison to team average
- Pass rate trends

### 7. QA Score & Label Distribution
- Histogram of QA scores
- Bar chart of label distribution

### 8. Coaching Insights
- Most common coaching suggestions
- Frequency analysis

### 9. Complete Rubric Reference
- Searchable rubric with 1,500+ items
- Browse by section
- Download as Excel

### 10. Individual Call Details
- Full call information
- Rubric details table
- Export individual call reports

### 11. Call Volume Analysis
- Volume by agent
- Volume trends over time

### 12. Time of Day Analysis
- QA scores by hour
- Call volume by hour

### 13. Reason/Outcome Analysis
- Most common call reasons
- Most common outcomes

## Agent-Specific Views

When an agent logs in with a mapped `agent_id`:
- **Only sees their own calls**: Automatically filtered
- **Team comparisons**: Shows their metrics vs team averages
- **Trend analysis**: Their performance over time vs team
- **No access to other agents' data**: Privacy maintained

## Admin Views

Admins (users without `agent_id` mapping):
- **Full data access**: See all agents' calls
- **Agent filtering**: Can select any agent(s) to view
- **Complete analytics**: All dashboard features available
- **Agent leaderboards**: Compare all agents

## Export Options

### Excel Export
Includes multiple sheets:
- **QA Data**: All filtered call data
- **Agent Performance**: Performance metrics by agent
- **Summary**: Aggregate statistics
- **Charts**: Embedded visualizations

### PDF Export
- All charts and visualizations
- Professional formatting
- Multiple pages as needed

### Individual Call Report
- Text file with complete call details
- Includes all rubric information
- Suitable for coaching sessions

### Rubric Export
- Complete rubric as Excel file
- All 1,500+ items with full details
- Auto-formatted columns

## Troubleshooting

### S3 Connection Issues
- Verify AWS credentials in `secrets.toml`
- Check bucket name and region
- Ensure IAM permissions allow S3 read access

### PDF Parsing Errors
- Verify PDF format matches expected structure
- Check that "BPO Agent ##" appears in PDF content
- Ensure PDFs are not corrupted

### Agent View Not Working
- Verify `agent_id` in `secrets.toml` matches Agent field in PDFs
- Check that agent has calls in the date range
- Agent IDs are case-sensitive

### Rubric Not Loading
- Ensure `Rubric_v33.json` is in the dashboard directory
- Check file permissions
- Verify JSON format is valid

## Known Limitations

- Visualizations are static (no interactive zoom/hover)
- Large datasets may take time to load (use date filters)
- PDF parsing requires specific format structure
- Agent IDs must match exactly between mapping and PDF data

## File Structure

```
jomashop-dashboard/
├── streamlit_app_1_3_4.py    # Main dashboard application
├── pdf_parser.py              # PDF parsing logic
├── Rubric_v33.json            # Rubric reference data
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── secrets.toml          # Configuration and credentials
└── README.md                  # This file
```

## Support

For issues or questions, contact the development team.

---

Made with ❤️ by [Valence](https://www.getvalenceai.com) | Jomashop QA Dashboard © 2025
