# Jomashop QA Dashboard - User Guide

Welcome to the Jomashop QA Rubric Dashboard! This comprehensive guide will help you navigate and make the most of all the powerful features available in the platform.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Filtering & Search](#filtering--search)
4. [Performance Metrics](#performance-metrics)
5. [Analytics & Insights](#analytics--insights)
6. [Agent-Specific Features](#agent-specific-features)
7. [Admin Features](#admin-features)
8. [Export Options](#export-options)
9. [Rubric Reference](#rubric-reference)
10. [Tips & Best Practices](#tips--best-practices)

---

## Getting Started

### Logging In

1. **Access the Dashboard**: Navigate to your dashboard URL
2. **Enter Credentials**: Use your username and password
3. **Welcome Screen**: Once logged in, you'll see a personalized welcome message

**Note**: If you're an agent, you'll automatically see only your own calls. If you're an admin, you'll have access to all data.

### First-Time Setup

- The dashboard automatically loads data from your S3 bucket
- Initial load may take a few minutes depending on data volume
- Data is cached for faster subsequent access

---

## Dashboard Overview

### Main Sections

The dashboard is organized into several key sections:

1. **Sidebar Filters** - Control what data you see
2. **Summary Metrics** - Key performance indicators at a glance
3. **Performance Alerts** - Important notifications about low scores
4. **Analytics Sections** - Detailed insights and visualizations
5. **Call Details** - Individual call information
6. **Export Options** - Download data and reports

### Navigation

- **Sidebar**: Use the left sidebar to filter and control data
- **Expandable Sections**: Click on section headers to expand/collapse
- **Tabs**: Some sections have tabs for different views
- **Scroll**: Scroll down to see all available sections

---

## Filtering & Search

### Date Range Filter

**Location**: Sidebar → 📆 Date Range

**Options**:
- **All Time**: View all available data
- **This Week**: Current week's data
- **Last 7 Days**: Past week
- **Last 30 Days**: Past month
- **Custom**: Select your own date range

**How to Use**:
1. Select a preset option or choose "Custom"
2. If "Custom", use the date pickers to select start and end dates
3. Data automatically updates based on your selection

### Agent Filter (Admin Only)

**Location**: Sidebar → Select Agents

**Features**:
- **Multi-select**: Choose one or multiple agents
- **Select All**: Option to quickly select all agents
- **Agent View**: Agents automatically see only their own data

**How to Use**:
1. Click the dropdown
2. Select agents you want to analyze
3. Leave empty to see all agents

### QA Score Range

**Location**: Sidebar → 📊 QA Score Range

**How to Use**:
1. Drag the slider to set minimum and maximum scores
2. View only calls within your specified range
3. Useful for finding high or low performers

### Label Filter

**Location**: Sidebar → 🏷️ Select Labels

**Options**:
- **Pass**: Calls that passed QA
- **Fail**: Calls that failed QA
- **Neutral**: Calls with neutral status

**How to Use**:
1. Select one or more labels
2. Filter to focus on specific call outcomes
3. Combine with other filters for precise analysis

### Rubric Code Filter

**Location**: Sidebar → Select Rubric Codes

**Features**:
- Filter by specific rubric codes
- Choose filter type:
  - **Any Status**: Shows calls with this code regardless of pass/fail
  - **Failed Only**: Shows only calls that failed this code
  - **Passed Only**: Shows only calls that passed this code

**How to Use**:
1. Select the filter type
2. Choose one or more rubric codes from the dropdown
3. View calls that match your criteria

### Text Search

**Location**: Sidebar → 🔍 Search

**Searches Across**:
- Call Reason
- Summary
- Outcome
- Strengths
- Challenges
- Coaching Suggestions

**How to Use**:
1. Type your search term in the search box
2. Search is case-insensitive
3. Results update as you type
4. Combine with other filters for precise results

**Tips**:
- Use specific keywords for better results
- Search for product names, issue types, or agent names
- Combine with date filters to find recent issues

---

## Performance Metrics

### Summary Metrics

**Location**: Top of dashboard

**Key Metrics Displayed**:
- **Total Calls**: Number of calls in your filtered view
- **Average QA Score**: Mean QA score percentage
- **Pass Rate**: Percentage of calls that passed
- **Unique Agents**: Number of different agents in the data

**How to Use**:
- These metrics update automatically based on your filters
- Compare metrics across different date ranges
- Use to quickly assess overall performance

### Performance Alerts

**Location**: Below Summary Metrics

**Features**:
- **Alert Threshold**: Set your own threshold (sidebar slider)
- **Low Score Alerts**: Highlights calls below threshold
- **At-Risk Agents**: Identifies agents trending toward threshold
- **Risk Factors**: Explains why agents are at risk

**How to Use**:
1. Set your alert threshold in the sidebar
2. View alerts for calls below threshold
3. Review at-risk agents and their risk factors
4. Take proactive action on declining trends

**Risk Factors Include**:
- Declining trend (negative slope)
- High volatility (inconsistent scores)
- Proximity to threshold (close to failing)

---

## Analytics & Insights

### Agent Performance Leaderboard

**Location**: Agent Performance section

**Features**:
- **Ranking**: Agents sorted by performance
- **Key Metrics**: Total calls, average score, pass rate, call duration
- **Sortable**: Click column headers to sort
- **Comparison**: Agent view shows "My Performance vs Team Average"

**Metrics Shown**:
- Total Calls
- Average QA Score
- Pass Rate
- Average Call Duration (AHT)
- Total Pass/Fail Counts

**How to Use**:
- Identify top and bottom performers
- Compare individual metrics
- Track improvement over time
- Use for coaching and recognition

### QA Score Trends

**Location**: QA Score Trends section

**Visualizations**:
- **Daily Average QA Scores**: Line chart showing score trends over time
- **Pass/Fail Rate Trends**: Stacked area chart showing pass/fail percentages
- **Agent-Specific Trends**: Individual agent performance (agent view)

**How to Use**:
- Identify trends and patterns
- Spot improvement or decline periods
- Compare your performance to team average (agent view)
- Plan training based on trend analysis

### Rubric Code Analysis

**Location**: Rubric Code Analysis section

**Features**:
- **Top 10 Failed Rubric Codes**: Most common failure points
- **Fail Rate Distribution**: Visual breakdown of failure rates
- **Fail Rate by Rubric Category**: Category-level analysis
- **Statistics Table**: Detailed metrics for each rubric code

**Metrics for Each Code**:
- Total occurrences
- Pass count and rate
- Fail count and rate
- N/A count
- Overall pass rate

**How to Use**:
- Identify common failure patterns
- Focus training on high-failure codes
- Track improvement in specific areas
- Compare failure rates across categories

### Call Volume Analysis

**Location**: Call Volume Analysis section

**Visualizations**:
- **Volume by Agent**: Bar chart showing call counts per agent
- **Volume Trends Over Time**: Line chart showing daily/weekly volume
- **Volume Distribution**: Histogram of call volumes

**How to Use**:
- Understand workload distribution
- Identify peak periods
- Plan staffing based on volume patterns
- Correlate volume with quality metrics

### Time of Day Analysis

**Location**: Time of Day Analysis section

**Features**:
- **QA Scores by Hour**: Performance patterns throughout the day
- **Call Volume by Hour**: When calls are most frequent
- **Heatmap**: Visual representation of performance by hour

**How to Use**:
- Identify peak performance times
- Schedule important calls during high-performance hours
- Understand workload patterns
- Optimize shift scheduling

### Call Reason & Outcome Analysis

**Location**: Call Reason & Outcome Analysis section (expandable)

**Tabs**:
1. **Reasons**: Most common call reasons
2. **Outcomes**: Most common call outcomes
3. **Products**: Product-related analysis

**Visualizations**:
- Top 10 reasons/outcomes (bar charts)
- Distribution pie charts
- Trends over time

**How to Use**:
- Understand customer needs
- Identify common issues
- Track outcome patterns
- Improve product knowledge

### Coaching Insights

**Location**: Coaching Insights section

**Features**:
- **Most Common Coaching Suggestions**: Frequency analysis
- **Coaching by Category**: Grouped by topic
- **Trend Analysis**: How coaching needs change over time

**How to Use**:
- Identify recurring coaching needs
- Develop targeted training programs
- Track improvement in coached areas
- Share best practices

### Week-over-Week Comparison

**Location**: Analytics → Week-over-Week Comparison tab

**Features**:
- Compare current week to previous week
- Key metrics side-by-side
- Percentage change indicators
- Visual comparisons

**Metrics Compared**:
- Total calls
- Average QA score
- Pass rate
- Average call duration
- Top failing rubric codes

**How to Use**:
- Track weekly progress
- Identify week-to-week changes
- Celebrate improvements
- Address declining metrics

### Agent Improvement Trends

**Location**: Analytics → Agent Improvement Trends tab

**Features**:
- Individual agent trend lines
- Improvement/decline indicators
- Period-over-period comparisons
- Ranking changes

**How to Use**:
- Track individual agent progress
- Identify consistent improvers
- Support agents who need help
- Recognize improvement achievements

---

## Agent-Specific Features

### My Performance Dashboard

**Access**: Automatically shown when logged in as an agent

**Features**:
- **Personalized View**: Only your calls are displayed
- **Team Comparison**: Your metrics vs team averages
- **Performance Summary**: Quick overview of your stats
- **Trend Analysis**: Your performance over time

### My Performance Summary

**Metrics Shown**:
- Total Calls
- Average QA Score (vs Team Average)
- Pass Rate (vs Team Average)
- Average Call Duration (vs Team Average)

**How to Use**:
- Understand where you stand relative to the team
- Identify areas for improvement
- Track your progress over time
- Set personal goals

### My Performance Trend vs Team Average

**Visualizations**:
- Your QA score trend line
- Team average trend line
- Pass rate comparison
- Call duration comparison

**How to Use**:
- See if you're improving faster than the team
- Identify periods of strong/weak performance
- Understand seasonal patterns
- Set realistic improvement targets

### Individual Call Details

**Location**: Individual Call Details section

**For Each Call, You Can See**:
- Call ID and date
- QA Score and label
- Reason, Summary, and Outcome
- Strengths and Challenges
- Coaching Suggestions
- Complete Rubric Details
- Export individual call report

**How to Use**:
- Review specific calls in detail
- Understand why you received a particular score
- Learn from strengths and challenges
- Use coaching suggestions for improvement
- Export for personal records or coaching sessions

---

## Admin Features

### Full Data Access

**Access**: Available to users without agent_id mapping

**Features**:
- View all agents' calls
- Compare all agents side-by-side
- Access complete analytics
- Export comprehensive reports

### Agent Leaderboard

**Location**: Agent Performance section

**Features**:
- All agents ranked by performance
- Multiple sortable metrics
- Filter by date range
- Export leaderboard data

**How to Use**:
- Identify top performers for recognition
- Find agents who need support
- Track team-wide trends
- Make data-driven decisions

### Performance Alerts (Admin View)

**Features**:
- **Low Score Calls**: All calls below threshold
- **At-Risk Agents**: Agents trending toward threshold
- **Risk Analysis**: Detailed risk factor breakdown
- **Proactive Alerts**: Early warning system

**How to Use**:
- Set appropriate thresholds
- Monitor at-risk agents
- Provide timely coaching
- Prevent quality issues before they become problems

### System Monitoring (Super Admin Only)

**Location**: System Monitoring & Metrics (expandable)

**Features**:
- **Usage Metrics**: Total sessions, errors, features used
- **Error Tracking**: Repeated failures and recent errors
- **Feature Usage**: Most used features
- **Audit Log**: Complete activity log (Chloe & Shannon only)

**How to Use**:
- Monitor system health
- Identify technical issues
- Understand feature adoption
- Track user activity

### Data Management

**Location**: Sidebar buttons

**Features**:
- **Refresh New Data**: Process only new CSV files (fast)
- **Reload ALL Data**: Clear cache and reload everything (admin only)

**When to Use**:
- **Refresh New Data**: After new files are added to S3 (recommended)
- **Reload ALL Data**: When cache is corrupted or after major data changes

---

## Export Options

### Excel Export

**Location**: Export Options section

**What's Included**:
- **QA Data Sheet**: All filtered call data with all columns
- **Agent Performance Sheet**: Aggregated metrics by agent
- **Summary Sheet**: Overall statistics and KPIs
- **Charts Sheet**: Embedded visualizations

**How to Use**:
1. Apply your desired filters
2. Click "Export to Excel"
3. File downloads automatically
4. Open in Excel for further analysis

**Use Cases**:
- Share data with stakeholders
- Create custom reports
- Perform additional analysis
- Archive data for records

### PDF Export

**Location**: Export Options section

**What's Included**:
- All charts and visualizations
- Professional formatting
- Multiple pages as needed
- Print-ready format

**How to Use**:
1. Set up your dashboard view
2. Apply desired filters
3. Click "Export Charts to PDF"
4. File downloads automatically

**Use Cases**:
- Executive presentations
- Printed reports
- Email attachments
- Documentation

### Individual Call Report

**Location**: Individual Call Details section → Export button

**What's Included**:
- Complete call information
- All rubric details
- Strengths and challenges
- Coaching suggestions
- Formatted text file

**How to Use**:
1. Navigate to Individual Call Details
2. Find the call you want
3. Click "Export Call Report"
4. File downloads as text file

**Use Cases**:
- One-on-one coaching sessions
- Personal records
- Performance reviews
- Training materials

### Rubric Export

**Location**: Rubric Reference section

**What's Included**:
- Complete rubric (1,500+ items)
- All sections and categories
- Full details for each item
- Excel format with formatting

**How to Use**:
1. Navigate to Rubric Reference section
2. Click "Download Rubric as Excel"
3. File downloads automatically

**Use Cases**:
- Training materials
- Reference documentation
- Offline access
- Sharing with team

---

## Rubric Reference

### Search All Items

**Location**: Rubric Reference → Search All Items tab

**Features**:
- **Full-Text Search**: Search across all rubric items
- **Real-Time Results**: Results update as you type
- **Highlighted Matches**: Search terms are highlighted
- **Detailed View**: See full description for each item

**How to Use**:
1. Type your search term
2. Browse matching results
3. Click on items to see full details
4. Use for quick reference during call reviews

### Browse by Section

**Location**: Rubric Reference → Browse by Section tab

**Features**:
- **Section Navigation**: Browse by rubric sections
- **Category Filter**: Filter by category within sections
- **Hierarchical View**: See section → category → items
- **Quick Access**: Jump to specific sections

**How to Use**:
1. Select a section from dropdown
2. Optionally filter by category
3. Browse items in that section
4. Click items for full details

### Rubric Details

**For Each Rubric Item, You Can See**:
- **Code**: Unique identifier (e.g., "1.2.3")
- **Section**: Which section it belongs to
- **Category**: Category classification
- **Description**: Full item description
- **Notes**: Additional information if available

**How to Use**:
- Understand scoring criteria
- Prepare for QA reviews
- Learn expectations
- Improve performance

---

## Tips & Best Practices

### Filtering Tips

1. **Start Broad, Then Narrow**: Begin with a wide date range, then apply specific filters
2. **Combine Filters**: Use multiple filters together for precise analysis
3. **Save Common Filters**: Note your favorite filter combinations for quick access
4. **Reset When Needed**: Clear filters to see the full picture

### Performance Analysis Tips

1. **Look for Trends**: Don't focus on single data points; look for patterns
2. **Compare Periods**: Use date filters to compare different time periods
3. **Identify Root Causes**: Use rubric code analysis to find underlying issues
4. **Track Improvements**: Regularly review trends to see progress

### Agent Tips

1. **Review Your Trends**: Check your performance trends regularly
2. **Compare to Team**: Understand where you stand relative to peers
3. **Focus on Weak Areas**: Use rubric code analysis to identify improvement areas
4. **Learn from Strengths**: Review your strong areas to maintain performance
5. **Export for Records**: Save your performance data for personal tracking

### Admin Tips

1. **Set Appropriate Thresholds**: Configure alert thresholds based on your standards
2. **Monitor At-Risk Agents**: Use predictive analytics to prevent issues
3. **Regular Reviews**: Schedule regular dashboard reviews
4. **Share Insights**: Export and share key findings with the team
5. **Celebrate Success**: Use leaderboards to recognize top performers

### Data Management Tips

1. **Use Refresh New Data**: Prefer "Refresh New Data" over full reload when possible
2. **Cache is Your Friend**: Let the cache work; it speeds up subsequent loads
3. **Full Reload Sparingly**: Only use "Reload ALL Data" when necessary
4. **Monitor System Health**: Check system metrics if experiencing issues

### Export Tips

1. **Filter Before Exporting**: Apply filters to export only relevant data
2. **Use Excel for Analysis**: Excel exports are great for further analysis
3. **PDF for Presentations**: Use PDF exports for sharing with stakeholders
4. **Individual Reports for Coaching**: Export individual call reports for one-on-ones

---

## Keyboard Shortcuts & Quick Actions

### Streamlit Shortcuts

- **R**: Rerun the app
- **C**: Clear cache
- **?**: Show keyboard shortcuts (press `?` in the app)

### Quick Navigation

- **Scroll to Top**: Click the sidebar header to jump to top
- **Expand/Collapse**: Click section headers to toggle
- **Switch Tabs**: Click tab names to switch views

---

## Troubleshooting

### Data Not Loading

**If data isn't loading**:
1. Check your internet connection
2. Verify S3 credentials are correct
3. Try "Refresh New Data" button
4. Contact admin if issue persists

### Filters Not Working

**If filters aren't applying**:
1. Clear all filters and reapply
2. Check that you have data in the selected date range
3. Verify filter selections are valid
4. Refresh the page if needed

### Performance Issues

**If dashboard is slow**:
1. Apply date filters to reduce data volume
2. Use "Refresh New Data" instead of full reload
3. Clear browser cache
4. Close other browser tabs

### Export Issues

**If exports aren't working**:
1. Ensure you have data in your filtered view
2. Check browser download settings
3. Try a different browser
4. Contact support if problem persists

---

## Getting Help

### Support Resources

- **Documentation**: Refer to this guide and README.md
- **Admin Support**: Contact your dashboard administrator
- **Technical Issues**: Report to the development team

### Common Questions

**Q: Why can't I see other agents' data?**
A: If you're logged in as an agent, you'll only see your own calls. Admins can see all data.

**Q: How often is data updated?**
A: Data is cached for performance. Use "Refresh New Data" to get the latest updates.

**Q: Can I save my filter settings?**
A: Filters are session-based. You'll need to reapply them each session, but the dashboard remembers your last selections.

**Q: How do I export data for a specific agent?**
A: Use the agent filter to select the agent, then export. The export will only include that agent's data.

**Q: What's the difference between "Refresh New Data" and "Reload ALL Data"?**
A: "Refresh New Data" only processes new files (fast). "Reload ALL Data" clears cache and reloads everything (slow, 10-20 minutes).

---

## Feature Summary

### Core Features
✅ Secure user authentication  
✅ Real-time data loading from S3  
✅ Intelligent caching for performance  
✅ Agent-specific and admin views  
✅ Comprehensive filtering and search  

### Analytics Features
✅ Performance metrics and KPIs  
✅ Trend analysis and forecasting  
✅ Rubric code analysis  
✅ Agent leaderboards  
✅ Time-of-day analysis  
✅ Call volume analysis  

### Export Features
✅ Excel exports with multiple sheets  
✅ PDF chart exports  
✅ Individual call reports  
✅ Rubric reference exports  

### Reference Features
✅ Complete rubric search (1,500+ items)  
✅ Browse by section and category  
✅ Detailed rubric item information  

---

## Conclusion

The Jomashop QA Dashboard is a powerful tool for tracking, analyzing, and improving call quality. Whether you're an agent tracking your own performance or an admin managing a team, this dashboard provides the insights you need to drive continuous improvement.

**Remember**:
- Use filters to focus on what matters
- Track trends, not just single data points
- Export data for deeper analysis
- Review regularly for best results

**Happy analyzing!** 🚀

---

*Made with ❤️ by [Valence](https://www.getvalenceai.com) | Jomashop QA Dashboard © 2025*

