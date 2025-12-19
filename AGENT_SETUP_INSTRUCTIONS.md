# Agent-Specific Views Setup Instructions

## Overview

The dashboard now supports **agent-specific views** where agents can:
- ✅ See only their own calls
- ✅ Compare their performance to team averages
- ✅ View their trends over time vs team trends

Admins (like Chloe and Shannon) can see all data and filter by any agent.

## How It Works

### For Agents:
1. When an agent logs in with their username
2. The system checks if they have an `agent_id` mapping in `secrets.toml`
3. If found, they only see calls where `Agent` field matches their `agent_id`
4. All metrics and charts show their data compared to team averages

### For Admins:
1. Users without `agent_id` mapping (or chloe/shannon) see all data
2. Can filter by any agent
3. Full dashboard access

## Setup Steps

### Step 1: Find Agent IDs

Agent IDs come from the PDF data. To find them:

1. **Option A:** Run the dashboard as admin and look at the "Agent" column
2. **Option B:** Check S3 bucket PDF filenames - agent IDs are in the filenames
3. **Option C:** Look at the data after loading - agent IDs are in the `Agent` field

Example agent IDs:
- `bpagent030844482`
- `bpagent123456789`
- etc.

### Step 2: Map Usernames to Agent IDs

Edit `.streamlit/secrets.toml` and add a `[user_mapping]` section:

```toml
# Add this section to your secrets.toml

[user_mapping]
# Map usernames to agent IDs from your PDF data

[user_mapping.daniel]
agent_id = "bpagent030844482"  # Replace with actual agent ID from your data

[user_mapping.alexa]
agent_id = "bpagent123456789"  # Replace with actual agent ID

[user_mapping.jordi]
agent_id = "bpagent987654321"  # Replace with actual agent ID

# ... add mappings for each agent

# Admins - leave agent_id empty or omit
[user_mapping.chloe]
# No agent_id = admin access (can see all data)

[user_mapping.shannon]
# No agent_id = admin access
```

### Step 3: Verify the Mapping

1. Log in as an agent user (e.g., "daniel")
2. You should see:
   - "👤 Agent View: bpagent030844482" in the sidebar
   - Only calls for that agent
   - Comparison metrics showing "My Performance vs Team Average"

3. Log in as admin (e.g., "chloe")
4. You should see:
   - "👑 Admin View: All Data" in the sidebar
   - All calls from all agents
   - Full filtering capabilities

## What Agents See

### Dashboard Sections for Agents:

1. **Summary Metrics** - Shows:
   - My Calls (vs total team calls)
   - My Avg Score (vs team average, with delta)
   - My Pass Rate (vs team average, with delta)
   - Overall Avg Score
   - Overall Pass Rate

2. **My Performance vs Team Average** - Comparison charts:
   - QA Score comparison bar chart
   - Pass Rate comparison bar chart

3. **My Performance Summary** - Table comparing:
   - My Performance vs Team Average for key metrics

4. **My Performance Trend vs Team Average** - Line charts:
   - My QA Score trend over time vs team average
   - My Pass Rate trend over time vs team average

5. **All Other Sections** - Filtered to show only their data:
   - Rubric Code Analysis (their codes only)
   - Individual Call Details (their calls only)
   - Time of Day Analysis (their calls only)
   - etc.

## What Admins See

- Full dashboard with all agents' data
- Agent filter dropdown to select specific agents
- Agent Performance leaderboard
- All comparison features

## Troubleshooting

### Agent sees "No calls found"
- Check that the `agent_id` in secrets matches exactly the `Agent` field in your PDF data
- Agent IDs are case-sensitive
- Check for extra spaces or typos

### Agent sees all data instead of filtered
- Verify the `[user_mapping]` section exists in secrets.toml
- Check that the username matches exactly (case-sensitive)
- Make sure `agent_id` is set (not empty)

### Admin can't see all data
- Remove `agent_id` from their mapping, or
- Don't include them in `[user_mapping]` at all
- Users "chloe" and "shannon" are default admins

## Example Complete Mapping

```toml
[user_mapping]

# Agent users
[user_mapping.daniel]
agent_id = "bpagent030844482"

[user_mapping.alexa]
agent_id = "bpagent123456789"

[user_mapping.jordi]
agent_id = "bpagent987654321"

[user_mapping.maria]
agent_id = "bpagent111222333"

# Admins (no agent_id)
[user_mapping.chloe]

[user_mapping.shannon]
```

## Notes

- Agent IDs must match exactly what's in your PDF data
- If an agent's ID changes, update the mapping
- New agents can be added by adding their mapping
- The system defaults to admin view if no mapping is found

