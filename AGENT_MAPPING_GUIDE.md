# Agent ID Mapping Guide

## How to Map Users to Agent IDs

To enable agent-specific views, you need to add agent ID mappings to your `.streamlit/secrets.toml` file.

## Step 1: Find Agent IDs

Agent IDs are extracted from PDF filenames. They look like:
- `bpagent030844482`
- `bpagent123456789`
- etc.

You can see all available agent IDs by:
1. Running the dashboard
2. Looking at the "Agent" column in the data
3. Or checking the PDF filenames in S3

## Step 2: Add Mapping to secrets.toml

Add this section to your `.streamlit/secrets.toml` file:

```toml
[user_mapping]
# Map usernames to agent IDs
# Leave agent_id empty or omit for admin access (can see all data)

[user_mapping.daniel]
agent_id = "bpagent030844482"  # Replace with actual agent ID

[user_mapping.alexa]
agent_id = "bpagent123456789"  # Replace with actual agent ID

[user_mapping.jordi]
agent_id = "bpagent987654321"  # Replace with actual agent ID

# ... add for each agent user

# Admins (no agent_id = can see all data)
[user_mapping.chloe]
# No agent_id = admin access

[user_mapping.shannon]
# No agent_id = admin access
```

## Step 3: How It Works

- **Agents** (users with `agent_id`): 
  - Only see their own calls
  - See comparison to team average
  - Cannot see other agents' data

- **Admins** (users without `agent_id` or chloe/shannon):
  - See all data
  - Can filter by any agent
  - Full dashboard access

## Example Mapping

If agent "bpagent030844482" corresponds to user "daniel":

```toml
[user_mapping.daniel]
agent_id = "bpagent030844482"
```

When Daniel logs in, he will:
- Only see calls where Agent = "bpagent030844482"
- See his performance compared to team average
- Not see other agents' data

