# Anonymous Mode Setup

The dashboard now supports an **Anonymous Mode** that automatically de-identifies all data for privacy-compliant sharing.

## Features

When logged in as the "anonymous" user:
- ✅ **Agent names** are replaced with `Agent-001`, `Agent-002`, etc.
- ✅ **Company names** are replaced with `Company-001`, `Company-002`, etc.
- ✅ **Call IDs** are replaced with `Call-001`, `Call-002`, etc.
- ✅ **Consistent mapping**: Same real values always map to same anonymous IDs
- ✅ **All views anonymized**: Tables, charts, exports, and filters all use anonymized data
- ✅ **Exports anonymized**: Excel and CSV exports contain only anonymized data

## Setup

### 1. Add Anonymous User to Secrets

Add the anonymous user to your `.streamlit/secrets.toml` file:

```toml
[credentials.usernames.anonymous]
email = "anonymous@example.com"
name = "Anonymous User"
password = "$2b$12$..."  # Use streamlit-authenticator to hash a password
```

### 2. Generate Password Hash

To generate a password hash for the anonymous user, run this in Python:

```python
import streamlit_authenticator as stauth
hashed_password = stauth.Hasher(['your_password_here']).generate()[0]
print(hashed_password)
```

Then copy the hash into your `secrets.toml` file.

### 3. Login

1. Start the Streamlit app
2. Login with username: `anonymous` and your chosen password
3. You'll see a warning banner: "🔒 **Anonymous Mode**: All identifying information has been anonymized"
4. All data will be automatically anonymized

## How It Works

- **Consistent Mapping**: The anonymization uses sorted unique values to create deterministic mappings. This means:
  - The same agent always gets the same anonymous ID (e.g., "bpagent01" → "Agent-001")
  - The same company always gets the same anonymous ID
  - The same call ID always gets the same anonymous ID
  
- **Session Persistence**: Mappings are stored in session state, so they remain consistent throughout your session

- **Full Coverage**: Anonymization applies to:
  - All data tables and displays
  - Charts and visualizations
  - Excel exports
  - CSV exports
  - Filtered views
  - Agent-specific views

## Example

**Before (Admin View):**
- Agent: `bpagent01`
- Company: `Jomashop`
- Call ID: `CALL-2024-001`

**After (Anonymous View):**
- Agent: `Agent-001`
- Company: `Company-001`
- Call ID: `Call-001`

## Notes

- The anonymization is **one-way**: There's no way to reverse the mapping from anonymous IDs back to real values
- The mapping is **deterministic**: Same inputs always produce same outputs
- **All other data is preserved**: QA scores, dates, times, rubric details, etc. remain unchanged
- Only identifying information (Agent, Company, Call ID) is anonymized

