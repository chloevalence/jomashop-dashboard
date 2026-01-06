#!/usr/bin/env python3
"""
Agent Mapping Verification Script
Fetches all agents from data and verifies mappings are correct.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import hashlib

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()
log_dir = PROJECT_ROOT / "logs"
CACHE_FILE = log_dir / "cached_calls_data.json"
AGENT_MAPPING_FILE = log_dir / "agent_id_mapping.json"

# Known agent mappings (from streamlit_app_1_3_4.py)
KNOWN_AGENT_MAPPINGS = {
    # Agent 1: Jesus
    "unknown": "Agent 1",
    "bp016803073": "Agent 1",
    "bp016803074": "Agent 1",
    "bp agent 016803073": "Agent 1",
    "bp agent 016803074": "Agent 1",
    # Agent 2: Gerardo
    "bpagent024577540": "Agent 2",
    "bp agent 024577540": "Agent 2",
    # Agent 3: Edgar
    "bpagent030844482": "Agent 3",
    "bp agent 030844482": "Agent 3",
    # Agent 4: Osiris
    "bpagent047779576": "Agent 4",
    "bp agent 047779576": "Agent 4",
    # Agent 6: Daniela
    "bpagent065185612": "Agent 6",
    "bp agent 065185612": "Agent 6",
    # Agent 7: Yasmin
    "bpagent072229421": "Agent 7",
    "bp agent 072229421": "Agent 7",
    # Agent 8: Moises
    "bpagent089724913": "Agent 8",
    "bp agent 089724913": "Agent 8",
    # Agent 9: Marcos
    "bpagent093540654": "Agent 9",
    "bp agent 093540654": "Agent 9",
    # Agent 10: Angel
    "bpagent102256681": "Agent 10",
    "bp agent 102256681": "Agent 10",
    # Agent 5: (Left, but calls need to be accessible to admins)
    "bpagent051705087": "Agent 5",
    "bp agent 051705087": "Agent 5",
    # Agent 11: (No agent account, but viewable to admins)
    "bpagent113827380": "Agent 11",
    "bp agent 113827380": "Agent 11",
}

def load_agent_mapping():
    """Load agent ID mapping from file."""
    if AGENT_MAPPING_FILE.exists():
        try:
            with open(AGENT_MAPPING_FILE, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                # Merge with known mappings (known mappings take precedence)
                merged_mapping = {**mapping, **KNOWN_AGENT_MAPPINGS}
                return merged_mapping
        except Exception as e:
            print(f"Warning: Failed to load agent mapping file: {e}")
            return KNOWN_AGENT_MAPPINGS.copy()
    return KNOWN_AGENT_MAPPINGS.copy()

def get_or_create_agent_mapping(agent_id_lower, mapping):
    """Get agent number for an agent ID (matches streamlit_app logic)."""
    # Normalize by removing spaces
    agent_id_normalized = agent_id_lower.replace(" ", "").replace("_", "")
    
    # Check if already mapped
    if agent_id_lower in mapping:
        return mapping[agent_id_lower]
    if agent_id_normalized in mapping:
        return mapping[agent_id_normalized]
    
    # Check if already in normalized format
    if agent_id_lower.startswith("agent "):
        agent_num_str = agent_id_lower.replace("agent ", "").strip()
        try:
            agent_num = int(agent_num_str)
            return f"Agent {agent_num}"
        except ValueError:
            pass
    
    # Check KNOWN_AGENT_MAPPINGS first
    for known_id, known_name in KNOWN_AGENT_MAPPINGS.items():
        known_id_normalized = known_id.replace(" ", "").replace("_", "")
        if agent_id_normalized == known_id_normalized or agent_id_lower == known_id.lower():
            return known_name
    
    # Check special cases
    if agent_id_normalized == "unknown" or agent_id_lower == "unknown":
        return "Agent 1"
    if agent_id_normalized in ["bp016803073", "bp016803074"]:
        return "Agent 1"
    if agent_id_normalized.startswith("bp01"):
        return "Agent 1"
    
    # Hash-based assignment
    hash_value = int(hashlib.md5(agent_id_normalized.encode()).hexdigest(), 16)
    agent_number = (hash_value % 99) + 1
    # Skip 5 and 11 in hash assignment (they're reserved for known mappings)
    if agent_number == 5:
        agent_number = 12  # Skip 5, use 12 instead
    if agent_number == 11:
        agent_number = 12  # Skip 11, use 12 instead
    
    return f"Agent {agent_number}"

def main():
    print("=" * 80)
    print("AGENT MAPPING VERIFICATION")
    print("=" * 80)
    print()
    
    # Load agent mapping file
    print(f"📂 Loading agent mapping from: {AGENT_MAPPING_FILE}")
    mapping = load_agent_mapping()
    print(f"✅ Loaded {len(mapping)} mappings from file")
    print()
    
    # Load cached call data
    if not CACHE_FILE.exists():
        print(f"❌ Cache file not found: {CACHE_FILE}")
        print("   Run the Streamlit app first to create the cache.")
        return
    
    print(f"📂 Loading call data from: {CACHE_FILE}")
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        
        if isinstance(cached_data, dict):
            call_data = cached_data.get("call_data", cached_data.get("calls", cached_data.get("data", [])))
        else:
            call_data = cached_data
        
        if not call_data:
            print("❌ No call data found in cache")
            return
        
        print(f"✅ Loaded {len(call_data)} calls")
        print()
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return
    
    # Extract all unique raw agent IDs from data
    print("🔍 Analyzing agent IDs in data...")
    raw_agent_ids = set()
    agent_id_to_calls = defaultdict(int)
    
    for call in call_data:
        if isinstance(call, dict):
            # Check both "agent" and "Agent" keys
            agent_id = call.get("agent") or call.get("Agent")
            if agent_id and not pd.isna(agent_id):
                agent_str = str(agent_id).strip()
                if agent_str:
                    raw_agent_ids.add(agent_str)
                    agent_id_to_calls[agent_str] += 1
    
    print(f"✅ Found {len(raw_agent_ids)} unique raw agent IDs")
    print()
    
    # Map each raw agent ID to normalized format
    print("=" * 80)
    print("AGENT ID MAPPINGS")
    print("=" * 80)
    print()
    
    # Group by normalized agent
    normalized_to_raw = defaultdict(list)
    raw_to_normalized = {}
    
    for raw_id in sorted(raw_agent_ids):
        raw_id_lower = raw_id.lower().strip()
        normalized = get_or_create_agent_mapping(raw_id_lower, mapping)
        raw_to_normalized[raw_id] = normalized
        normalized_to_raw[normalized].append(raw_id)
    
    # Display mappings grouped by normalized agent
    print("📊 Mappings (Grouped by Normalized Agent):")
    print()
    
    for normalized_agent in sorted(normalized_to_raw.keys(), key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 999):
        raw_ids = normalized_to_raw[normalized_agent]
        total_calls = sum(agent_id_to_calls[raw_id] for raw_id in raw_ids)
        
        print(f"  {normalized_agent}:")
        print(f"    Total calls: {total_calls}")
        print(f"    Raw agent IDs ({len(raw_ids)}):")
        for raw_id in sorted(raw_ids):
            call_count = agent_id_to_calls[raw_id]
            raw_id_lower_check = raw_id.lower()
            is_known = raw_id_lower_check in KNOWN_AGENT_MAPPINGS or raw_id_lower_check.replace(" ", "") in [k.replace(" ", "") for k in KNOWN_AGENT_MAPPINGS.keys()]
            known_marker = " ✅ KNOWN" if is_known else " ⚠️  UNKNOWN"
            print(f"      - '{raw_id}' ({call_count} calls){known_marker}")
        print()
    
    # Check for issues
    print("=" * 80)
    print("ISSUES DETECTED")
    print("=" * 80)
    print()
    
    issues_found = False
    
    # Check for unexpected agent numbers (> 11, since Agent 11 is now valid)
    unexpected_agents = [agent for agent in normalized_to_raw.keys() 
                        if agent.split()[-1].isdigit() and int(agent.split()[-1]) > 11]
    
    if unexpected_agents:
        issues_found = True
        print(f"⚠️  Unexpected agent numbers found: {', '.join(sorted(unexpected_agents, key=lambda x: int(x.split()[-1])))}")
        print("   These should be mapped to known agents (1-11).")
        print()
        for agent in sorted(unexpected_agents, key=lambda x: int(x.split()[-1])):
            print(f"   {agent} maps from:")
            for raw_id in normalized_to_raw[agent]:
                print(f"     - '{raw_id}' ({agent_id_to_calls[raw_id]} calls)")
        print()
    
    # Check if known agents are correctly mapped
    print("🔍 Verifying known agent mappings...")
    known_issues = []
    for known_id, expected_agent in KNOWN_AGENT_MAPPINGS.items():
        # Check both exact match and case-insensitive
        found = False
        for raw_id in raw_agent_ids:
            if raw_id.lower() == known_id.lower() or raw_id.lower().replace(" ", "") == known_id.lower().replace(" ", ""):
                actual_agent = raw_to_normalized.get(raw_id)
                if actual_agent != expected_agent:
                    known_issues.append((raw_id, expected_agent, actual_agent))
                found = True
                break
    
    if known_issues:
        issues_found = True
        print("❌ Known agent mapping mismatches:")
        for raw_id, expected, actual in known_issues:
            print(f"   '{raw_id}' should map to {expected} but maps to {actual}")
        print()
    else:
        print("✅ All known agent IDs are correctly mapped")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique raw agent IDs: {len(raw_agent_ids)}")
    print(f"Total unique normalized agents: {len(normalized_to_raw)}")
    expected_count = len([a for a in normalized_to_raw.keys() if a.split()[-1].isdigit() and 1 <= int(a.split()[-1]) <= 11])
    print(f"Expected agents (1-11): {expected_count}")
    print(f"Unexpected agents (>11): {len(unexpected_agents)}")
    print()
    
    if not issues_found:
        print("✅ No issues detected! All agents are correctly mapped.")
    else:
        print("⚠️  Issues detected. Review the mappings above.")

if __name__ == "__main__":
    main()

