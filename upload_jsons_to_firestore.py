import os
import json
import zipfile
from pathlib import Path
from datetime import datetime
from dateutil import parser as date_parser  # pip install python-dateutil
import firebase_admin
from firebase_admin import credentials, firestore

# --- CONFIGURATION ---
zip_path = "/Users/Chloe/Downloads/JSONsLastWeek.zip"
firebase_key_path = "/Users/Chloe/Downloads/valence-acsi-dashboard-firebase-adminsdk-fbsvc-e8065d1b80.json"
collection_name = "calls"
skipped_log_path = "skipped_files.txt"

# --- INITIALIZE FIREBASE ---
print("✅ Initializing Firebase...")
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- UNZIP FILES ---
print("✅ Unzipping JSON files...")
temp_dir = Path("/tmp/json_upload")
if temp_dir.exists():
    for f in temp_dir.rglob("*"):
        if f.is_file():
            f.unlink()
else:
    temp_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(temp_dir)

# --- VALIDATION FUNCTION ---
def is_valid_json(filepath: Path):
    # 1) Skip any ".mp3.json" dumps
    if filepath.name.endswith(".mp3.json"):
        return False, "Filename is an mp3 dump"

    # 2) Must parse as JSON
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"JSON parsing error: {e}"

    # 3) metadata must include agent, company, time, date
    metadata = data.get("metadata", {})
    for field in ("agent", "company", "time", "date"):
        if not metadata.get(field):
            return False, f"Missing metadata field: {field}"

    # 4) Must have an average_happiness_value
    if data.get("average_happiness_value") is None:
        return False, "Missing average_happiness_value"

    # If we reach here, it’s valid
    return True, data

# --- PROCESS & UPLOAD ---
skipped_files = []
uploaded_count = 0

for file_path in temp_dir.rglob("*.json"):
    valid, result = is_valid_json(file_path)
    if not valid:
        skipped_files.append((file_path.name, result))
        continue

    data = result
    # --- Flatten metadata ---
    metadata    = data.get("metadata", {})
    document_id = metadata.get("call_id", file_path.stem)
    agent       = metadata.get("agent")
    company     = metadata.get("company")
    time_of_day = metadata.get("time")
    date_raw    = metadata.get("date")  # MMDDYYYY string

    # --- Parse the real timestamp from top-level call_date ---
    try:
        parsed_call_date = datetime.strptime(date_raw, "%m%d%Y")
    except Exception:
        parsed_call_date = None

    # --- Flatten emotion_counts dict ---
    ec      = data.get("emotion_counts", {})
    angry   = ec.get("angry",   0)
    happy   = ec.get("happy",   0)
    neutral = ec.get("neutral", 0)
    sad     = ec.get("sad",     0)

    # --- Speaking times (optional) ---
    speaking = data.get("speaking_time_per_speaker", {})

    # --- Build the payload ---
    upload_payload = {
        "call_id":                  document_id,
        "agent":                    agent,
        "company":                  company,
        "time":                     time_of_day,
        "date_raw":                 date_raw,
        "call_date":                parsed_call_date,
        "average_happiness_value":  data.get("average_happiness_value"),
        "low_confidences":          metadata.get("low_confidences", data.get("low_confidences", 0)),
        "angry":                    angry,
        "happy":                    happy,
        "neutral":                  neutral,
        "sad":                      sad,
        "speaking_time_per_speaker": speaking,
        # ... add other top-level fields if needed ...
    }

    # --- Upload to Firestore ---
    try:
        db.collection(collection_name).document(document_id).set(upload_payload)
        uploaded_count += 1
        if uploaded_count % 100 == 0:
            print(f"✅ Uploaded {uploaded_count} calls...")
    except Exception as e:
        skipped_files.append((file_path.name, f"Firestore error: {e}"))

# --- LOG SKIPPED FILES ---
if skipped_files:
    with open(skipped_log_path, "w") as f:
        for fn, reason in skipped_files:
            f.write(f"{fn} — {reason}\n")
    print(f"⚠️ Skipped {len(skipped_files)} files. Details logged to {skipped_log_path}")
else:
    print("✅ No skipped files.")

print(f"🎉 Upload complete: {uploaded_count} calls uploaded successfully.")
