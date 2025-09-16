import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase if not already
cred = credentials.Certificate("/Users/Chloe/Downloads/valence-acsi-dashboard-firebase-adminsdk-fbsvc-e8065d1b80.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Count documents
calls_ref = db.collection('calls')
docs = calls_ref.stream()

count = sum(1 for _ in docs)
print(f"✅ Total documents in Firestore 'calls' collection: {count}")

calls_ref = db.collection('calls')
docs = calls_ref.stream()

missing_call_date = []
for doc in docs:
    data = doc.to_dict()
    if not data.get("Call Date"):
        missing_call_date.append(doc.id)

print(f"❓ Documents missing 'Call Date': {len(missing_call_date)}")
print(f"Example missing documents: {missing_call_date[:5]}")

import zipfile
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore

# --- CONFIGURATION ---

# Your paths
zip_path = "/Users/Chloe/Downloads/JSONs-20250424T060428Z-001.zip"
firebase_key_path = "/Users/Chloe/Downloads/valence-acsi-dashboard-firebase-adminsdk-fbsvc-e8065d1b80.json"

# --- SETUP ---

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- STEP 1: Read all JSON filenames from the ZIP ---

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    all_json_filenames = [Path(f).stem for f in zip_ref.namelist() if f.endswith(".json")]

print(f"✅ Total JSON files inside ZIP: {len(all_json_filenames)}")

# --- STEP 2: Fetch all document IDs from Firestore ---

calls_ref = db.collection('calls')
docs = calls_ref.stream()

firestore_doc_ids = [doc.id for doc in docs]
print(f"✅ Total documents in Firestore: {len(firestore_doc_ids)}")

# --- STEP 3: Find missing files ---

missing_files = set(all_json_filenames) - set(firestore_doc_ids)

if missing_files:
    print(f"❌ Missing {len(missing_files)} file(s):")
    for missing in missing_files:
        print("-", missing)
else:
    print("✅ No missing files! Everything matches.")

import zipfile
import json
from pathlib import Path

zip_path = "/Users/Chloe/Downloads/JSONs-20250424T060428Z-001.zip"

# Open the zip
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    json_files = [f for f in zip_ref.namelist() if f.endswith(".json")]

    print(f"🔍 Checking {len(json_files)} JSON files...")

    bad_files = []

    for file_name in json_files:
        try:
            with zip_ref.open(file_name) as f:
                content = f.read().decode("utf-8")  # Attempt UTF-8 decode
                json.loads(content)  # Attempt to parse as JSON
        except Exception as e:
            bad_files.append((file_name, str(e)))

# Report
if bad_files:
    print(f"❌ Found {len(bad_files)} bad file(s):")
    for name, error in bad_files:
        print(f"- {name}: {error}")
else:
    print("✅ All JSONs parsed successfully.")

import zipfile
import json
from pathlib import Path

required_fields = ["metadata", "emotion_graph"]

zip_path = "/Users/Chloe/Downloads/JSONs-20250424T060428Z-001.zip"

# Open the zip
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    json_files = [f for f in zip_ref.namelist() if f.endswith(".json")]

    print(f"🔍 Checking {len(json_files)} JSON files for required fields...")

    incomplete_files = []

    for file_name in json_files:
        with zip_ref.open(file_name) as f:
            content = f.read().decode("utf-8")
            data = json.loads(content)
            if not all(field in data for field in required_fields):
                incomplete_files.append(file_name)

# Report
if incomplete_files:
    print(f"❌ Found {len(incomplete_files)} incomplete file(s):")
    for name in incomplete_files:
        print(f"- {name}")
else:
    print("✅ All JSONs have required fields.")
