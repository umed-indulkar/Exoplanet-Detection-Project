import pandas as pd
import os
import shutil

# --- PATHS ---
MASTER_PATH   = r"D:\ppp\data\fetched\master_koi_catalog.csv"
SOURCE_DIR    = r"D:\ppp\data\raw_files"
CANDIDATE_DIR = r"D:\ppp\data\dataset_500\candidates_only"

if not os.path.exists(CANDIDATE_DIR):
    os.makedirs(CANDIDATE_DIR)

# 1. Load the catalog
print("Reading Master Catalog...")
df = pd.read_csv(MASTER_PATH)

# 2. Get list of KIDs that are ONLY Candidates
# We filter for stars where the disposition is 'CANDIDATE'
candidate_kids = df[df['koi_disposition'] == 'CANDIDATE']['kepid'].unique().tolist()

print(f"Found {len(candidate_kids)} unique Candidate Stars in catalog.")

moved_count = 0

# 3. Match by KID number in the filename
for kid in candidate_kids:
    # Convert to string and remove leading zeros to match your 'res_12345.csv' format
    kid_str = str(kid).lstrip('0')
    
    # Construct the exact filename you confirmed
    filename = f"res_{kid_str}_1.csv"
    src_path = os.path.join(SOURCE_DIR, filename)
    
    if os.path.exists(src_path):
        try:
            shutil.move(src_path, os.path.join(CANDIDATE_DIR, filename))
            moved_count += 1
        except Exception as e:
            print(f"Error moving {filename}: {e}")

print(f"--- PROCESS COMPLETE ---")
print(f"SUCCESS: Moved {moved_count} files to {CANDIDATE_DIR}")
print(f"REMAINING: Files in {SOURCE_DIR} are now your Training Set.")