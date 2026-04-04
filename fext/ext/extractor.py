import pandas as pd
import numpy as np
import multiprocessing
import os
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tqdm import tqdm

# ---------- CONFIGURATION ----------
INPUT_PATH = r"D:\ppp\data\dataset_500\raw_curve_500_cleaned.csv"
OUTPUT_PATH = r"D:\ppp\data\features_500\features_curve_500.csv"
CHECKPOINT_DIR = r"D:\ppp\data\features_500\checkpoints"
WORKERS = 8       # Adjust to your CPU cores (e.g., 4, 8, 16)
CHUNK_SIZE = 50   # Number of planets per worker task

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------- HELPER: RAM MONITOR ----------
def print_ram_usage():
    mem = psutil.virtual_memory()
    print(f"--- [RAM Status] Used: {mem.percent}% ({mem.used / (1024**3):.2f}GB / {mem.total / (1024**3):.2f}GB) ---")

# ---------- STEP 1: WIDE → LONG (MEMORY EFFICIENT) ----------
def wide_to_long(df):
    print("🔄 Converting Wide format to Long (Time Series)...")
    flux_cols = [c for c in df.columns if "flux_" in c.lower()]
    
    # We keep 'planet_id' as the anchor
    df_long = df.melt(id_vars=["planet_id"], value_vars=flux_cols, var_name="time", value_name="flux")
    
    # Clean string 'flux_1' to integer 1 for sorting
    df_long["time"] = df_long["time"].str.replace("flux_", "", case=False).astype(int)
    
    gc.collect()
    return df_long

# ---------- STEP 2: EXTRACTION WORKER ----------
def extraction_worker(task):
    df_chunk, params = task
    # Returns extracted features for a small chunk of planets
    return extract_features(
        df_chunk,
        column_id="planet_id",
        column_sort="time",
        column_value="flux",
        default_fc_parameters=params,
        n_jobs=0, # Important: 0 tells tsfresh not to spawn its own sub-processes
        impute_function=impute,
        disable_progressbar=True
    )

# ---------- STEP 3: PIPELINE CORE ----------
class ExoplanetProcessor:
    def __init__(self, workers):
        self.workers = workers
        self.params = EfficientFCParameters()

    def run(self, df_long):
        ids = df_long["planet_id"].unique()
        chunks = [df_long[df_long["planet_id"].isin(ids[i:i+CHUNK_SIZE])] 
                  for i in range(0, len(ids), CHUNK_SIZE)]
        
        tasks = [(chunk, self.params) for chunk in chunks]
        results = []

        print(f"\n🚀 Starting Parallel Extraction on {len(ids)} planets using {self.workers} workers...")
        print_ram_usage()

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(extraction_worker, t) for t in tasks]
            
            # TQDM provides the ETA and progress bar
            with tqdm(total=len(futures), desc="Extracting Features", unit="chunk") as pbar:
                for i, f in enumerate(as_completed(futures)):
                    res = f.result()
                    results.append(res)
                    
                    # Periodic Local Checkpoint (Every 10 chunks)
                    if i % 10 == 0:
                        temp_df = pd.concat(results)
                        temp_df.to_csv(os.path.join(CHECKPOINT_DIR, f"checkpoint_latest.csv"))
                    
                    pbar.update(1)
        
        print("\n✅ Extraction Complete. Concatenating results...")
        return pd.concat(results).sort_index()

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    # 1. Load Data
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: {INPUT_PATH} not found.")
        exit()

    raw_df = pd.read_csv(INPUT_PATH)
    raw_df['planet_id'] = raw_df.index # Generate ID if missing
    
    # Save labels to re-join later
    labels = raw_df[['planet_id', 'Label']].set_index('planet_id')

    # 2. Convert Format
    long_df = wide_to_long(raw_df)
    del raw_df # Free memory
    gc.collect()

    # 3. Process
    processor = ExoplanetProcessor(workers=WORKERS)
    X_features = processor.run(long_df)

    # 4. Final Join and Save
    print("💾 Finalizing dataset...")
    final_dataset = X_features.join(labels)
    final_dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✨ SUCCESS!")
    print(f"📂 Final Output: {OUTPUT_PATH}")
    print(f"📊 Features Created: {X_features.shape[1]}")
    print_ram_usage()