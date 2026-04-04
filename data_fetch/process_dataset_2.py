import pandas as pd
import numpy as np
import os
import argparse
import sys
from astropy.io import fits

# --- HARD-CODED PATHS ---
MASTER_PATH  = r"D:\ppp\data\fetched\master_koi_catalog.csv"
TEMP_OUT_DIR = r"D:\ppp\data\raw_files" 
NUM_BINS = 500  

def process(file_path):
    time_data = None
    flux_data = None
    
    try:
        if not os.path.exists(MASTER_PATH):
            print(f"PYTHON ERROR: Master catalog missing at {MASTER_PATH}")
            sys.exit(1)
            
        master_df = pd.read_csv(MASTER_PATH)
        fname = os.path.basename(file_path)
        
        kepid_str = fname.split('-')[0].lstrip('0')
        
        match = master_df[master_df['kepid'].astype(str) == kepid_str]
        if match.empty:
            print(f"PYTHON: KID {kepid_str} not found in Candidate CSV. Skipping.")
            sys.exit(1) 
        
        row = match.iloc[0]
        period = float(row['koi_period'])
        epoch = float(row['koi_time0bk'])
        disp = row['koi_disposition']

        # 1. Open and IMMEDIATELY copy data to RAM
        with fits.open(file_path, memmap=False) as hdul:
            time_data = np.copy(hdul[1].data['TIME'])
            flux_data = np.copy(hdul[1].data['PDCSAP_FLUX'])
            
        # 2. Clean NaNs
        mask = ~np.isnan(time_data) & ~np.isnan(flux_data)
        time_clean, flux_clean = time_data[mask], flux_data[mask]
        
        if len(time_clean) < 10: 
            print(f"PYTHON: Insufficient valid data for {kepid_str}.")
            sys.exit(1)

        # 3. Normalize & Phase Fold
        flux_clean /= np.median(flux_clean)
        phase = ((time_clean - epoch + 0.5 * period) % period) / period - 0.5

        # 4. Binning
        bins = np.linspace(-0.5, 0.5, NUM_BINS + 1)
        digitized = np.digitize(phase, bins)
        
        binned_flux = []
        for i in range(1, NUM_BINS + 1):
            subset = flux_clean[digitized == i]
            binned_flux.append(np.mean(subset) if len(subset) > 0 else 1.0)
        
        # --- 5. Save to UNIQUE CSV (UPDATED WITH KID) ---
        label = 1 if disp == "CONFIRMED" else 0
        final_csv_path = os.path.join(TEMP_OUT_DIR, f"res_{kepid_str}_{label}.csv")
        
        # The list now starts with kepid_str, then the label, then the 500 flux bins
        row_data = [kepid_str, label] + binned_flux
        
        pd.DataFrame([row_data]).to_csv(final_csv_path, header=False, index=False)
        print(f"PYTHON: Success for {kepid_str}")

    except Exception as e:
        print(f"PYTHON ERROR: {str(e)}")
        sys.exit(1)
    finally:
        # 6. DELETE THE FILE
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"DELETE ERROR: Still locked! {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    args = parser.parse_args()
    process(args.file)