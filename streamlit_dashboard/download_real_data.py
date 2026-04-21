"""
Download 10 real Kepler / K2 light curves for testing the dashboard.

Uses the `lightkurve` package (pip install lightkurve) to pull from the
MAST archive. Each target is chosen for a specific property so you can
sanity-check the pipeline across a range of transit strengths.

Usage:
    pip install lightkurve
    python download_real_data.py

Saves files into  light_curves_csv/real_<nickname>.csv   as (time, flux).
"""

import os
import sys
import time as _time

try:
    import lightkurve as lk
except ImportError:
    print("Missing dependency. Run:  pip install lightkurve")
    sys.exit(1)

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "light_curves_csv")
os.makedirs(OUT, exist_ok=True)

# (nickname, target_id, mission, quarter/campaign, expected disposition)
TARGETS = [
    # ── Confirmed planets — clear transits ───────────────────────
    ("kepler10b",    "KIC 11904151",  "Kepler", 3,
        "Kepler-10 b  · rocky, P=0.837d, depth ~250 ppm (subtle)"),
    ("kepler7b",     "KIC 5780885",   "Kepler", 5,
        "Kepler-7 b   · hot Jupiter, P=4.89d, ~6700 ppm (clear)"),
    ("tres2b",       "KIC 11446443",  "Kepler", 5,
        "TrES-2 b     · 'darkest planet', P=2.47d, ~1700 ppm"),
    ("hatp7b",       "KIC 10666592",  "Kepler", 5,
        "HAT-P-7 b    · hot Jupiter, P=2.20d, ~6600 ppm"),
    ("kepler8b",     "KIC 6922244",   "Kepler", 5,
        "Kepler-8 b   · hot Jupiter, P=3.52d, ~9600 ppm"),
    ("kepler22b",    "KIC 10593626",  "Kepler", 5,
        "Kepler-22 b  · habitable zone, P=289.86d, ~500 ppm"),
    ("kepler16b",    "KIC 12644769",  "Kepler", 5,
        "Kepler-16 b  · circumbinary, eclipses + transits"),
    # ── Multi-planet system ──────────────────────────────────────
    ("kepler9b",     "KIC 3323887",   "Kepler", 5,
        "Kepler-9 b/c · multi-planet, TTV-famous"),
    # ── Anomalous / non-planet ───────────────────────────────────
    ("tabbysstar",   "KIC 8462852",   "Kepler", 16,
        "Boyajian's star · irregular deep dips (NOT a planet)"),
    # ── Quiet reference star (should have NO transits) ───────────
    ("quietstar",    "KIC 7940546",   "Kepler", 5,
        "KIC 7940546  · quiet star, no known transits"),
]


def fetch(nickname, target, mission, quarter, description):
    out_path = os.path.join(OUT, f"real_{nickname}.csv")
    if os.path.exists(out_path):
        print(f"  [skip] {nickname} already exists -> {out_path}")
        return True

    print(f"  downloading {nickname}  ({target}, quarter {quarter})...")
    try:
        search = lk.search_lightcurve(target, mission=mission, quarter=quarter)
        if len(search) == 0:
            # Try without quarter restriction
            search = lk.search_lightcurve(target, mission=mission)
        if len(search) == 0:
            print(f"  [FAIL] no data found for {target}")
            return False
        lc = search[0].download()
        if lc is None:
            print(f"  [FAIL] download returned None for {target}")
            return False
        lc = lc.remove_nans().normalize()
        t = np.asarray(lc.time.value, dtype=float)
        f = np.asarray(lc.flux.value,  dtype=float)
        ok = np.isfinite(t) & np.isfinite(f)
        df = pd.DataFrame({"time": t[ok], "flux": f[ok]})
        df.to_csv(out_path, index=False)
        print(f"  [OK]   {len(df)} points -> real_{nickname}.csv")
        print(f"         {description}")
        return True
    except Exception as e:
        print(f"  [FAIL] {nickname}: {e}")
        return False


if __name__ == "__main__":
    print(f"Downloading {len(TARGETS)} real Kepler light curves into:")
    print(f"  {OUT}\n")
    ok = 0
    for nickname, target, mission, quarter, description in TARGETS:
        if fetch(nickname, target, mission, quarter, description):
            ok += 1
        _time.sleep(1)  # be polite to MAST
    print(f"\nDone. {ok}/{len(TARGETS)} files saved.")
    print("\nSuggested test order in the dashboard:")
    print("  1. real_kepler7b.csv    (deep, clear)")
    print("  2. real_hatp7b.csv      (deep hot Jupiter)")
    print("  3. real_tres2b.csv      (moderate dip)")
    print("  4. real_kepler10b.csv   (very subtle, good stress test)")
    print("  5. real_kepler22b.csv   (habitable zone, single transit in quarter)")
    print("  6. real_kepler16b.csv   (eclipsing binary + planet)")
    print("  7. real_kepler9b.csv    (multi-planet)")
    print("  8. real_kepler8b.csv    (another hot Jupiter)")
    print("  9. real_tabbysstar.csv  (weird non-planet dips)")
    print(" 10. real_quietstar.csv   (should say: no planet)")
