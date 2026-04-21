"""
Download 10 confirmed-exoplanet + 10 non-planet Kepler light curves
for model validation.

Planet curves are phase-folded + binned to 500 pts so the stacked transit
is clearly visible (matches training data format).
No-planet curves use a 500-pt median-resampled window.
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk

OUT_DIR = os.path.join(os.path.dirname(__file__), "light_curves_csv")
os.makedirs(OUT_DIR, exist_ok=True)
N_BINS = 500

# ── Confirmed exoplanet targets ───────────────────────────────────────
# (KIC id, label, period_days, t0_bkjd, transit_duration_hours)
PLANET_TARGETS = [
    ("KIC 10666592", "Kepler-1b",  2.20473530, 120.0018,  3.97),   # HAT-P-7b
    ("KIC 5780885",  "Kepler-7b",  4.88548965, 135.5779,  5.02),
    ("KIC 11804465", "Kepler-12b",17.85520000, 171.0530, 11.53),
    ("KIC 10874614", "Kepler-6b",  3.23469600, 121.4916,  3.00),
    ("KIC 8191672",  "Kepler-5b",  3.54846553, 121.3397,  3.38),
    ("KIC 6922244",  "Kepler-8b",  3.52254830, 121.1215,  3.21),
    ("KIC 11853905", "Kepler-4b",  3.21346000, 120.4948,  3.86),
    ("KIC 3861595",  "Kepler-17b", 1.48571153, 120.2955,  2.26),
    ("KIC 9941662",  "Kepler-14b", 6.79012000, 124.5540,  3.60),
    ("KIC 11446443", "Kepler-3b",  4.88782450, 122.4807,  3.88),   # HAT-P-11b
]

# ── Non-planet targets ────────────────────────────────────────────────
# Quiet solar-like Kepler stars with no confirmed planets
NO_PLANET_TARGETS = [
    ("KIC 3427720",  "noplanet_kic3427720"),
    ("KIC 4914923",  "noplanet_kic4914923"),
    ("KIC 6116048",  "noplanet_kic6116048"),
    ("KIC 6603624",  "noplanet_kic6603624"),
    ("KIC 7296438",  "noplanet_kic7296438"),
    ("KIC 8006161",  "noplanet_kic8006161"),
    ("KIC 8379927",  "noplanet_kic8379927"),
    ("KIC 9139151",  "noplanet_kic9139151"),
    ("KIC 10644253", "noplanet_kic10644253"),
    ("KIC 12069449", "noplanet_kic12069449"),
]


# ── Helpers ───────────────────────────────────────────────────────────

def bin_to_n(time, flux, n=N_BINS):
    """Bin arrays to exactly n points by block-averaging."""
    idx = np.argsort(time)
    time, flux = time[idx], flux[idx]
    edges = np.linspace(0, len(flux), n + 1).astype(int)
    t_out = np.array([time[edges[i]:edges[i+1]].mean() for i in range(n)])
    f_out = np.array([flux[edges[i]:edges[i+1]].mean() for i in range(n)])
    return t_out, f_out


def fetch_lc(kic_id, quarter=None, exptime="long"):
    """Download and stitch quarters. Suppresses lightkurve Unicode progress."""
    import io as _io, contextlib

    # lightkurve prints Unicode arrows to stdout — capture to avoid cp1252 error
    with contextlib.redirect_stdout(_io.StringIO()), \
         contextlib.redirect_stderr(_io.StringIO()):
        results = lk.search_lightcurve(
            kic_id, mission="Kepler", exptime=exptime, quarter=quarter
        )
        if len(results) == 0:
            return None
        lcs = []
        for r in results[:8]:
            try:
                lc = r.download()
                lc = lc.remove_nans().normalize()
                lcs.append(lc)
            except Exception:
                continue

    if not lcs:
        return None
    from lightkurve import LightCurveCollection
    return LightCurveCollection(lcs).stitch()


def save_planet(kic_id, label, period, t0, dur_h):
    """Phase-fold light curve, bin to 500 pts, save time,flux CSV."""
    fname = f"val_planet_{label.replace(' ', '_')}.csv"
    fpath = os.path.join(OUT_DIR, fname)

    if os.path.exists(fpath):
        print(f"  [skip]  {fname} already exists")
        return fpath, True

    lc = fetch_lc(kic_id)
    if lc is None:
        print(f"  [FAIL]  {kic_id}: no data")
        return None, False

    try:
        # Phase fold: each point mapped to phase in [0, 1)
        phase = ((lc.time.value - t0) % period) / period
        flux  = lc.flux.value

        mask = np.isfinite(phase) & np.isfinite(flux)
        phase, flux = phase[mask], flux[mask]

        # Sigma clip to remove flares/outliers before folding
        med = np.median(flux)
        std = np.std(flux)
        mask2 = np.abs(flux - med) < 5 * std
        phase, flux = phase[mask2], flux[mask2]

        # Bin to 500 phase bins — stacks all transits coherently
        _, f_binned = bin_to_n(phase, flux, N_BINS)

        # Ratio-normalize
        med_b = np.median(f_binned)
        if med_b != 0:
            f_binned = f_binned / med_b

        t_out = np.linspace(0, period, N_BINS)

        df = pd.DataFrame({"time": t_out, "flux": f_binned})
        df.to_csv(fpath, index=False)

        dip = (1 - f_binned.min()) * 100
        n_transits = len(lc) * period / (period * N_BINS)
        print(f"  [OK]    {fname}  ({len(lc)} pts, {int(lc.time.value[-1] - lc.time.value[0]) // int(period)} transits stacked, dip={dip:.3f}%)")
        return fpath, True

    except Exception as e:
        print(f"  [FAIL]  {kic_id}: {e}")
        import traceback; traceback.print_exc()
        return None, False


def save_noplanet(kic_id, label, quarter=3):
    """Download one quarter, bin to 500 pts, save as time,flux CSV."""
    fname = f"val_noplanet_{label}.csv"
    fpath = os.path.join(OUT_DIR, fname)

    if os.path.exists(fpath):
        print(f"  [skip]  {fname} already exists")
        return fpath, True

    lc = fetch_lc(kic_id, quarter=quarter)
    if lc is None:
        print(f"  [FAIL]  {kic_id}: no data")
        return None, False

    try:
        time = lc.time.value
        flux = lc.flux.value
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]

        # Sigma clip positive outliers only
        med = np.median(flux)
        std = np.std(flux)
        mask2 = flux < med + 4 * std
        time, flux = time[mask2], flux[mask2]

        t_out, f_out = bin_to_n(time, flux, N_BINS)

        med_b = np.median(f_out)
        if med_b != 0:
            f_out = f_out / med_b

        df = pd.DataFrame({"time": t_out, "flux": f_out})
        df.to_csv(fpath, index=False)

        dip = (1 - f_out.min()) * 100
        print(f"  [OK]    {fname}  ({len(flux)} pts binned→500, dip={dip:.3f}%)")
        return fpath, True

    except Exception as e:
        print(f"  [FAIL]  {kic_id}: {e}")
        return None, False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("-" * 65)
    print("Downloading & phase-folding Kepler validation light curves")
    print("-" * 65)

    manifest_rows = []

    print("\n-- Confirmed exoplanets (phase-folded) --")
    for kic_id, label, period, t0, dur in PLANET_TARGETS:
        print(f"\n{label} ({kic_id}, P={period:.4f}d)")
        fpath, ok = save_planet(kic_id, label, period, t0, dur)
        if ok:
            manifest_rows.append({
                "file": os.path.basename(fpath),
                "label": label,
                "truth": "planet",
                "kic": kic_id,
                "period_days": period,
            })

    print("\n-- Non-planet stars (binned single quarter) --")
    for kic_id, label in NO_PLANET_TARGETS:
        print(f"\n{label} ({kic_id})")
        fpath, ok = save_noplanet(kic_id, label)
        if ok:
            manifest_rows.append({
                "file": os.path.basename(fpath),
                "label": label,
                "truth": "no_planet",
                "kic": kic_id,
                "period_days": None,
            })

    # ── Summary ─────────────────────────────────────────────────────
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(OUT_DIR, "validation_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    planet_ok    = len(manifest[manifest["truth"] == "planet"])
    noplanet_ok  = len(manifest[manifest["truth"] == "no_planet"])

    print("\n" + "=" * 65)
    print(f"Done: {planet_ok} planet + {noplanet_ok} no-planet files saved")
    print(f"Manifest: {manifest_path}")
    print(manifest[["file", "truth"]].to_string(index=False))
    return manifest


if __name__ == "__main__":
    main()
