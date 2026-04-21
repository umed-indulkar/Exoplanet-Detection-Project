"""
Download and phase-fold a Kepler light curve for a given KIC ID + orbital parameters.
Used when the user uploads a KOI metadata table.
"""

import contextlib
import io as _io
import numpy as np

_lk = None  # lazy import


def _lightkurve():
    global _lk
    if _lk is None:
        import lightkurve as lk
        _lk = lk
    return _lk


def fetch_folded_flux(kepid: int, period: float, t0: float,
                      n_bins: int = 500, max_quarters: int = 8):
    """Download Kepler long-cadence data, phase-fold, bin to n_bins points.

    Parameters
    ----------
    kepid       : Kepler Input Catalog ID
    period      : orbital period in days
    t0          : transit epoch in BKJD
    n_bins      : output size (default 500 to match model input)
    max_quarters: maximum quarters to download (more = cleaner transit)

    Returns
    -------
    flux_binned : ndarray of shape (n_bins,), ratio-normalised (~1.0)
    n_pts       : total raw points used
    n_transits  : approx number of stacked transits
    error       : None on success, string message on failure
    """
    lk = _lightkurve()

    try:
        with contextlib.redirect_stdout(_io.StringIO()), \
             contextlib.redirect_stderr(_io.StringIO()):
            results = lk.search_lightcurve(
                f"KIC {kepid}", mission="Kepler", exptime="long"
            )

        if len(results) == 0:
            return None, 0, 0, f"No Kepler data found for KIC {kepid}"

        lcs = []
        with contextlib.redirect_stdout(_io.StringIO()), \
             contextlib.redirect_stderr(_io.StringIO()):
            for r in results[:max_quarters]:
                try:
                    lc = r.download()
                    lc = lc.remove_nans().normalize()
                    lcs.append(lc)
                except Exception:
                    continue

        if not lcs:
            return None, 0, 0, f"Failed to download any quarters for KIC {kepid}"

        from lightkurve import LightCurveCollection
        stitched = LightCurveCollection(lcs).stitch()

        time = stitched.time.value.astype(np.float64)
        flux = stitched.flux.value.astype(np.float64)

        # Remove NaN / inf
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]

        # Sigma-clip positive outliers (flares) only
        med = np.median(flux)
        std = np.std(flux)
        good = flux < med + 5 * std
        time, flux = time[good], flux[good]

        n_pts = len(flux)
        if n_pts < 50:
            return None, 0, 0, f"Too few valid points ({n_pts}) for KIC {kepid}"

        # Phase fold
        phase = ((time - t0) % period) / period
        phase = np.where(phase < 0, phase + 1, phase)  # ensure [0,1)

        # Sort by phase
        order = np.argsort(phase)
        phase = phase[order]
        flux  = flux[order]

        # Bin to n_bins equal-width phase bins
        edges  = np.linspace(0.0, 1.0, n_bins + 1)
        binned = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask_b = (phase >= edges[i]) & (phase < edges[i + 1])
            if mask_b.any():
                binned[i] = np.median(flux[mask_b])

        # Interpolate over empty bins
        if np.isnan(binned).any():
            x_all  = np.arange(n_bins)
            x_good = x_all[~np.isnan(binned)]
            if len(x_good) > 1:
                binned = np.interp(x_all, x_good, binned[x_good])
            else:
                binned = np.where(np.isnan(binned), 1.0, binned)

        # Ratio-normalise
        med_b = float(np.median(binned))
        if med_b != 0:
            binned = binned / med_b

        n_transits = max(1, int(np.round((time[-1] - time[0]) / period)))
        return binned.astype(np.float32), n_pts, n_transits, None

    except Exception as exc:
        return None, 0, 0, str(exc)
