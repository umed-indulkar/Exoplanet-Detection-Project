import numpy as np
import pandas as pd

try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters
    _TSFRESH_AVAILABLE = True
except Exception:
    _TSFRESH_AVAILABLE = False

from ..core.data_loader import LightCurve


def _to_long_df(lc: LightCurve, kind: str = "flux") -> pd.DataFrame:
    df = pd.DataFrame({"id": 0, "time": lc.time, kind: lc.flux})
    return df


def extract_tsfresh_features(lc: LightCurve, *, default_fc_parameters: dict | None = None,
                              disable_progressbar: bool = True, n_jobs: int = 0) -> pd.DataFrame:
    if not _TSFRESH_AVAILABLE:
        raise ImportError("tsfresh is not installed. Install with: pip install tsfresh statsmodels")

    if default_fc_parameters is None:
        default_fc_parameters = EfficientFCParameters()

    df = _to_long_df(lc)
    # tsfresh expects a long format with value column name
    df_long = df.rename(columns={"time": "time", "flux": "value"})
    features = extract_features(
        df_long,
        column_id="id",
        column_sort="time",
        default_fc_parameters=default_fc_parameters,
        disable_progressbar=disable_progressbar,
        n_jobs=n_jobs,
    )
    features.reset_index(drop=True, inplace=True)
    return features
