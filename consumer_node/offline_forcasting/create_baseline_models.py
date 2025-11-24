"""Create simple baseline joblib models for each topic/horizon.

These baseline models implement a `predict(X)` method that:
- tries to use any `_lag1` feature in `X` (persistence),
- otherwise uses the first numeric feature in `X`,
- otherwise returns NaN.

The script writes files to the repository `artifacts/` directory using the
naming convention expected by the pipeline:
  model_<COLUMN>_H<h>_hgb.joblib

Run this from the repository (or inside the consumer container) to create
placeholder models so the forecaster finds them.
"""

import os
import joblib
import numpy as np
import pandas as pd

from offline_forecasting import OfflineForecaster


class BaselinePersistenceModel:
    """Model that returns lag-1 value when available, otherwise first numeric value.

    This class intentionally keeps a liberal predict signature so it works with
    the forecaster's X (a pandas DataFrame with possibly many lag/rolling cols).
    """

    def __init__(self, topic_key):
        self.topic_key = topic_key
        # leave feature_names_in_ absent to allow flexible column matching
        self.feature_names_in_ = None

    def predict(self, X):
        # X may be a DataFrame with one or more rows; return an array
        try:
            if X is None:
                return np.array([np.nan])
            if isinstance(X, (pd.DataFrame,)) and len(X) >= 1:
                row = X.iloc[0]
                # prefer any column that contains '_lag1'
                for col in X.columns:
                    if isinstance(col, str) and (col.endswith('_lag1') or '_lag1' in col):
                        val = row[col]
                        try:
                            return np.array([float(val)])
                        except Exception:
                            return np.array([np.nan])
                # otherwise, prefer first numeric column
                for col in X.columns:
                    try:
                        v = float(row[col])
                        return np.array([v])
                    except Exception:
                        continue
                return np.array([np.nan])
            else:
                return np.array([np.nan])
        except Exception:
            return np.array([np.nan])


def main():
    # determine artifacts dir using same logic as OfflineForecaster
    current_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    # Also attempt to create/write into root `/artifacts` which some containers
    # mount as the artifacts volume. This makes the script work both locally
    # and inside containers that expect files under `/artifacts`.
    root_artifacts = '/artifacts'
    try:
        os.makedirs(root_artifacts, exist_ok=True)
        write_root = True
    except Exception:
        write_root = False

    horizons = [1, 2, 3]
    created = []
    missing = []

    for topic_key, col_name in OfflineForecaster.TOPIC_TO_COLUMN.items():
        safe_col = col_name
        for h in horizons:
            fname = f"model_{safe_col}_H{h}_hgb.joblib"
            path = os.path.join(artifacts_dir, fname)
            try:
                model = BaselinePersistenceModel(topic_key)
                joblib.dump(model, path)
                created.append(path)
                # also write into /artifacts if possible
                if write_root:
                    try:
                        joblib.dump(model, os.path.join(root_artifacts, fname))
                    except Exception:
                        # non-fatal: keep going
                        pass
            except Exception as e:
                missing.append((path, str(e)))

    print("Baseline model creation complete.")
    if created:
        print(f"Created {len(created)} model files (examples):\n  {created[:5]}")
    if missing:
        print("Failures:\n" + "\n".join(f"{p}: {e}" for p, e in missing))


if __name__ == '__main__':
    main()
