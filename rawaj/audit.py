"""Audit module for rawaj transfer extract data."""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _is_blank_or_missing(val: Any) -> bool:
    """Check if a value is missing (NaN, None, or empty/whitespace string)."""
    if val is None:
        return True
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def _get_missing_mask(s: pd.Series) -> pd.Series:
    """Return boolean Series where True indicates NaN/None or blank/whitespace string."""
    isna = s.isna()
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return isna
    # For object, string, categorical, or mixed types
    blank_mask = s.map(lambda x: isinstance(x, str) and x.strip() == "")
    return isna | blank_mask


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict[str, Any]:
    """Audit a transfer dataset for data quality, leakage, and integrity issues."""
    for col in (target, id_col, time_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe.")

    n_rows = int(len(df))

    # Missing fractions per column
    missing = {}
    missing_masks = {}
    for col in df.columns:
        mask = _get_missing_mask(df[col])
        missing_masks[col] = mask
        missing[col] = float(mask.mean()) if n_rows > 0 else 0.0

    # Duplicate IDs (non-missing only, sorted)
    id_series = df[id_col][~missing_masks[id_col]]
    id_counts = id_series.value_counts()
    dup_ids_raw = id_counts[id_counts > 1].index.tolist()
    try:
        duplicate_ids = sorted(dup_ids_raw)
    except TypeError:
        duplicate_ids = sorted(dup_ids_raw, key=lambda x: (str(type(x)), str(x)))

    # Duplicate rows (rows that exactly repeat an earlier row)
    duplicate_rows = int(df.duplicated().sum())

    # Constant columns: columns other than target with at most one distinct non-missing value
    constant_columns = []
    for col in df.columns:
        if col == target:
            continue
        non_missing = df[col][~missing_masks[col]]
        if non_missing.nunique() <= 1:
            constant_columns.append(col)
    constant_columns.sort()

    # Unlabelled rows (target is missing)
    target_missing = missing_masks[target]
    unlabelled_rows = int(target_missing.sum())

    # Positive rate on labelled rows only
    labelled_target = df[target][~target_missing]
    if len(labelled_target) == 0:
        positive_rate = 0.0
    else:
        target_num = pd.to_numeric(labelled_target, errors="coerce")
        pos_mask = (target_num == 1) | (labelled_target == 1) | (labelled_target == True) | (labelled_target == "1")
        positive_rate = float(pos_mask.mean())

    # Timestamps audit
    time_series = df[time_col][~missing_masks[time_col]]
    parsed_timestamps = pd.to_datetime(time_series, errors="coerce")
    bad_timestamps = int(parsed_timestamps.isna().sum())

    valid_parsed = parsed_timestamps.dropna()
    if valid_parsed.empty:
        time_range = [None, None]
    else:
        time_range = [valid_parsed.min().isoformat(), valid_parsed.max().isoformat()]

    # Leakage suspects
    leakage_suspects = []
    labelled_df = df[~target_missing]

    if len(labelled_df) >= 20:
        y_labelled = labelled_df[target]
        y_labelled_num = pd.to_numeric(y_labelled, errors="coerce")
        y_labelled_num = y_labelled_num.fillna(
            y_labelled.map(lambda x: 1 if x in (True, "1", "true", "True") else (0 if x in (False, "0", "false", "False") else np.nan))
        )
        unique_classes = set(y_labelled_num.dropna().unique())

        if {0, 1}.issubset(unique_classes):
            for col in df.columns:
                if col in (target, id_col, time_col):
                    continue

                col_labelled = labelled_df[col]
                mask_labelled = missing_masks[col].loc[col_labelled.index]
                col_valid = col_labelled[~mask_labelled]

                # A column with a single distinct value is never a suspect (missing counts as a value)
                if mask_labelled.all() or (not mask_labelled.any() and col_valid.nunique() <= 1):
                    continue

                is_suspect = False

                # (a) Check numeric or boolean
                is_bool = pd.api.types.is_bool_dtype(col_labelled)
                is_num = pd.api.types.is_numeric_dtype(col_labelled) and not is_bool

                col_num_vals = None
                if is_bool:
                    col_num_vals = col_valid.astype(float)
                elif is_num:
                    col_num_vals = pd.to_numeric(col_valid, errors="coerce")
                else:
                    if len(col_valid) > 0 and col_valid.map(lambda x: isinstance(x, (bool, np.bool_))).all():
                        is_bool = True
                        col_num_vals = col_valid.astype(float)
                    else:
                        coerced = pd.to_numeric(col_valid, errors="coerce")
                        if coerced.notna().all():
                            is_num = True
                            col_num_vals = coerced

                if (is_num or is_bool) and col_num_vals is not None:
                    if len(col_valid) >= 20:
                        y_valid = y_labelled_num.loc[col_valid.index]
                        if len(set(y_valid.unique())) >= 2:
                            try:
                                auc = roc_auc_score(y_valid, col_num_vals)
                                if auc >= 0.98 or auc <= 0.02:
                                    is_suspect = True
                            except Exception:
                                pass

                # (b) Missingness indicator check (1 = missing) against label on labelled rows
                if not is_suspect:
                    m = mask_labelled.astype(int)
                    if m.nunique() >= 2:
                        try:
                            auc_m = roc_auc_score(y_labelled_num, m)
                            if auc_m >= 0.98 or auc_m <= 0.02:
                                is_suspect = True
                        except Exception:
                            pass

                if is_suspect:
                    leakage_suspects.append(col)

    leakage_suspects.sort()

    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }
