"""Metrics module for evaluating threshold policies on fraud detection scores."""

from __future__ import annotations

import math
from typing import Any
import numpy as np


def _validate_inputs(y_true: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate y_true and scores inputs according to strict assignment rules."""
    try:
        y_arr = np.asarray(y_true)
        s_arr = np.asarray(scores, dtype=float)
    except Exception as e:
        raise ValueError(f"Could not convert inputs to arrays: {e}") from e

    if y_arr.ndim != 1 or s_arr.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional.")

    if len(y_arr) != len(s_arr):
        raise ValueError(f"Lengths differ: len(y_true)={len(y_arr)}, len(scores)={len(s_arr)}")

    if len(y_arr) == 0:
        raise ValueError("Input arrays must not be empty.")

    # Check for NaN in scores
    if np.isnan(s_arr).any():
        raise ValueError("Scores array contains NaN values.")

    # Check labels: every element must be 0 or 1
    for val in y_arr:
        if isinstance(val, (bool, np.bool_)):
            continue
        if isinstance(val, (int, np.integer, float, np.floating)):
            if np.isnan(val) or val not in (0, 1, 0.0, 1.0):
                raise ValueError(f"Label {val} is not 0 or 1.")
        else:
            raise ValueError(f"Label {val} is not 0 or 1.")

    y_int = y_arr.astype(int)
    return y_int, s_arr


def confusion_at(y_true: Any, scores: Any, threshold: float) -> dict[str, int]:
    """Compute confusion matrix dict at the given score threshold."""
    y_arr, s_arr = _validate_inputs(y_true, scores)
    flagged = s_arr >= threshold

    tp = int(np.sum(flagged & (y_arr == 1)))
    fp = int(np.sum(flagged & (y_arr == 0)))
    tn = int(np.sum((~flagged) & (y_arr == 0)))
    fn = int(np.sum((~flagged) & (y_arr == 1)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at(y_true: Any, scores: Any, threshold: float) -> tuple[float, float]:
    """Compute (precision, recall) at the given threshold."""
    c = confusion_at(y_true, scores, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]

    precision = 0.0 if (tp + fp) == 0 else float(tp / (tp + fp))
    recall = 0.0 if (tp + fn) == 0 else float(tp / (tp + fn))

    return precision, recall


def precision_at_k(y_true: Any, scores: Any, k: int) -> float:
    """Compute precision among the top-k highest-scoring predictions."""
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer > 0.")

    y_arr, s_arr = _validate_inputs(y_true, scores)
    n = len(y_arr)
    effective_k = min(int(k), n)

    # Rank by score descending; stable sort preserves original index order on ties
    indices = sorted(range(n), key=lambda i: s_arr[i], reverse=True)
    top_k_indices = indices[:effective_k]

    positives = sum(1 for i in top_k_indices if y_arr[i] == 1)
    return float(positives / effective_k)


def expected_cost(y_true: Any, scores: Any, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """Compute expected business cost of false positives and false negatives."""
    c = confusion_at(y_true, scores, threshold)
    return float(c["fp"] * cost_fp + c["fn"] * cost_fn)


def best_threshold(y_true: Any, scores: Any, *, cost_fp: float, cost_fn: float) -> float:
    """Find the threshold that minimizes expected cost.
    
    Candidates are every distinct score plus math.inf.
    On tie, returns the highest threshold (fewer flags for the same cost).
    """
    y_arr, s_arr = _validate_inputs(y_true, scores)
    unique_scores = sorted(list(set(s_arr.tolist())))
    candidates = unique_scores + [math.inf]

    def sort_key(t: float) -> tuple[float, float]:
        cost = expected_cost(y_arr, s_arr, t, cost_fp=cost_fp, cost_fn=cost_fn)
        neg_t = -float("inf") if math.isinf(t) else -t
        return (cost, neg_t)

    return min(candidates, key=sort_key)


def threshold_for_precision(y_true: Any, scores: Any, min_precision: float) -> float | None:
    """Find the lowest threshold among distinct scores whose precision is >= min_precision.
    
    Returns None if no threshold reaches min_precision.
    """
    y_arr, s_arr = _validate_inputs(y_true, scores)
    unique_scores = sorted(list(set(s_arr.tolist())))

    valid_thresholds = []
    for t in unique_scores:
        prec, _ = precision_recall_at(y_arr, s_arr, t)
        if prec >= min_precision or math.isclose(prec, min_precision, rel_tol=1e-9, abs_tol=1e-9):
            valid_thresholds.append(t)

    if not valid_thresholds:
        return None

    return min(valid_thresholds)
