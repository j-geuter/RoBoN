from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import os
import glob
import math
import ast
import json
import numpy as np

try:
    from zoneinfo import ZoneInfo  # py>=3.9

    _TZ = ZoneInfo("America/New_York")
except Exception:
    _TZ = None


def timestamp(fmt="%Y%m%d_%H%M%S"):
    now = datetime.now(_TZ) if _TZ else datetime.now()
    return now.strftime(fmt)


def timestamped_path(path: str, sep: str = "_", ts: str | None = None) -> str:
    """Insert timestamp before extension: plot.png -> plot_YYYYmmdd_HHMMSS.png"""
    p = Path(path)
    ts = ts or timestamp()
    return str(p.with_name(f"{p.stem}{sep}{ts}{p.suffix}"))

# ------------------ Utilities ------------------
def infer_task_from_dataset(name: str) -> str:
    name = name.lower()
    if name in {"humaneval", "mbpp"}:
        return "code"
    return "math"


def parse_list(s: str, conv=str):
    return [conv(x.strip()) for x in s.split(",") if x.strip()]


def safe_softmax_beta(scores: np.ndarray, beta: float) -> np.ndarray:
    if scores.size == 0:
        return np.zeros_like(scores, dtype=np.float64)
    m = float(np.max(scores))
    z = beta * (scores - m)
    ez = np.exp(z)
    denom = max(float(ez.sum()), 1e-10)
    return ez / denom


def expected_metric(rewards: np.ndarray, outcomes: np.ndarray, beta: float) -> float:
    w = safe_softmax_beta(rewards, beta)
    return float(np.dot(w, outcomes))


def ci95(x: np.ndarray) -> Tuple[float, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    se = sd / max(math.sqrt(x.size), 1e-10)
    return m, 1.96 * se


def count_asserts(tests_src: str) -> int:
    try:
        t = ast.parse(tests_src)
    except Exception:
        return 0

    class V(ast.NodeVisitor):
        def __init__(self):
            self.n = 0

        def visit_Assert(self, node):
            self.n += 1

    v = V()
    v.visit(t)
    return v.n


def _mean_ci_auto(vals: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = float(arr.mean())
    # if binary -> Wilson; else -> normal approx on mean
    is_binary = np.all((arr == 0.0) | (arr == 1.0))
    if is_binary:
        z = 1
        denom = 1.0 + z * z / n
        center = (m + z * z / (2 * n)) / denom
        half = z * np.sqrt((m * (1 - m) + z * z / (4 * n)) / n) / denom
        lo, hi = max(0.0, center - half), min(1.0, center + half)
        return (m, lo, hi)
    else:
        # normal CI on mean of bounded [0,1] values
        z = 1
        se = float(arr.std(ddof=1) / np.sqrt(max(n, 1)))
        lo, hi = m - z * se, m + z * se
        lo, hi = float(max(0.0, lo)), float(min(1.0, hi))
        return (m, lo, hi)


def iter_records(run_dir: str, task: str, dataset: str):
    pat = os.path.join(run_dir, task, dataset, "*", "*.jsonl")
    for path in glob.glob(pat):
        model_alias = os.path.basename(os.path.dirname(path))
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield model_alias, json.loads(line)


def to_float(v):
    try:
        return float(v)
    except Exception:
        return None
    

def binary_mean_and_wilson(acc_list: List[float], **kwargs):
    """acc_list entries are 0/1 (math accuracy or code pass@1)."""
    arr = np.asarray(acc_list, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = float(arr.mean())
    # Wilson score interval
    from math import sqrt

    z = 1
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))