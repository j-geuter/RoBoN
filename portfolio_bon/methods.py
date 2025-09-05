# methods.py
# Portfolio methods for combining models under a total budget n.
# Each method returns per-prompt expected metric values using soft selection by rewards.
#
# API expected by test_bon.py:
#   METHODS_REGISTRY[name] = function
#   function(by_prompt: Dict[prompt_id, {"models": {model: {"rm": np.ndarray, "y": np.ndarray}}, "extra": {...}}],
#            n: int, beta: float, task: str) -> Dict[prompt_id, float]
#
# Notes:
# - For each prompt, we build a union of the first a_m responses from each model m,
#   according to the method's allocation (sum_m a_m <= n).
# - We compute expected metric = sum softmax(beta*(r - max r)) * y over the union.
# - Missing entries in first a_m for any model cause that prompt to skip those items (we only
#   include responses with both reward and outcome present); if the union is empty, we return None.

from typing import Dict, List, Tuple
import numpy as np
import random
from collections import Counter
import re
from time import time



def _expected_metric(
        chosen_triplets, 
        beta: float, 
        alpha: float = 0.7
    ) -> float:
    """
    Soft-BoN expected metric.
    chosen_triplets: list of (reward, y_unused, answer_string).
      - reward r_i: float (can be raw RM score; we normalize via ECDF locally)
      - y_unused   : ignored (kept for tuple compatibility)
      - answer     : model's predicted answer; used for self-consistency agreement
    beta: temperature for soft-BoN weights on rewards
    alpha: mix weight in [0,1] for p_hat = alpha * p_rm + (1-alpha) * agreement

    Returns F_beta = sum_i w_i * p_hat_i, with w_i ∝ exp(beta * (r_i - max r)).
    """
    if not chosen_triplets:
        return float("nan")
    # Extract rewards + answers; drop non-finite rewards
    rewards, answers = [], []
    for r, _y, a in chosen_triplets:
        try:
            rf = float(r)
        except Exception:
            rf = np.nan
        if not np.isfinite(rf):
            continue
        rewards.append(rf)
        answers.append("" if a is None else str(a))

    if len(rewards) == 0:
        return float("nan")

    r = np.asarray(rewards, dtype=np.float64)

    # Soft-BoN weights over rewards (stable)
    m = float(np.max(r))
    z = beta * (r - m)
    z = z - np.max(z)  # prevent overflow
    ez = np.exp(z)
    denom = max(float(ez.sum()), 1e-10)
    w = ez / denom  # weights sum to 1

    # ECDF (mid-rank) on rewards within the chosen set -> p_rm in [0,1]
    # r_sorted = np.sort(r, kind="mergesort")
    # left  = np.searchsorted(r_sorted, r, side="left")
    # right = np.searchsorted(r_sorted, r, side="right")
    # p_rm = (left + right) / (2.0 * max(len(r), 1e-10))
    p_rm = r
    # If all answers empty after canon, drop the agreement term
    if len(set(answers)) == 1 and answers[0] == "":
        agree = np.zeros_like(p_rm, dtype=np.float64)
        mix_alpha = 1.0
    else:
        cnt = Counter(answers)
        N = float(len(answers))
        agree = np.array([cnt[a] / max(N, 1e-10) for a in answers], dtype=np.float64)
        mix_alpha = float(alpha)
    # Predicted success prob
    p_hat = mix_alpha * p_rm + (1.0 - mix_alpha) * agree
    # Clamp numeric noise
    p_hat = np.clip(p_hat, 0.0, 1.0)

    return float(np.dot(w, p_hat))

def _expected_metric_union(
        chosen: List[tuple], 
        beta: float
    ) -> float:
    if not chosen:
        return float("nan")
    r = np.array([c[0] for c in chosen], dtype=np.float64)
    y = np.array([c[1] for c in chosen], dtype=np.float64)
    m = float(np.max(r))
    z = beta * (r - m)
    ez = np.exp(z - z.max())
    denom = max(float(ez.sum()), 1e-10)
    w = ez / denom
    return float(np.dot(w, y))

def _wrap_output(
        models: List[str], 
        chosen_ids: List[Tuple[str, int]], 
        entry: Dict, 
        beta: float,
        extra_metrics: Dict[str, float] | None = None
    ) -> Dict:
    """Compute expected metric from chosen ids and return rich dict."""
    if not chosen_ids:
        return {
            "expected": float("nan"),
            "ids": [],
            "count_by_model": {m: 0 for m in models},
            "metrics": extra_metrics or {},
        }
    chosen_pairs = []
    for m, i in chosen_ids:
        try:
            r = float(entry["models"][m]["rm"][i])
            y = float(entry["models"][m]["y"][i])
        except Exception:
            continue
        chosen_pairs.append((r, y))
    val = _expected_metric_union(chosen_pairs, beta) if chosen_pairs else float("nan")
    cnt = Counter(m for m, _ in chosen_ids)
    return {
        "expected": float(val) if np.isfinite(val) else float("nan"),
        "ids": chosen_ids,
        "count_by_model": {m: int(cnt.get(m, 0)) for m in models},
        "metrics": extra_metrics or {},
    }

def equal_method(by_prompt: Dict, n: int, **kwargs):
    """
    Distribute exactly n samples across k models as evenly as possible and RETURN INDICES:
      out[pid] -> List[(model_alias, idx)].

    - Uses the first indices (head) per model.
    - Respects per-model capacity; leftover quota is reassigned round-robin to models with remaining items.
    """
    out: Dict[str, List[Tuple[str, int]]] = {}

    for pid, entry in by_prompt.items():
        models = list(entry["models"].keys())
        k = len(models)
        if k == 0 or n <= 0:
            out[pid] = []
            continue

        # Per-model capacities
        L = {m: int(getattr(entry["models"][m]["rm"], "size",
                             len(entry["models"][m]["rm"]))) for m in models}

        # Initial equal allocation (base + first `rem` models get +1)
        base, rem = divmod(n, k)
        alloc = {m: base for m in models}
        for i in range(rem):
            alloc[models[i % k]] += 1

        # Clamp to capacity
        take = {m: min(alloc[m], L[m]) for m in models}
        chosen_total = sum(take.values())

        # If underfilled due to caps, top-up round-robin where capacity remains
        if chosen_total < n:
            need = n - chosen_total
            idx = 0
            while need > 0:
                m = models[idx % k]
                if take[m] < L[m]:
                    take[m] += 1
                    need -= 1
                idx += 1
                # break if no capacity anywhere
                if all(take[x] >= L[x] for x in models):
                    break

        # Materialize indices (head items)
        chosen_ids: List[Tuple[str, int]] = []
        for m in models:
            t = int(take[m])
            for i in range(t):
                chosen_ids.append((m, i))

        # Truncate in the unlikely event we exceeded n due to rounding
        if len(chosen_ids) > n:
            chosen_ids = chosen_ids[:n]

        out[pid] = chosen_ids

    return out



def routed_online_best_of_n(
        by_prompt: Dict, 
        n: int, 
        beta: float, 
        plot: bool = False,
        **kwargs
    ):
    entry = next(iter(by_prompt.values()))
    models = list(entry["models"].keys())
    model_counts = {m: 0 for m in models}

    out: Dict[str, List[Tuple[str, int]]] = {}
    for pid, entry in by_prompt.items():
        models = list(entry["models"].keys())
        k = len(models)
        if n <= 0 or k == 0:
            out[pid] = []
            continue

        # Per-model arrays
        rm = {m: entry["models"][m]["rm"] for m in models}
        y = {m: entry["models"][m]["y"] for m in models}  # kept for tuple shape; scorer ignores y
        ans = {m: entry["models"][m]["answers"] for m in models}
        L = {m: rm[m].size for m in models}

        # --- Special case: n == 1 -> take one random available head item ---
        if n == 1:
            avail = [m for m in models if L[m] >= 1]
            if not avail:
                out[pid] = []
                continue
            m0 = random.choice(avail)
            out[pid] = [(m0, 0)]
            continue

        # State
        alloc = {m: 0 for m in models}
        chosen_triplets: List[tuple] = []  # (r, y, answer) (y ignored by scorer)
        chosen_ids: List[Tuple[str, int]] = []
        steps = max(0, n - k + 1)
        for _ in range(steps):
            avail = [m for m in models if alloc[m] < L[m]]
            if not avail:
                break

            # Choose model maximizing F_beta(chosen ∪ {next_m}) under agreement scorer
            best_m, best_val = None, -1e18
            for m in avail:
                a = alloc[m]
                cand = chosen_triplets + [(rm[m][a], y[m][a], ans[m][a])]
                val = _expected_metric(cand, beta)
                if np.isnan(val):
                    continue
                if val > best_val:
                    best_val, best_m = val, m
            if best_m is None:
                break

            # Commit selection
            a = alloc[best_m]
            chosen_triplets.append((rm[best_m][a], y[best_m][a], ans[best_m][a]))
            chosen_ids.append((best_m, a))
            alloc[best_m] += 1
        
            out[pid] = chosen_ids
    if plot:
        for m in models:
            model_counts[m] /= len(by_prompt)
        
        return out, model_counts
    else:
        return out


METHODS_REGISTRY = {
    "equal": equal_method,
    "RoBoN": routed_online_best_of_n,
}
