#!/usr/bin/env python3
# test_bon.py
# Multi-dataset soft Best-of-n figure:
# - rows = betas
# - cols = datasets
# Lines per subplot:
#   * one per model (selection by reward, metric = accuracy (math) or #passed tests (code))
#   * "average" = average of per-model curves (prompt-wise mean across selected models)
#   * one per requested portfolio method (equal, RoBoN, ...), implemented in methods.py
#
# Expected JSONL per record:
#   math:  {"id", "question", "responses":[{"rm_score":float, "correct":bool/int, ...}, ...]}

import os, argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

from utils import timestamped_path, infer_task_from_dataset, to_float, count_asserts, iter_records, _mean_ci_auto, parse_list, binary_mean_and_wilson
from methods import METHODS_REGISTRY  # dict: name -> fn

# ------------------ Defaults ------------------
DEFAULT_MODELS = [
    "deepseek-coder-6.7b",
    "llama-3.1-8b",
    "qwen-coder-7b",
    "qwen-math-7b",
]
DEFAULT_DATASETS = ["MATH500", "OlympiadBench", "MinervaMath"]
DEFAULT_BETAS = [100000]

def pick_argmax_reward(entry, chosen_ids: List[Tuple[str, int]], **kwargs) -> Tuple[float, float]:
    """Return (y, rm) of the single chosen response: the argmax-reward inside the union."""
    if not chosen_ids:
        return (np.nan, np.nan)
    r = []
    y = []
    for m, i in chosen_ids:
        r.append(entry["models"][m]["rm"][i])
        y.append(entry["models"][m]["y"][i])  # math: 0/1; code: fraction or bool
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    j = int(np.nanargmax(r))
    return (float(y[j]), float(r[j]))


def pick_sbon_reward(
    entry: Dict,
    chosen_ids: List[Tuple[str, int]],
    beta: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Sample *one* response from the chosen union according to soft BoN over rewards:
        w_i ∝ exp(beta * (r_i - max_j r_j))
    Returns: (y_pick, r_pick). If no valid candidate, returns (nan, nan).

    - entry["models"][m]["rm"][i] : reward (float)
    - entry["models"][m]["y"][i]  : ground-truth metric for eval (math: 0/1; code: fraction/tests)
    - chosen_ids: list of (model_alias, idx) selected by the method.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not chosen_ids:
        return (float("nan"), float("nan"))

    r_list, y_list = [], []
    for m, i in chosen_ids:
        try:
            r = float(entry["models"][m]["rm"][i])
            y = float(entry["models"][m]["y"][i])
        except Exception:
            r, y = np.nan, np.nan
        r_list.append(r)
        y_list.append(y)

    r = np.asarray(r_list, dtype=float)
    y = np.asarray(y_list, dtype=float)

    # keep only finite rewards; if none finite, bail
    mask = np.isfinite(r)
    if not np.any(mask):
        return (float("nan"), float("nan"))
    r = r[mask]
    y = y[mask]

    # softmax over rewards (stable): w ∝ exp(beta * (r - r_max))
    r_max = float(np.max(r))
    z = beta * (r - r_max)
    z = z - np.max(z)  # extra stabilization
    ez = np.exp(z)
    denom = float(ez.sum())
    if denom <= 1e-10 or not np.isfinite(denom):
        # fallback uniform over valid candidates
        j = int(rng.integers(len(r)))
    else:
        p = ez / denom
        # numerical guard to sum to 1
        p = p / max(float(p.sum()), 1e-10)
        j = int(rng.choice(len(r), p=p))

    return (float(y[j]), float(r[j]))


# ------------------ Data aggregation ------------------
# We build a structure per dataset:
# prompts[pid]["models"][model] = {"rm": np.array, "y": np.array} where
#   - math: y = 0/1 correctness per response


def load_dataset_runs(
    runs_dir: str,
    dataset: str,
    use_models: List[str],
    reward_key="rm_score",
    answer_key="normalized_answer",
):
    task = infer_task_from_dataset(dataset)
    prompts: Dict[str, Dict] = defaultdict(lambda: {"models": {}})
    models_present = set()

    for model_alias, rec in iter_records(runs_dir, task, dataset):
        if use_models and model_alias not in use_models:
            continue
        resp = rec.get("responses", [])
        if not resp:
            continue

        pid = rec.get("id") or rec.get("task_id") or rec.get("problem_id") or rec.get("question_id")
        if pid is None:
            # fallback: skip if no stable id
            continue

        # Build rewards and outcomes in original order
        rm, y, answers = [], [], []
        if task == "math":
            for r in resp:
                rv = to_float(r.get(reward_key, None))
                cv_raw = r.get("correct", None)
                answer = r.get(answer_key, None)
                if isinstance(cv_raw, bool):
                    cv = 1.0 if cv_raw else 0.0
                else:
                    cv = to_float(cv_raw)
                    if cv is not None:
                        cv = float(np.clip(cv, 0.0, 1.0))
                rm.append(rv)
                y.append(cv)
                answers.append(answer)
        else:  # code
            raise(NotImplementedError("Code task not implemented yet."))

        prompts[pid]["models"][model_alias] = {
            "rm": np.array([(-1e9 if v is None else v) for v in rm], dtype=np.float64),
            "y": np.array([(-1.0 if v is None else v) for v in y], dtype=np.float64),
            "answers": answers if task == "math" else [],
        }
        models_present.add(model_alias)

    return {"task": task, "prompts": prompts, "models": sorted(models_present)}


# ------------------ Per-model curves ------------------
def per_model_curve(data: Dict, model_alias: str, ns: List[int], beta: float, hard_bon: bool):
    """
    For a single model:
      - For each n, for each prompt, form the 'chosen union' = first n responses of this model.
      - Sample ONE response via soft BoN over rewards (temperature beta).
      - Record its y as the per-prompt accuracy value (0/1 for math; fraction for code).
      - Return dict: n -> [mean_acc, ci_lo, ci_hi].
    """
    prompts = data["prompts"]
    per_n_values = {}

    for n in ns:
        accs = []
        for _, entry in prompts.items():
            if model_alias not in entry["models"]:
                continue
            rm_all = entry["models"][model_alias]["rm"]
            L = int(getattr(rm_all, "size", len(rm_all)))
            k = min(n, L)
            if k <= 0:
                continue

            # chosen union = first k responses from this model
            chosen_ids = [(model_alias, i) for i in range(k)]
            if not hard_bon:
                y_pick, _ = pick_sbon_reward(entry, chosen_ids, beta)
            else:
                y_pick, _ = pick_argmax_reward(entry, chosen_ids)
            if np.isfinite(y_pick):
                accs.append(float(y_pick))

        mean_acc, ci_lo, ci_hi = _mean_ci_auto(accs)
        per_n_values[n] = [mean_acc, ci_lo, ci_hi]

    return per_n_values


# ------------------ "Average" line across models ------------------
def average_curve(data: Dict, model_aliases: List[str], ns: List[int], beta: float):
    """
    'Average' baseline:
      - For each model in `model_aliases`, and each prompt, sample ONE response
        from that model’s first n using soft BoN; take its y.
      - Pool all these per-prompt y values across the listed models.
      - Return dict: n -> [mean_acc, ci_lo, ci_hi].
    This matches "average over individual models" when prompts overlap.
    """
    prompts = data["prompts"]
    per_n_values = {}

    for n in ns:
        pooled = []
        for _, entry in prompts.items():
            for m in model_aliases:
                if m not in entry["models"]:
                    continue
                rm_all = entry["models"][m]["rm"]
                L = int(getattr(rm_all, "size", len(rm_all)))
                k = min(n, L)
                if k <= 0:
                    continue
                chosen_ids = [(m, i) for i in range(k)]
                y_pick, _ = pick_sbon_reward(entry, chosen_ids, beta)
                if np.isfinite(y_pick):
                    pooled.append(float(y_pick))

        mean_acc, ci_lo, ci_hi = _mean_ci_auto(pooled)
        per_n_values[n] = [mean_acc, ci_lo, ci_hi]

    return per_n_values


# ------------------ Methods (portfolio lines) ------------------
# The method functions live in methods.py. Import mapping at runtime.
def build_records_by_prompt_for_methods(data, use_models: List[str]):
    """Package per-prompt, per-model arrays for methods."""
    task = data["task"]
    prompts = data["prompts"]
    by_prompt = {}
    for pid, entry in prompts.items():
        record = {}
        ok_any = False
        for m, arr in entry["models"].items():
            if use_models and m not in use_models:
                continue
            record[m] = {"rm": arr["rm"], "y": arr["y"], "answers": arr["answers"]}
            ok_any = True
        if ok_any:
            extra = {}
            if task == "code":
                extra["tests_count"] = entry.get("tests_count", 0)
            by_prompt[pid] = {"models": record, "extra": extra}
    return {"task": task, "by_prompt": by_prompt}


def plot_avg_selection_over_n_area(
    runs_dir: str,
    datasets: List[str],
    use_models: List[str],
    method_name: str,
    ns: List[int],
    beta: float,
    reward_key: str = "rm_score",
    out_path: Optional[str] = None,
):
    """
    For each n in ns, run the portfolio method and compute, per model, the
    percentage of selections out of n (per question). The stacked area for
    each n sums to 100%. Averaged over datasets.

    Notes:
    - We first divide by n to get per-model fractions; then we renormalize
      each n-column so the stack sums to 100% in case < n items were selected.
    """
    if method_name not in METHODS_REGISTRY:
        raise ValueError(f"Unknown method '{method_name}'. Available: {list(METHODS_REGISTRY)}")
    method_fn = METHODS_REGISTRY[method_name]
    model_list = use_models

    series_per_model = {d: {m: [] for m in model_list} for d in datasets}
    for d in datasets:
        data = load_dataset_runs(runs_dir, d, use_models, reward_key=reward_key)
        data_for_methods = build_records_by_prompt_for_methods(data, use_models)
        by_prompt = data_for_methods["by_prompt"]

        any_valid = False
        for n in ns:
            # Expect: method_fn(... ) -> (out_by_pid, counts_by_model)
            _, counts = method_fn(by_prompt=by_prompt, n=int(n), beta=beta, task=data_for_methods["task"], plot=True)

            # Build raw vector in model order
            v = np.array([float(counts.get(m, 0.0)) for m in model_list], dtype=float)

            # Convert to fractions of n (guard n=0) then to percentages
            if n > 0:
                v_frac = v / float(n)
            else:
                v_frac = np.zeros_like(v)

            # Renormalize column to sum to 100% (robust even if total < 1 due to capacity/leak-free constraints)
            col_sum = float(np.nansum(v_frac))
            if col_sum > 0:
                v_pct = 100.0 * (v_frac / col_sum)
            else:
                v_pct = np.zeros_like(v_frac)

            for i, m in enumerate(model_list):
                series_per_model[d][m].append(v_pct[i])

            # Optional: log a concise line
            log_bits = ", ".join(f"{m}:{v_pct[i]:5.1f}%" for i, m in enumerate(model_list))
            print(f"n={n}: {log_bits}")
            any_valid = True

    if not any_valid:
        raise RuntimeError("No usable selections for any n; cannot plot area over n.")
    average_series_per_model = {m: [] for m in model_list}
    for m in model_list:
        for i in range(len(ns)):
            # Average over datasets, ignoring NaNs
            vals = [series_per_model[d][m][i] for d in datasets if np.isfinite(series_per_model[d][m][i])]
            if vals:
                avg_val = float(np.mean(vals))
            else:
                avg_val = float("nan")
            average_series_per_model[m].append(avg_val)

    x = np.array(ns, dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(ns) * 0.5), 4.8))

    # Stacked series (rows=models, cols=len(ns)) in PERCENT
    series = [np.array(average_series_per_model[m], dtype=float) for m in model_list]
    S = np.vstack(series) if len(series) > 0 else np.zeros((0, len(ns)))

    # Replace non-finite with zeros for plotting
    S_plot = np.where(np.isfinite(S), S, 0.0)

    ax.stackplot(x, S_plot, labels=model_list)
    ax.set_xlabel("n", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_ylabel("Selection share (%) per question", fontsize=14)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(4, 256)

    ax.set_title(f"Share of Models Selected (Averaged over Datasets)")
    ax.grid(True, linestyle="--", alpha=0.3)
    leg = ax.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.7, loc="lower right")
    leg.get_frame().set_facecolor("white")

    if out_path is None:
        out_path = f"selection_area_over_n_{method_name}_pct_beta{beta:g}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=180)
    print(f"Saved stacked percentage area-over-n to {out_path}")

def portfolio_method_curve(
    method_fn, method_name: str, data_for_methods, ns: List[int], beta: float, hard_bon: bool, **kwargs
):
    """Call method per (dataset, n); method returns per-prompt expected metric list."""
    task = data_for_methods["task"]
    by_prompt = data_for_methods["by_prompt"]
    per_n_values = {n: [] for n in ns}
    # Call once per n with all prompts; method should return dict pid->value OR list aligned in any order
    for n in ns:
        chosen = method_fn(by_prompt=by_prompt, n=n, beta=beta, task=task, **kwargs)
        accs = []
        for pid, entry in by_prompt.items():
            chosen_ids = chosen.get(pid, [])
            if hard_bon:
                y_pick, _ = pick_argmax_reward(entry, chosen_ids)
            else:
                y_pick, _ = pick_sbon_reward(entry, chosen_ids, beta)
            accs.append(0.0 if not np.isfinite(y_pick) else y_pick)
        mean_acc, ci_lo, ci_hi = binary_mean_and_wilson(accs)

        per_n_values[n] = [mean_acc, ci_lo, ci_hi]
    return per_n_values


# ------------------ Plotting ------------------
def plot_grid(
    runs_dir: str,
    datasets: List[str],
    use_models: List[str],
    ns: List[int],
    betas: List[float],
    methods: List[str],
    reward_key="rm_score",
    out_path=None,
    agreement_weight=0.3,
    hard_bon=False,
):
    from cycler import cycler
    plt.style.use('seaborn-v0_8')
    plt.rcParams['axes.prop_cycle'] = cycler('color', [
        '#1b9e77',  # greenish
        '#d95f02',  # orange
        '#7570b3',   # purple
        '#e7298a',  # pink
        "#7de409",  # green
        "#fdc62d",  # yellow
        "#7e4e17",  # brown
    ])

    n_rows = len(betas)
    n_cols = len(datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.8 * n_cols, 1 + 3.6 * n_rows), squeeze=False, sharey="col"
    )

    for i, ds in tqdm(enumerate(datasets), total=len(datasets), desc="Datasets"):
        task = infer_task_from_dataset(ds)
        data = load_dataset_runs(runs_dir, ds, use_models, reward_key=reward_key)
        # Per-model curves
        model_curves = {}
        for m in use_models or data["models"]:
            model_curves[m] = {}

        # Precompute method view
        data_for_methods = build_records_by_prompt_for_methods(data, use_models)

        for j, beta in tqdm(enumerate(betas), desc="Betas"):
            ax = axes[j][i]
            if j == 0:
                ax.set_title(f"{ds}", fontsize=18)

            # model lines
            for m in use_models or data["models"]:
                per_n = per_model_curve(data, m, ns, beta, hard_bon=hard_bon)
                model_curves[m] = per_n
                means, los, his = [], [], []
                for n in ns:
                    means.append(per_n.get(n, [None])[0])
                    los.append(per_n.get(n, [None])[1])
                    his.append(per_n.get(n, [None])[2])
                x = np.array(ns, dtype=int)
                ax.plot(x, means, marker="o", label=m, alpha=0.4)
                #ax.fill_between(x, np.array(means)-np.array(los), np.array(means)+np.array(his), alpha=0.15)

            # average line across models
            means, los, his = [], [], []
            for n in ns:
                n_means = np.array([model_curves[m].get(n, [None])[0] for m in use_models], dtype=np.float64)
                n_los = np.array([model_curves[m].get(n, [None])[1] for m in use_models], dtype=np.float64)
                n_his = np.array([model_curves[m].get(n, [None])[2] for m in use_models], dtype=np.float64)
                means.append(np.nanmean(n_means))
                los.append(np.nanmean(n_los))
                his.append(np.nanmean(n_his))
            x = np.array(ns, dtype=int)
            ax.plot(x, means, marker="o", linestyle="--", color="black", label="average")
            #ax.fill_between(x, np.array(means)-np.array(los), np.array(means)+np.array(his), color="black", alpha=0.10)

            # portfolio methods
            for meth in methods:
                if meth not in METHODS_REGISTRY:
                    continue
                per_n = portfolio_method_curve(
                    METHODS_REGISTRY[meth],
                    meth,
                    data_for_methods,
                    ns,
                    beta,
                    agreement_weight=agreement_weight,
                    hard_bon=hard_bon
                )
                means, los, his = [], [], []
                for n in ns:
                    means.append(per_n.get(n, [None])[0])
                    los.append(per_n.get(n, [None])[1])
                    his.append(per_n.get(n, [None])[2])
                ax.plot(x, means, marker="D", label=meth)
                #ax.fill_between(x, np.array(means)-np.array(los), np.array(means)+np.array(his), alpha=0.12)

            # Ax cosmetics
            ylabel = "Avg accuracy" if task == "math" else "Avg # passed tests"
            ax.set_xlabel("n")
            if i == 0:
                ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            if j == 0 and i == n_cols - 1:
                leg = ax.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.7, loc="lower right")
                leg.get_frame().set_facecolor("white")

    plt.tight_layout()
    out_path = timestamped_path(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=180)
    print(f"Saved figure to {out_path}")


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser("Multi-dataset soft Best-of-n figure")
    ap.add_argument(
        "--runs_dir",
        default=".",
        help="Root runs dir (e.g., runs)",
    )
    ap.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model aliases (default: {','.join(DEFAULT_MODELS)})",
    )
    ap.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help=f"Comma-separated dataset names (default: {','.join(DEFAULT_DATASETS)})",
    )
    ap.add_argument(
        "--ns", default="1,4,16,64,256", help="Comma-separated n values (default: 1,4,16,64,256)"
    )
    ap.add_argument(
        "--betas",
        default=",".join(str(b) for b in DEFAULT_BETAS),
        help=f"Comma-separated beta values (default: {','.join(str(b) for b in DEFAULT_BETAS)})",
    )
    ap.add_argument(
        "--methods",
        default="equal,RoBoN",
        help='Comma-separated portfolio methods (default: "equal,RoBoN")',
    )
    ap.add_argument(
        "--reward_field", default="rm_normalized", help="Reward key for selection (default: rm_normalized)"
    )
    ap.add_argument(
        "--agreement_weight",
        type=float,
        default=0.4,
        help="Agreement weight for RoBoN method (default: 0.4)",
    )
    ap.add_argument(
        "--not_hard_bon", 
        action="store_true",
        help="Do not use hard Best-of-n for final response selection. Instead, use soft BoN."
    )
    ap.add_argument(
        "--area_over_n", 
        action="store_true", 
        help="Plot stacked area with x=n and y=avg selection per question"
    )

    ap.add_argument(
        "--out", 
        default="bon_grid.png", 
        help="Output PNG file name"
    )
    args = ap.parse_args()

    models = parse_list(args.models, str)
    datasets = parse_list(args.datasets, str)
    ns = parse_list(args.ns, int)
    betas = parse_list(args.betas, float)
    methods = parse_list(args.methods, str)
    
    plot_grid(
        args.runs_dir,
        datasets,
        models,
        ns,
        betas,
        methods,
        reward_key=args.reward_field,
        out_path=args.out,
        agreement_weight=args.agreement_weight,
        hard_bon=not args.not_hard_bon
    )
    
    if args.area_over_n:
        meth = 'RoBoN'
        # Build an out path that includes the method name if needed
        out_path = 'selection_area_over_n.png'
        if "{method}" in out_path:
            out_file = out_path.format(method=meth)
        else:
            if out_path.lower().endswith(".png") or out_path.lower().endswith(".pdf"):
                root, ext = os.path.splitext(out_path)
                out_file = f"{root}_{meth}{ext}"
            else:
                out_file = f"{out_path}_{meth}.png"

        for beta in betas:
            print(f"\nPlotting stacked area-over-n for method '{meth}' on datasets {datasets} with beta={beta}...")
            plot_avg_selection_over_n_area(
                runs_dir=args.runs_dir,
                datasets=datasets,
                use_models=models,
                method_name=meth,
                ns=ns,
                beta=beta,
                reward_key=args.reward_field,
                out_path=out_file,
            )

if __name__ == "__main__":
    main()