from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import metrics



##############################################
############# Core stat functions ############
##############################################

def odds_ratio(y_bool: np.ndarray,
               label_arr: np.ndarray,
               return_table: bool = False,
               smoothed: bool = False,
               epsilon: float = 1e-6,
               flag_zero: bool = False) -> Tuple[Any, ...]:
    """
    Fisher OR (tp*tn)/(fp*fn) for boolean predictions vs 0/1 labels.
    Optionally apply Haldane-Anscombe smoothing for the OR (but Fisher p-value stays exact).
    """
    y_bool = np.asarray(y_bool, dtype=bool)
    label_arr = np.asarray(label_arr)

    tp = int(np.sum((label_arr == 1) & y_bool))
    fp = int(np.sum((label_arr == 0) & y_bool))
    fn = int(np.sum((label_arr == 1) & (~y_bool)))
    tn = int(np.sum((label_arr == 0) & (~y_bool)))

    table = [[tp, fp], [fn, tn]]
    stat, pval = sp.stats.fisher_exact(table)

    if smoothed:
        tp_s = tp + epsilon
        fp_s = fp + epsilon
        fn_s = fn + epsilon
        tn_s = tn + epsilon
        stat = (tp_s * tn_s) / (fp_s * fn_s)

    if flag_zero and (0 in np.array(table)):
        stat = np.nan

    if return_table:
        return table, stat, pval
    return stat, pval


def trap_area(
    x: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray] = None,
    normalize: bool = True,
    xmax: Optional[float] = None,
    xmin: Optional[float] = None,
) -> float:
    """
    Trapezoid integration of y(x).
    If normalize=True, returns mean y over the x window (area / width).
    If w is provided, returns weighted mean: ∫ y w dr / ∫ w dr.
    If xmax or xmin are provided, uses left-continous step convention.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if w is not None:
        w = np.asarray(w, float)
        w = w[m]
        w = w[order]

    if xmin is not None and x[0] > xmin:
        x = np.r_[xmin, x]
        y = np.r_[y[0], y]
        if w is not None:
            w = np.r_[w[0], w]
    if xmax is not None and x[-1] < xmax:
        x = np.r_[x, xmax]
        y = np.r_[y, y[-1]]
        if w is not None:
            w = np.r_[w, w[-1]]

    if w is None:
        area = float(np.trapz(y, x=x))
        if not normalize:
            return area
        width = float(x[-1] - x[0])
        return float(area / width) if width > 0 else np.nan

    num = float(np.trapz(y * w, x=x))
    den = float(np.trapz(w, x=x))
    return float(num / den) if den != 0 else np.nan


##############################################
########### Enrichment–Recall (ER) ###########
##############################################

def enrichment_recall(
    df: pd.DataFrame,
    method: str,
    min_recall: float,
    max_recall: float,
    ascending: bool = True,
    gold_col: str = "label",
) -> pd.DataFrame:
    """ 
    Calculates enrichment over recall per row in df.
    Adapted from Dorans et al. NG 2025.
    Removed drop_duplicates (handled in auerc).
    """
    tmp = df[[method, gold_col]].copy()
    tmp = tmp.sort_values(by = method, ascending = ascending).reset_index(drop = True)

    linked = tmp[method].notna().astype(int)
    linked_cum = linked.cumsum()
    gold_cum = tmp[gold_col].cumsum()
    total_gold = gold_cum.max()
    total_length = len(tmp)

    recall = gold_cum / total_gold
    enrich_denom = total_gold / total_length

    enrichment = (gold_cum.divide(linked_cum)) / enrich_denom
    out = pd.DataFrame({"recall": recall, "enrichment": enrichment})
    out = out[linked.astype(bool)]
    out = out[(out["recall"] >= min_recall) & (out["recall"] <= max_recall)]
    return out


def auerc(
    df: pd.DataFrame,
    method: str,
    ascending: bool = True,
    gold_col: str = "label",
    min_recall: float = 0.0,
    max_recall: float = 1.0,
    weighted: bool = True,
    average: bool = False
) -> float:
    """
    Area-under enrichment-recall curve (AUERC).
    If average=True, uses point average with first (smallest) recall.
    If weighted=True, uses w(r)=1-r.
    """
    er = enrichment_recall(
        df, method, min_recall=min_recall, max_recall=max_recall,
        ascending=ascending, gold_col=gold_col)
    
    if er.empty or len(er) < 2:
        return np.nan

    if average:
        er = er.sort_values("recall").drop_duplicates("recall", keep="first")
    
    weights = 1 - er["recall"] if weighted else None

    if average:
        if np.sum(weights) == 0:
            return np.nan
        else:
            return np.average(er["enrichment"], weights=weights)
    else:
        return trap_area(er["recall"], er["enrichment"], xmin=min_recall, xmax=max_recall, w=weights)


##############################################
########### TopK OR curve + AUORC ############
##############################################

def default_k_values(
    df: pd.DataFrame, 
    gold_col: str = "label",
) -> np.ndarray:
    """ 
    Evaluate at each k up to min(total positive, 99% of dataset).
    """
    total_gold = float(df[gold_col].sum())
    total_length = int(len(df))
    k_max = min(total_gold, int(0.99 * total_length))
    if k_max < 1:
        return np.array([1], dtype=int)
    k_values = np.arange(1, k_max + 1).astype(int)
    return k_values


def topk_log_odds_ratio(
    df: pd.DataFrame,
    method: str,
    k_values: Optional[List[int]] = None,
    ascending: bool = True,
    gold_col: str = "label",
    epsilon: float = 0.5,
) -> pd.DataFrame:
    """
    Top-k log odds ratio evaluated at fixed k values.
    Note: Requires binary labels.
    """

    if k_values is None:
        k_values = default_k_values(df, gold_col=gold_col)

    tmp = df[[method, gold_col]].copy()
    total_gold = float(tmp[gold_col].sum())
    total_length = int(len(tmp))

    empty_df = pd.DataFrame({"k": [], "tp": [], "fp": [], "fn": [], "tn": [], "log_odds_ratio": []})
    if total_gold <= 0 or total_gold >= total_length or total_length <= 0:
        return empty_df

    tmp = tmp.sort_values(by=method, ascending=ascending, kind="mergesort").reset_index(drop=True)
    tp_cum = tmp[gold_col].cumsum().to_numpy(dtype=float)

    k = np.array([int(x) for x in k_values if x is not None], dtype=int)
    k = np.unique(k)
    k = k[(k >= 1) & (k <= total_length)]
    if k.size == 0:
        return empty_df

    tp = tp_cum[k - 1]
    sel_n = k.astype(float)

    fp = sel_n - tp
    fn = total_gold - tp
    tn = (total_length - total_gold) - fp

    log_odds_ratio = np.log(((tp + epsilon) * (tn + epsilon)) / ((fp + epsilon) * (fn + epsilon)))

    out = pd.DataFrame({"k": k, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "log_odds_ratio": log_odds_ratio.astype(float)})
    return out


def auorc_log(
    df: pd.DataFrame,
    col: str,
    k_values: Optional[List[int]] = None,
    ascending: bool = True,
    gold_col: str = "label",
    max_recall: float = 1.0,
    epsilon: float = 0.5,
    weighted: bool = True,
    log: bool = True,
    average: bool = False,
) -> float:
    """
    AUORC_log: area under log_odds_ratio vs k.
    Weighted option uses w(k)=(1/k), equivalent to log-k weighting.
    Note: Using point average does not account for log-k spacing.
    """
    if k_values is None:
        k_values = default_k_values(df, gold_col=gold_col)

    k_or = topk_log_odds_ratio(
        df, col, k_values=k_values, gold_col=gold_col, ascending=ascending, epsilon=epsilon)

    if k_or.empty or len(k_or) < 2:
        return np.nan

    weights = (1.0 / k_or["k"]) if weighted else None

    if average:
        return np.average(k_or["log_odds_ratio"], weights=weights)
    else:
        return trap_area(k_or["k"], k_or["log_odds_ratio"], w=weights)


##############################################
######### PR metrics (AP, partialAP) #########
##############################################

def default_score_transform_pvalue(p: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Convert p-values (smaller=better) to a higher-is-better score."""
    p = np.asarray(p, float)
    return -np.log10(np.clip(p, eps, 1.0))


def average_precision(
    df: pd.DataFrame,
    col: str,
    gold_col: str = "label",
    score_transform: Optional[Callable[[np.ndarray], np.ndarray]] = default_score_transform_pvalue
) -> float:

    sub = df[[col, gold_col]].dropna()
    y_true = sub[gold_col].dropna().to_numpy().astype(int)
    if y_true.sum() == 0:
        return np.nan
    y_score = sub[col].to_numpy()
    if score_transform is not None:
        y_score = score_transform(y_score)
    return float(metrics.average_precision_score(y_true, y_score))


def partial_average_precision(
    df: pd.DataFrame,
    col: str,
    R: float = 0.2,
    gold_col: str = "label",
    normalize: bool = True,
    score_transform: Optional[Callable[[np.ndarray], np.ndarray]] = default_score_transform_pvalue
) -> float:
    """
    pAP@R = ∫_0^R precision(r) dr. If normalize=True, divide by R.
    """

    sub = df[[col, gold_col]].dropna()
    y_true = sub[gold_col].dropna().to_numpy().astype(int)
    if y_true.sum() == 0:
        return np.nan
    y_score = sub[col].to_numpy()
    if score_transform is not None:
        y_score = score_transform(y_score)

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    m = recall <= R
    if m.sum() < 2:
        return np.nan
    area = float(np.trapz(precision[m], recall[m]))
    return float(area / R) if normalize else area


##############################################
############## Bootstrap class ###############
##############################################

MetricFn = Callable[[pd.DataFrame, str], float]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: MetricFn
    larger_is_better: bool = True


def _paired_boot_summary(
    ref_boot: np.ndarray, 
    m_boot: np.ndarray, 
    alpha: float
) -> Dict[str, float]:
    """
    Paired CI on (ref - m) and a simple two-sided sign test p-value.
    """
    mask = np.isfinite(ref_boot) & np.isfinite(m_boot)
    if mask.sum() == 0:
        return {"diff_ci_lower": np.nan, "diff_ci_upper": np.nan, "p_value": np.nan}

    diffs = ref_boot[mask] - m_boot[mask]
    lo = np.nanquantile(diffs, alpha / 2)
    hi = np.nanquantile(diffs, 1 - alpha / 2)

    n = diffs.size
    n_lt = np.sum(diffs < 0)
    n_gt = np.sum(diffs > 0)
    n_eq = n - n_lt - n_gt
    p_lower = (n_lt + 0.5 * n_eq) / n
    p_upper = (n_gt + 0.5 * n_eq) / n
    pval = 2.0 * min(p_lower, p_upper)
    pval = float(min(1.0, max(0.0, pval)))

    return {"diff_ci_lower": float(lo), "diff_ci_upper": float(hi), "p_value": pval}


def _bootstrap_indices(
    N: int,
    rng: np.random.Generator,
    labels: Optional[np.ndarray] = None,
    stratified: bool = False,
) -> np.ndarray:
    """
    Bootstrap row indices.
    If stratified=True, sample positives and negatives separately to preserve class counts.
    """
    if (not stratified) or (labels is None):
        return rng.integers(0, N, size=N)

    y = np.asarray(labels, dtype=float)
    pos_idx = np.flatnonzero(y > 0)
    neg_idx = np.flatnonzero(y <= 0)

    if pos_idx.size == 0 or neg_idx.size == 0:
        return rng.integers(0, N, size=N)

    draw_pos = rng.choice(pos_idx, size=pos_idx.size, replace=True)
    draw_neg = rng.choice(neg_idx, size=neg_idx.size, replace=True)
    idx = np.concatenate([draw_pos, draw_neg])
    rng.shuffle(idx)
    return idx


def _safe_mean_finite(values: np.ndarray) -> float:
    """Mean over finite values; returns NaN if none are finite."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.nan
    return float(np.mean(arr[finite]))


def _prepare_work_df(
    all_df: pd.DataFrame,
    methods: List[str],
    gold_col: str,
    random_state: Optional[int],
    fillna: bool,
    jitter_amount: Optional[float],
    clip_min: Optional[float],
    handle_dup: Optional[str],
    dup_key_cols: Optional[List[str]],
    tie: str,
) -> pd.DataFrame:
    """
    Shared preprocessing: numeric coercion, clipping, fillna policy, optional jitter, optional dedup.
    """
    rng = np.random.default_rng(random_state)
    work_df = all_df.copy()

    if jitter_amount is not None:
        noise = rng.uniform(-jitter_amount, jitter_amount, size=work_df.shape[0])

    for m in methods:
        work_df[m] = pd.to_numeric(work_df[m], errors="coerce")
        if clip_min is not None:
            work_df[m] = work_df[m].clip(lower=clip_min)
        if fillna:
            work_df[m] = work_df[m].fillna(1.0)
        if jitter_amount is not None:
            mask = work_df[m].notna()
            work_df.loc[mask, m] = work_df.loc[mask, m] + noise[mask]

    if handle_dup is not None:
        if dup_key_cols is None:
            raise ValueError("dup_key_cols must be provided when handle_dup is set")
        work_df = dedup_df(
            work_df,
            key_cols=dup_key_cols,
            gold_col=gold_col,
            score_cols=methods,
            handle_dup=handle_dup,
            tie=tie,
        )

    return work_df


def _paired_bootstrap_est_and_samples(
    work_df: pd.DataFrame,
    methods: List[str],
    metrics_list: List[MetricSpec],
    n_bootstrap: int,
    rng: np.random.Generator,
    gold_col: str = "label",
    stratified_bootstrap: bool = False,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Dict[str, np.ndarray]]]:
    """Shared paired bootstrap core. Returns point estimates + bootstrap samples per metric/method."""
    N = len(work_df)
    if N == 0:
        return {}, {}

    est = {(ms.name, m): ms.fn(work_df, m) for ms in metrics_list for m in methods}

    boot: Dict[str, Dict[str, np.ndarray]] = {
        ms.name: {m: np.empty(n_bootstrap, dtype=float) for m in methods}
        for ms in metrics_list
    }

    label_arr = work_df[gold_col].to_numpy() if stratified_bootstrap else None

    for b in range(n_bootstrap):
        idx = _bootstrap_indices(
            N=N,
            rng=rng,
            labels=label_arr,
            stratified=stratified_bootstrap,
        )
        sample = work_df.iloc[idx].reset_index(drop=True)
        for ms in metrics_list:
            for m in methods:
                boot[ms.name][m][b] = ms.fn(sample, m)

    return est, boot


def compute_bootstrap_table(
    all_df: pd.DataFrame,
    methods: List[str],
    metrics_list: List[MetricSpec],
    gold_col: str = "label",
    reference_method: str = "ctar_filt",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: Optional[int] = None,
    fillna: bool = False,
    jitter_amount: Optional[float] = None,
    clip_min: Optional[float] = 1e-300,
    handle_dup: Optional[str] = None,
    dup_key_cols: Optional[List[str]] = None,
    tie: str = "zero",
    stratified_bootstrap: bool = False,
) -> pd.DataFrame:
    """
    Paired bootstrap across methods for multiple metrics.

    Output columns:
      method, metric,
      estimate, ci_lower, ci_upper,
      diff_estimate, diff_ci_lower, diff_ci_upper, p_value,
      n_bootstrap
    """

    if reference_method not in methods:
        raise ValueError(f"reference_method {reference_method} not in methods")
    for m in methods:
        if m not in all_df.columns:
            raise ValueError(f"Missing method column: {m}")
    if gold_col not in all_df.columns:
        raise ValueError(f"Missing gold_col: {gold_col}")

    rng = np.random.default_rng(random_state)
    work_df = _prepare_work_df(
        all_df=all_df,
        methods=methods,
        gold_col=gold_col,
        random_state=random_state,
        fillna=fillna,
        jitter_amount=jitter_amount,
        clip_min=clip_min,
        handle_dup=handle_dup,
        dup_key_cols=dup_key_cols,
        tie=tie,
    )

    N = len(work_df)
    if N == 0:
        return pd.DataFrame()

    alpha = 1.0 - ci

    est, boot = _paired_bootstrap_est_and_samples(
        work_df=work_df,
        methods=methods,
        metrics_list=metrics_list,
        n_bootstrap=n_bootstrap,
        rng=rng,
        gold_col=gold_col,
        stratified_bootstrap=stratified_bootstrap,
    )

    rows = []
    for ms in metrics_list:
        ref_boot = boot[ms.name][reference_method]
        ref_est = est[(ms.name, reference_method)]

        for m in methods:
            m_boot = boot[ms.name][m]
            m_est = est[(ms.name, m)]
            finite = np.isfinite(m_boot)
            n_finite = int(finite.sum())
            n_nonfinite = int((~finite).sum())

            row = {
                "method": m,
                "metric": ms.name,
                "estimate": float(m_est) if np.isfinite(m_est) else np.nan,
                "bootstrap_sd": float(np.nanstd(m_boot[finite], ddof=1)) if n_finite > 1 else np.nan,
                "ci_lower": float(np.nanquantile(m_boot[finite], alpha/2)) if finite.any() else np.nan,
                "ci_upper": float(np.nanquantile(m_boot[finite], 1-alpha/2)) if finite.any() else np.nan,
                "diff_estimate": float(ref_est - m_est) if (np.isfinite(ref_est) and np.isfinite(m_est)) else np.nan,
                "n_bootstrap_finite": n_finite,
                "n_bootstrap_nonfinite": n_nonfinite,
                "n_bootstrap": int(n_bootstrap),
            }

            if m == reference_method:
                row.update({"diff_ci_lower": 0.0, "diff_ci_upper": 0.0, "p_value": 1.0})
            else:
                row.update(_paired_boot_summary(ref_boot, m_boot, alpha))

            rows.append(row)

    return pd.DataFrame(rows)


def compute_bootstrap_table_seed_avg(
    all_df: pd.DataFrame,
    methods: List[str],
    metrics_list: List[MetricSpec],
    gold_col: str = "label",
    reference_method: str = "ctar_filt",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seeds: List[int] = None,
    fillna: bool = False,
    jitter_amount: Optional[float] = None,
    clip_min: Optional[float] = 1e-300,
    handle_dup: Optional[str] = None,
    dup_key_cols: Optional[List[str]] = None,
    tie: str = "zero",
    bootstrap_random_state: int = 0,
    ddof: int = 1,
    stratified_bootstrap: bool = False,
) -> pd.DataFrame:
    """
    Paired bootstrap with seed-averaged evaluation.
      - Point estimate = mean over seeds of metric(full_data, jittered(seed))
      - seed_sd = SD over seeds of metric(full_data, jittered(seed))
      - Bootstrap distribution: for each bootstrap replicate, resample rows once,
        compute metric on each seed's jittered df, then average across seeds.
      - Two-sided paired p-values/CI reuse _paired_boot_summary on seed-averaged boot arrays.
    """
    if seeds is None:
        seeds = list(range(10))

    if reference_method not in methods:
        raise ValueError(f"reference_method {reference_method} not in methods")
    for m in methods:
        if m not in all_df.columns:
            raise ValueError(f"Missing method column: {m}")
    if gold_col not in all_df.columns:
        raise ValueError(f"Missing gold_col: {gold_col}")

    # Prepare one preprocessed dataframe per seed
    seed_work = [
        _prepare_work_df(
            all_df=all_df,
            methods=methods,
            gold_col=gold_col,
            random_state=s,
            fillna=fillna,
            jitter_amount=jitter_amount,
            clip_min=clip_min,
            handle_dup=handle_dup,
            dup_key_cols=dup_key_cols,
            tie=tie,
        )
        for s in seeds
    ]

    N = len(seed_work[0])
    if N == 0:
        return pd.DataFrame()

    alpha = 1.0 - ci

    # Point estimates per seed, then mean/sd across seeds
    per_seed_est: Dict[Tuple[str, str], np.ndarray] = {}
    for ms in metrics_list:
        for m in methods:
            vals = np.array([ms.fn(df, m) for df in seed_work], dtype=float)
            per_seed_est[(ms.name, m)] = vals

    seed_mean = {(k): _safe_mean_finite(v) for k, v in per_seed_est.items()}
    seed_sd = {
        (k): float(np.nanstd(v, ddof=ddof))
        if np.isfinite(v).sum() > 1 else np.nan
        for k, v in per_seed_est.items()
    }

    # Bootstrap seed-averaged distribution
    rng = np.random.default_rng(bootstrap_random_state)

    boot: Dict[str, Dict[str, np.ndarray]] = {
        ms.name: {m: np.empty(n_bootstrap, dtype=float) for m in methods}
        for ms in metrics_list
    }

    label_arr = seed_work[0][gold_col].to_numpy() if stratified_bootstrap else None
    for b in range(n_bootstrap):
        idx = _bootstrap_indices(
            N=N,
            rng=rng,
            labels=label_arr,
            stratified=stratified_bootstrap,
        )
        for ms in metrics_list:
            for m in methods:
                vals = [ms.fn(df.iloc[idx].reset_index(drop=True), m) for df in seed_work]
                boot[ms.name][m][b] = _safe_mean_finite(np.asarray(vals, dtype=float))

    # Summarize as in compute_bootstrap_table, plus seed_sd
    rows = []
    for ms in metrics_list:
        ref_boot = boot[ms.name][reference_method]
        ref_est = seed_mean[(ms.name, reference_method)]

        for m in methods:
            m_boot = boot[ms.name][m]
            m_est = seed_mean[(ms.name, m)]

            finite = np.isfinite(m_boot)
            n_finite = int(finite.sum())
            n_nonfinite = int((~finite).sum())
            row = {
                "method": m,
                "metric": ms.name,
                "estimate": float(m_est) if np.isfinite(m_est) else np.nan,
                "seed_sd": float(seed_sd[(ms.name, m)]) if np.isfinite(seed_sd[(ms.name, m)]) else np.nan,
                "n_seeds": int(len(seeds)),
                "bootstrap_sd": float(np.nanstd(m_boot[finite], ddof=1)) if n_finite > 1 else np.nan,
                "ci_lower": float(np.nanquantile(m_boot[finite], alpha/2)) if finite.any() else np.nan,
                "ci_upper": float(np.nanquantile(m_boot[finite], 1-alpha/2)) if finite.any() else np.nan,
                "diff_estimate": float(ref_est - m_est) if (np.isfinite(ref_est) and np.isfinite(m_est)) else np.nan,
                "n_bootstrap_finite": n_finite,
                "n_bootstrap_nonfinite": n_nonfinite,
                "n_bootstrap": int(n_bootstrap),
            }

            if m == reference_method:
                row.update({"diff_ci_lower": 0.0, "diff_ci_upper": 0.0, "p_value": 1.0})
            else:
                row.update(_paired_boot_summary(ref_boot, m_boot, alpha))

            rows.append(row)

    return pd.DataFrame(rows)


##############################################
############### Deduplication ################
##############################################

def dedup_df(
    work_df: pd.DataFrame,
    key_cols: List[str],
    gold_col: str = "label",
    score_cols: Optional[List[str]] = None,
    handle_dup: str = "any",
    tie: str = "zero",
) -> pd.DataFrame:
    """
    Deduplicate rows by key_cols, aggregating gold_col and taking min() of score cols.
    """
    if key_cols is None or len(key_cols) == 0:
        raise ValueError("Provide key_cols for grouping.")

    def any_positive(x):
        return float((x.astype(float) > 0).any())

    def consensus_majority(x):
        x = x.astype(float)
        pos = (x > 0).sum()
        neg = len(x) - pos
        if pos > neg:
            return 1.0
        elif pos < neg:
            return 0.0
        return {"zero": 0.0, "half": 0.5, "prefer_positive": 1.0, "prefer_negative": 0.0}[tie]

    def proportion_positive(x):
        x = x.astype(float)
        return float((x > 0).mean())

    if handle_dup == "any":
        label_agg = any_positive
    elif handle_dup == "consensus":
        label_agg = consensus_majority
    elif handle_dup == "proportion":
        label_agg = proportion_positive
    else:
        raise ValueError("handle_dup must be one of: 'any', 'consensus', 'proportion'")

    agg_dic = {gold_col: label_agg}
    if score_cols:
        for sc in score_cols:
            agg_dic[sc] = "min"

    return work_df.groupby(list(key_cols), as_index=False).agg(agg_dic)


##############################################
############# Build metric specs #############
##############################################

def build_default_metric_specs(
    all_df: pd.DataFrame,
    gold_col: str = "label",
    pvals_smaller_is_better: bool = True,
    early_R: float = 0.2,
) -> List[MetricSpec]:
    """
    Default metrics:
    - AUERC (weighted) full
    - AE (weighted) full
    - AUERC (unweighted) early
    - AUORC (weighted log OR)
    - AP and pAP@R
    """
    ascending = True if pvals_smaller_is_better else False
    k_values = default_k_values(all_df, gold_col=gold_col)
    print(f'Top-k for AUORC_log: {k_values}')

    def _auerc_full(df: pd.DataFrame, m: str) -> float:
        return auerc(df, m, gold_col=gold_col, ascending=ascending,
                     min_recall=0.0, max_recall=1.0, weighted=True, average=False)

    def _avg_enrich(df: pd.DataFrame, m: str) -> float:
        return auerc(df, m, gold_col=gold_col, ascending=ascending,
                     min_recall=0.0, max_recall=1.0, weighted=True, average=True)

    def _auerc_early(df: pd.DataFrame, m: str) -> float:
        return auerc(df, m, gold_col=gold_col, ascending=ascending,
                     min_recall=0.0, max_recall=early_R, weighted=False, average=False)

    def _auorc_log(df: pd.DataFrame, m: str) -> float:
        return auorc_log(df, m, k_values=k_values, gold_col=gold_col, ascending=ascending,
                     epsilon=0.5, weighted=True)

    def _ap(df: pd.DataFrame, m: str) -> float:
        return average_precision(df, m, gold_col=gold_col, score_transform=default_score_transform_pvalue)

    def _pap(df: pd.DataFrame, m: str) -> float:
        return partial_average_precision(df, m, R=early_R, gold_col=gold_col, normalize=True,
                                         score_transform=default_score_transform_pvalue)

    return [
        MetricSpec("AUERC", _auerc_full),
        MetricSpec("AE", _avg_enrich),
        MetricSpec(f"AUERC≤{early_R}", _auerc_early),
        MetricSpec("AUORC_log", _auorc_log),
        MetricSpec("AP", _ap),
        MetricSpec(f"pAP@{early_R}", _pap),
    ]


##############################################
############### Example usage ################
##############################################

"""
methods = ['scmm','signac','ctar_filt','ctar_filt_z','ctar_filt_10k','scmm_mc','corr_mc']
metric_specs = build_default_metric_specs(gold_col="label", pvals_smaller_is_better=True, early_R=0.2)

boot_df = compute_bootstrap_table(
    all_df=eval_df,
    methods=methods,
    metrics_list=metric_specs,
    gold_col="label",
    reference_method="ctar_filt",
    n_bootstrap=1000,
    ci=0.95,
    random_state=0,
    fillna=False,
    clip_min=1e-300,
)

# boot_df will have one row per (method, metric).
# Pivot for display
# boot_df.pivot_table(index="method", columns="metric", values="estimate")
"""