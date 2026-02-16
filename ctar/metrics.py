from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import metrics



##############################
##### Core stats helpers #####
##############################

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


def trap_area_over_recall(recall: np.ndarray,
                          y: np.ndarray,
                          w: Optional[np.ndarray] = None,
                          normalize: bool = True) -> float:
    """
    Trapezoid integration of y(recall).
    If normalize=True, returns mean y over the recall window (area / width).
    If w is provided, returns weighted mean: ∫ y w dr / ∫ w dr.
    """
    recall = np.asarray(recall, float)
    y = np.asarray(y, float)

    m = np.isfinite(recall) & np.isfinite(y)
    recall = recall[m]
    y = y[m]
    if recall.size < 2:
        return np.nan

    order = np.argsort(recall)
    recall = recall[order]
    y = y[order]

    if w is None:
        area = float(np.trapz(y, recall))
        if not normalize:
            return area
        width = float(recall[-1] - recall[0])
        return float(area / width) if width > 0 else np.nan

    w = np.asarray(w, float)
    w = w[m][order]
    num = float(np.trapz(y * w, recall))
    den = float(np.trapz(w, recall))
    return float(num / den) if den != 0 else np.nan


##############################
### Enrichment–Recall (ER) ###
##############################

def pgb_enrichment_recall_tieblocks(
    df: pd.DataFrame,
    col: str,
    min_recall: float,
    max_recall: float,
    ascending: bool = True,
    gold_col: str = "label",
    extrapolate: bool = True,
) -> pd.DataFrame:
    """
    Tie-robust enrichment-recall:
    - "linked" items are those with df[col].notna() & (df[col] > 0)
    - sort by col (ascending=True for p-values)
    - group by identical score values (tie blocks)
    - compute cumulative linked and gold at block boundaries
    """
    tmp = df[[col, gold_col]].copy()

    total_gold = float(tmp[gold_col].sum())
    total_length = int(len(tmp))
    if total_gold <= 0 or total_length <= 0:
        return pd.DataFrame({"recall": [], "enrichment": []})

    linked_mask = tmp[col].notna() & (tmp[col] > 0)
    scored = tmp.loc[linked_mask].copy()
    unscored = tmp.loc[~linked_mask].copy()

    if scored.empty:
        # degenerate; optionally extrapolate to enrichment=1
        out = pd.DataFrame({"recall": [], "enrichment": []})
        if extrapolate and float(unscored[gold_col].sum()) > 0:
            out = pd.DataFrame({"recall": [min_recall], "enrichment": [1.0]})
        return out

    scored = scored.sort_values(by=col, ascending=ascending, kind="mergesort")
    g = scored.groupby(col, sort=False, dropna=False)

    block_n = g.size().to_numpy(dtype=float)
    block_gold = g[gold_col].sum().to_numpy(dtype=float)

    linked_cum = np.cumsum(block_n)
    gold_cum = np.cumsum(block_gold)

    recall = gold_cum / total_gold
    enrich_denom = total_gold / total_length
    enrichment = (gold_cum / linked_cum) / enrich_denom

    out = pd.DataFrame({"recall": recall, "enrichment": enrichment})

    # optional extrapolation over unscored positives
    if extrapolate:
        remaining_gold = float(unscored[gold_col].sum())
        if remaining_gold > 0:
            last_recall = float(out["recall"].iloc[-1])
            last_enrich = float(out["enrichment"].iloc[-1])

            num_new_points = int(np.ceil(remaining_gold))
            if num_new_points > 0 and last_recall < 1:
                recall_increment = (1 - last_recall) / num_new_points
                enrich_increment = (last_enrich - 1) / num_new_points
                new_recall = [last_recall + recall_increment * (i + 1) for i in range(num_new_points)]
                new_enrich = [last_enrich - enrich_increment * (i + 1) for i in range(num_new_points)]
                out = pd.concat([out, pd.DataFrame({"recall": new_recall, "enrichment": new_enrich})],
                                ignore_index=True)

    out = out[(out["recall"] >= min_recall) & (out["recall"] <= max_recall)]
    out = out.drop_duplicates(subset=["recall"], keep="first")
    return out


def auerc(
    df: pd.DataFrame,
    col: str,
    gold_col: str = "label",
    ascending: bool = True,
    min_recall: float = 0.0,
    max_recall: float = 1.0,
    weighted: bool = True,
    tieblocks: bool = True,
    extrapolate: bool = True,
) -> float:
    """
    Area-under enrichment-recall curve (AUERC), computed via trapezoid integration over recall.
    If weighted=True, uses w(r)=1-r.
    """
    if tieblocks:
        er = pgb_enrichment_recall_tieblocks(
            df, col, min_recall=min_recall, max_recall=max_recall,
            ascending=ascending, gold_col=gold_col, extrapolate=extrapolate
        )
    else:
        # fallback: treat each row as its own "block" (stable if no ties)
        tmp = df[[col, gold_col]].copy()
        total_gold = float(tmp[gold_col].sum())
        total_length = int(len(tmp))
        if total_gold <= 0 or total_length <= 0:
            return np.nan
        tmp = tmp.sort_values(by=col, ascending=ascending).reset_index(drop=True)
        linked = (tmp[col].notna() & (tmp[col] > 0)).astype(int)
        linked_cum = linked.cumsum().to_numpy(dtype=float)
        gold_cum = (tmp[gold_col] * linked).cumsum().to_numpy(dtype=float)
        recall = gold_cum / total_gold
        enrich_denom = total_gold / total_length
        enrichment = (gold_cum / linked_cum) / enrich_denom
        er = pd.DataFrame({"recall": recall, "enrichment": enrichment})
        er = er[(er["recall"] >= min_recall) & (er["recall"] <= max_recall)].drop_duplicates("recall")

    if er.empty or len(er) < 2:
        return np.nan

    r = er["recall"].to_numpy(dtype=float)
    e = er["enrichment"].to_numpy(dtype=float)
    w = (1.0 - r) if weighted else None
    return trap_area_over_recall(r, e, w=w, normalize=True)


##############################
### TopK OR curve + AUORC ####
##############################

def topk_or_curve_tieblocks(
    eval_df: pd.DataFrame,
    method: str,
    gold_col: str = "label",
    ascending: bool = True,
    epsilon: float = 0.5,
    k_max: Any = "P",            # "P", int, or None
    include_partial_last: bool = True,
    mask_scored: bool = True,
) -> pd.DataFrame:
    """
    Tie-robust topK OR curve.
    - sort rows by method score
    - group by tied score values
    - compute OR at block boundaries
    - optionally add a "partial block" point exactly at k_max (expected within-tie)
    """
    df = eval_df[[method, gold_col]].copy()

    if mask_scored:
        df = df[df[method].notna()].copy()

    df = df.sort_values(method, ascending=ascending, kind="mergesort")
    g = df.groupby(method, sort=False, dropna=False)

    block_n = g.size().to_numpy(dtype=float)
    block_pos = g[gold_col].sum().to_numpy(dtype=float)

    sel_n = np.cumsum(block_n)
    tp = np.cumsum(block_pos)

    N = float(len(df))
    P = float(df[gold_col].sum())

    if P <= 0 or P >= N:
        return pd.DataFrame({"k": sel_n.astype(int), "odds_ratio": np.nan, "tp": tp, "fp": np.nan, "fn": np.nan, "tn": np.nan})

    # decide cap
    if k_max == "P":
        Kcap = P
    elif k_max is None:
        Kcap = np.inf
    else:
        Kcap = float(k_max)

    fp = sel_n - tp
    fn = P - tp
    tn = (N - P) - fp
    OR = ((tp + epsilon) * (tn + epsilon)) / ((fp + epsilon) * (fn + epsilon))

    out = pd.DataFrame({"k": sel_n, "odds_ratio": OR, "tp": tp, "fp": fp, "fn": fn, "tn": tn})

    if np.isfinite(Kcap):
        out_trunc = out[out["k"] <= Kcap].copy()
        last_k = float(out_trunc["k"].iloc[-1]) if len(out_trunc) else 0.0

        if include_partial_last and last_k < Kcap:
            j = int(np.searchsorted(sel_n, Kcap, side="left"))
            prev_sel = sel_n[j - 1] if j > 0 else 0.0
            prev_tp = tp[j - 1] if j > 0 else 0.0

            m = block_n[j]
            gpos = block_pos[j]
            t = Kcap - prev_sel

            tp_partial = prev_tp + t * (gpos / m)
            sel_partial = Kcap
            fp_partial = sel_partial - tp_partial
            fn_partial = P - tp_partial
            tn_partial = (N - P) - fp_partial
            or_partial = ((tp_partial + epsilon) * (tn_partial + epsilon)) / ((fp_partial + epsilon) * (fn_partial + epsilon))

            extra = pd.DataFrame({
                "k": [Kcap],
                "odds_ratio": [or_partial],
                "tp": [tp_partial], "fp": [fp_partial], "fn": [fn_partial], "tn": [tn_partial],
            })
            out_trunc = pd.concat([out_trunc, extra], ignore_index=True)

        out = out_trunc

    out["k"] = out["k"].round().astype(int)
    out["odds_ratio"] = out["odds_ratio"].astype(float)
    return out


def auorc(
    df: pd.DataFrame,
    col: str,
    gold_col: str = "label",
    ascending: bool = True,
    max_recall: float = 1.0,
    epsilon: float = 0.5,
    k_max: Any = "P",
    weighted: bool = True,
    log_or: bool = True,
    include_partial_last: bool = True,
) -> float:
    """
    AUORC: area under (log) odds_ratio vs recall (recall = k/P), using trapezoid integration.
    Weighted option uses w(r)=1-r.
    """
    or_df = topk_or_curve_tieblocks(
        df, col, gold_col=gold_col, ascending=ascending, epsilon=epsilon,
        k_max=k_max, include_partial_last=include_partial_last, mask_scored=True
    )
    if or_df.empty or len(or_df) < 2:
        return np.nan

    P = float(df[gold_col].sum())
    if P <= 0:
        return np.nan

    r = or_df["k"].to_numpy(dtype=float) / P
    y = or_df["odds_ratio"].to_numpy(dtype=float)

    m = np.isfinite(r) & np.isfinite(y) & (r > 0) & (r <= max_recall) & (y > 0)
    r = r[m]
    y = y[m]
    if r.size < 2:
        return np.nan

    if log_or:
        y = np.log(y)

    w = (1.0 - r) if weighted else None
    return trap_area_over_recall(r, y, w=w, normalize=True)


##############################
# PR metrics (AP, partialAP) #
##############################

def default_score_transform_pvalue(p: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """Convert p-values (smaller=better) to a higher-is-better score."""
    p = np.asarray(p, float)
    return -np.log10(np.clip(p, eps, 1.0))


def average_precision(df: pd.DataFrame,
                      col: str,
                      gold_col: str = "label",
                      score_transform: Optional[Callable[[np.ndarray], np.ndarray]] = default_score_transform_pvalue) -> float:
    sub = df[[col, gold_col]].dropna()
    y_true = sub[gold_col].dropna().to_numpy().astype(int)
    if y_true.sum() == 0:
        return np.nan
    y_score = sub[col].to_numpy()
    if score_transform is not None:
        y_score = score_transform(y_score)
    return float(metrics.average_precision_score(y_true, y_score))


def partial_average_precision(df: pd.DataFrame,
                              col: str,
                              R: float = 0.2,
                              gold_col: str = "label",
                              normalize: bool = True,
                              score_transform: Optional[Callable[[np.ndarray], np.ndarray]] = default_score_transform_pvalue) -> float:
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


##############################
###### Bootstrap class #######
##############################

MetricFn = Callable[[pd.DataFrame, str], float]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: MetricFn
    larger_is_better: bool = True


def _paired_boot_summary(ref_boot: np.ndarray, m_boot: np.ndarray, alpha: float) -> Dict[str, float]:
    """
    Paired CI on (ref - m) and a simple two-sided "sign test style" p-value.
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
    clip_min: Optional[float] = 1e-300,
    handle_dup: Optional[str] = None,
    dup_key_cols: Optional[List[str]] = None,
    tie: str = "zero",
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
    work_df = all_df.copy()

    # Clean score columns
    for m in methods:
        work_df[m] = pd.to_numeric(work_df[m], errors="coerce")
        if clip_min is not None:
            # for p-values this avoids -inf in transforms
            work_df[m] = work_df[m].clip(lower=clip_min)
        if fillna:
            # "missing means worst" policy
            work_df[m] = work_df[m].fillna(1.0)

    # Optional dedup
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

    N = len(work_df)
    if N == 0:
        return pd.DataFrame()

    alpha = 1.0 - ci

    # point estimates on full data
    est = {(ms.name, m): ms.fn(work_df, m) for ms in metrics_list for m in methods}

    # bootstrap arrays: metric -> method -> samples
    boot: Dict[str, Dict[str, np.ndarray]] = {
        ms.name: {m: np.empty(n_bootstrap, dtype=float) for m in methods}
        for ms in metrics_list
    }

    # one resample per replicate, reused for all metrics and methods (paired bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        sample = work_df.iloc[idx].reset_index(drop=True)
        for ms in metrics_list:
            for m in methods:
                boot[ms.name][m][b] = ms.fn(sample, m)

    rows = []
    for ms in metrics_list:
        ref_boot = boot[ms.name][reference_method]
        ref_est = est[(ms.name, reference_method)]

        for m in methods:
            m_boot = boot[ms.name][m]
            m_est = est[(ms.name, m)]

            row = {
                "method": m,
                "metric": ms.name,
                "estimate": float(m_est) if np.isfinite(m_est) else np.nan,
                "ci_lower": float(np.nanquantile(m_boot[np.isfinite(m_boot)], alpha/2)) if np.isfinite(m_boot).any() else np.nan,
                "ci_upper": float(np.nanquantile(m_boot[np.isfinite(m_boot)], 1-alpha/2)) if np.isfinite(m_boot).any() else np.nan,
                "diff_estimate": float(ref_est - m_est) if (np.isfinite(ref_est) and np.isfinite(m_est)) else np.nan,
                "n_bootstrap": int(n_bootstrap),
            }

            if m == reference_method:
                row.update({"diff_ci_lower": 0.0, "diff_ci_upper": 0.0, "p_value": 1.0})
            else:
                row.update(_paired_boot_summary(ref_boot, m_boot, alpha))

            rows.append(row)

    return pd.DataFrame(rows)


##############################
####### Deduplication ########
##############################

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


##############################
##### Build metric specs #####
##############################

def build_default_metric_specs(
    gold_col: str = "label",
    pvals_smaller_is_better: bool = True,
    early_R: float = 0.2,
) -> List[MetricSpec]:
    """
    Default metrics you likely want:
    - AUERC (weighted) full and early
    - AUORC (weighted log OR) full and early
    - AP and pAP@R
    """
    ascending = True if pvals_smaller_is_better else False

    def _auerc_full(df: pd.DataFrame, m: str) -> float:
        return auerc(df, m, gold_col=gold_col, ascending=ascending,
                     min_recall=0.0, max_recall=1.0, weighted=True, tieblocks=True)

    def _auerc_early(df: pd.DataFrame, m: str) -> float:
        return auerc(df, m, gold_col=gold_col, ascending=ascending,
                     min_recall=0.0, max_recall=early_R, weighted=True, tieblocks=True)

    def _auorc_full(df: pd.DataFrame, m: str) -> float:
        return auorc(df, m, gold_col=gold_col, ascending=ascending,
                     max_recall=1.0, epsilon=0.5, k_max="P",
                     weighted=True, log_or=True, include_partial_last=True)

    def _auorc_early(df: pd.DataFrame, m: str) -> float:
        return auorc(df, m, gold_col=gold_col, ascending=ascending,
                     max_recall=early_R, epsilon=0.5, k_max="P",
                     weighted=True, log_or=True, include_partial_last=True)

    def _ap(df: pd.DataFrame, m: str) -> float:
        return average_precision(df, m, gold_col=gold_col, score_transform=default_score_transform_pvalue)

    def _pap(df: pd.DataFrame, m: str) -> float:
        return partial_average_precision(df, m, R=early_R, gold_col=gold_col, normalize=True,
                                         score_transform=default_score_transform_pvalue)

    return [
        MetricSpec("AUERC", _auerc_full),
        MetricSpec(f"AUERC≥{early_R}", _auerc_early),
        MetricSpec("AUORC_log", _auorc_full),
        MetricSpec(f"AUORC_log≥{early_R}", _auorc_early),
        MetricSpec("AP", _ap),
        MetricSpec(f"pAP@{early_R}", _pap),
    ]


##############################
####### Example usage ########
##############################

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