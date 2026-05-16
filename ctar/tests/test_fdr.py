import numpy as np
import pandas as pd
import pytest
import ctar


def _mixture(n, pi1, mu_signal, rng):
    n_sig = int(n * pi1)
    stat = np.concatenate([rng.normal(mu_signal, 1, n_sig),
                           rng.normal(0, 1, n - n_sig)])
    truth = np.zeros(n, dtype=bool)
    truth[:n_sig] = True
    return stat, truth


def test_all_null_no_discoveries():
    rng = np.random.default_rng(0)
    stat = rng.normal(0, 1, 500)
    ctrl = rng.normal(0, 1, 5000)
    q, t_star, _, _ = ctar.method.empirical_fdr(stat, ctrl, target_fdr=0.1)
    assert t_star is None
    assert np.nanmedian(q) > 0.5


def test_clear_signal_low_fdr():
    rng = np.random.default_rng(1)
    stat = np.concatenate([rng.normal(5, 0.3, 100), rng.normal(0, 1, 400)])
    ctrl = rng.normal(0, 1, 5000)
    q, _, _, _ = ctar.method.empirical_fdr(stat, ctrl, target_fdr=0.1)
    assert np.all(q[stat > 4] < 0.05)


def test_q_values_monotone():
    rng = np.random.default_rng(2)
    stat, _ = _mixture(1000, 0.15, 3.0, rng)
    ctrl = rng.normal(0, 1, 10000)
    q, _, _, _ = ctar.method.empirical_fdr(stat, ctrl)
    order = np.argsort(stat)[::-1]
    assert np.all(np.diff(q[order]) >= -1e-9)


def test_nan_passthrough():
    rng = np.random.default_rng(3)
    stat = rng.normal(0, 1, 200).astype(float)
    nan_idx = [0, 50, 199]
    stat[nan_idx] = np.nan
    q, _, _, _ = ctar.method.empirical_fdr(stat, rng.normal(0, 1, 2000))
    assert np.all(np.isnan(q[nan_idx]))
    assert np.all(np.isfinite(q[~np.isnan(stat)]))


def test_laplace_no_zero_qvals():
    rng = np.random.default_rng(4)
    stat = np.concatenate([rng.normal(6, 0.2, 50), rng.normal(0, 1, 950)])
    ctrl = rng.normal(0, 1, 1000)
    q, _, _, _ = ctar.method.empirical_fdr(stat, ctrl, tail='laplace')
    assert np.all(q[np.isfinite(q)] > 0)


def test_laplace_more_conservative_than_raw():
    # (null_counts+1)/(B+1) > null_counts/B for all finite thresholds,
    # so Laplace should produce a higher median q-value than the uncorrected form.
    rng = np.random.default_rng(5)
    stat, _ = _mixture(1000, 0.2, 3.0, rng)
    ctrl = rng.normal(0, 1, 5000)

    q_laplace, _, _, _ = ctar.method.empirical_fdr(stat, ctrl, tail='laplace')

    # Build raw (no +1 correction) q-values manually for comparison
    thresholds = np.sort(np.unique(stat))[::-1]
    ctrl_asc = np.sort(ctrl)
    stat_asc = np.sort(stat)
    n_disc = len(stat_asc) - np.searchsorted(stat_asc, thresholds, side='left')
    null_counts = len(ctrl_asc) - np.searchsorted(ctrl_asc, thresholds, side='left')
    fdr_raw = np.minimum((null_counts / len(ctrl)) / (n_disc / len(stat)), 1.0)
    fdr_raw = np.minimum.accumulate(fdr_raw[::-1])[::-1]
    idx = np.searchsorted(thresholds[::-1], stat, side='left').clip(0, len(fdr_raw) - 1)
    q_raw = fdr_raw[::-1][idx]

    assert np.nanmedian(q_laplace) >= np.nanmedian(q_raw) - 1e-6


def test_fdr_calibration():
    # Over many seeds, the true FDR at q<=0.1 should average <=0.15.
    true_fdrs = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        stat, truth = _mixture(1000, 0.2, 3.0, rng)
        ctrl = rng.normal(0, 1, 10000)
        q, _, _, _ = ctar.method.empirical_fdr(stat, ctrl, target_fdr=0.1)
        rej = q <= 0.1
        if rej.sum() > 0:
            true_fdrs.append((~truth[rej]).sum() / rej.sum())
    assert np.mean(true_fdrs) <= 0.15


def test_pi0_conservative():
    rng = np.random.default_rng(6)
    stat, _ = _mixture(500, 0.3, 3.0, rng)
    ctrl = rng.normal(0, 1, 5000)
    q_cons, _, _, _ = ctar.method.empirical_fdr(stat, ctrl, pi0=1.0)
    q_opt,  _, _, _ = ctar.method.empirical_fdr(stat, ctrl, pi0=0.7)
    assert np.nanmean(q_cons) >= np.nanmean(q_opt) - 1e-6


def test_b_override_inflates_fdr():
    # Passing B smaller than len(ctrl) overestimates P(null>=t), inflating FDR.
    rng = np.random.default_rng(7)
    stat, _ = _mixture(500, 0.3, 3.0, rng)
    ctrl = rng.normal(0, 1, 5000)
    q_full,  _, _, _ = ctar.method.empirical_fdr(stat, ctrl, B=5000)
    q_small, _, _, _ = ctar.method.empirical_fdr(stat, ctrl, B=500)
    assert np.nanmean(q_small) >= np.nanmean(q_full) - 1e-6


def _make_binned_inputs(rng, n_bins=3, n_per_bin=200, b_ctrl=500):
    rows = []
    ctrl_dic = {}
    for k in range(n_bins):
        stat, _ = _mixture(n_per_bin, 0.2, 3.0, rng)
        bin_id = str(k)
        for s in stat:
            rows.append({'stat': s, 'bin': bin_id})
        ctrl_dic[bin_id] = rng.normal(0, 1, b_ctrl)
    return pd.DataFrame(rows), ctrl_dic


def test_binned_fdr_returns_dataframe_with_qval():
    rng = np.random.default_rng(20)
    df, ctrl_dic = _make_binned_inputs(rng)
    out = ctar.method.binned_empirical_fdr(
        df, ctrl_dic, stat_col='stat', bin_col='bin', qval_col='q_value'
    )
    assert isinstance(out, pd.DataFrame)
    assert 'q_value' in out.columns
    assert len(out) == len(df)


def test_binned_fdr_qvals_match_per_bin():
    # q-values from binned_empirical_fdr must match calling empirical_fdr
    # independently on each bin.
    rng = np.random.default_rng(21)
    df, ctrl_dic = _make_binned_inputs(rng, n_bins=2)
    out = ctar.method.binned_empirical_fdr(
        df, ctrl_dic, stat_col='stat', bin_col='bin', qval_col='q_value'
    )
    for bin_id, group in df.groupby('bin'):
        q_expected, _, _, _ = ctar.method.empirical_fdr(
            group['stat'].values, ctrl_dic[bin_id]
        )
        q_actual = out.loc[group.index, 'q_value'].values
        np.testing.assert_allclose(q_actual, q_expected, rtol=1e-6)


def test_binned_fdr_no_zero_qvals_laplace():
    rng = np.random.default_rng(22)
    df, ctrl_dic = _make_binned_inputs(rng)
    out = ctar.method.binned_empirical_fdr(
        df, ctrl_dic, stat_col='stat', bin_col='bin',
        qval_col='q_value', tail='laplace'
    )
    assert (out['q_value'].dropna() == 0).sum() == 0


def test_binned_fdr_missing_bin_raises():
    rng = np.random.default_rng(23)
    df, ctrl_dic = _make_binned_inputs(rng, n_bins=2)
    ctrl_dic_incomplete = {k: v for i, (k, v) in enumerate(ctrl_dic.items()) if i > 0}
    with pytest.raises(KeyError):
        ctar.method.binned_empirical_fdr(
            df, ctrl_dic_incomplete, stat_col='stat', bin_col='bin'
        )
