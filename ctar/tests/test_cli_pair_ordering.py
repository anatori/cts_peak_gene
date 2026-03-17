import argparse
import pickle

import numpy as np
import pandas as pd

import CLI_ctar
import ctar


BIN_CONFIG = "5.5.5.5.2"
COMBINED_BIN_COL = "combined_bin_5.5.5.5"


def make_args(results_path):
    return argparse.Namespace(
        job="compute_pval",
        job_id="test",
        multiome_file=None,
        batch_size="10",
        links_file=None,
        target_path=None,
        results_path=str(results_path),
        genome_file=None,
        binning_config=BIN_CONFIG,
        binning_type="mean_gc",
        pybedtools_path=None,
        covar_file=None,
        array_idx=None,
        flag_se="False",
        flag_ll="False",
        n_cores=None,
    )


def write_binned_results_fixture(results_path, cis_links_df, cis_idx_dic, cis_coeff_dic, ctrl_coeff_dic):
    cis_links_df.to_csv(results_path / "cis_links_df.csv")

    with open(results_path / "cis_idx_dic.pkl", "wb") as handle:
        pickle.dump(cis_idx_dic, handle)
    with open(results_path / "cis_coeff_dic.pkl", "wb") as handle:
        pickle.dump(cis_coeff_dic, handle)
    with open(results_path / "ctrl_coeff_dic.pkl", "wb") as handle:
        pickle.dump(ctrl_coeff_dic, handle)


def test_compute_pval_preserves_pair_order_for_binned_dicts(tmp_path, monkeypatch):
    results_path = tmp_path / "results"
    results_path.mkdir()

    cis_links_df = pd.DataFrame(
        {
            "peak": ["peak_0", "peak_1", "peak_2", "peak_3", "peak_4"],
            "gene": ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"],
            COMBINED_BIN_COL: ["bin_b", "bin_a", "bin_b", "bin_a", "bin_b"],
        },
        index=["pair_0", "pair_1", "pair_2", "pair_3", "pair_4"],
    )

    cis_idx_dic = {
        "bin_b": [0, 2, 4],
        "bin_a": [1, 3],
    }
    cis_coeff_dic = {
        "bin_b": np.array([0.11, 0.22, 0.33]),
        "bin_a": np.array([0.44, 0.55]),
    }
    ctrl_coeff_dic = {
        "bin_b": np.array([9.0, 8.0, 7.0]),
        "bin_a": np.array([6.0, 5.0]),
    }

    write_binned_results_fixture(
        results_path,
        cis_links_df,
        cis_idx_dic,
        cis_coeff_dic,
        ctrl_coeff_dic,
    )

    def fake_binned_mcpval(cis_pairs_dic, ctrl_pairs_dic, b=None, flag_interp=False, flag_zp=False):
        del ctrl_pairs_dic, b, flag_interp, flag_zp
        return (
            {key: np.asarray(value) + 100.0 for key, value in cis_pairs_dic.items()},
            {key: np.asarray(value) + 200.0 for key, value in cis_pairs_dic.items()},
        )

    monkeypatch.setattr(ctar.method, "binned_mcpval", fake_binned_mcpval)

    CLI_ctar.main(make_args(results_path))

    output_df = pd.read_csv(results_path / "cis_links_df.csv", index_col=0)

    expected_mcpval = np.array([100.11, 100.44, 100.22, 100.55, 100.33])
    expected_ppval = np.array([200.11, 200.44, 200.22, 200.55, 200.33])

    assert np.allclose(output_df[f"{BIN_CONFIG}_mcpval"].to_numpy(), expected_mcpval)
    assert np.allclose(output_df[f"{BIN_CONFIG}_ppval"].to_numpy(), expected_ppval)


def test_compute_pval_preserves_pair_order_for_se_binned_dicts(tmp_path, monkeypatch):
    results_path = tmp_path / "results"
    results_path.mkdir()

    cis_links_df = pd.DataFrame(
        {
            "peak": ["peak_0", "peak_1", "peak_2", "peak_3", "peak_4"],
            "gene": ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"],
            COMBINED_BIN_COL: ["bin_b", "bin_a", "bin_b", "bin_a", "bin_b"],
        },
        index=["pair_0", "pair_1", "pair_2", "pair_3", "pair_4"],
    )

    cis_idx_dic = {
        "bin_b": [0, 2, 4],
        "bin_a": [1, 3],
    }
    cis_coeff_dic = {
        "bin_b": np.array([[1.0, 11.0], [2.0, 12.0], [4.0, 16.0]]),
        "bin_a": np.array([[5.0, 15.0], [10.0, 20.0]]),
    }
    ctrl_coeff_dic = {
        "bin_b": np.array([[2.0, 21.0], [4.0, 22.0], [8.0, 24.0]]),
        "bin_a": np.array([[3.0, 23.0], [6.0, 26.0]]),
    }

    write_binned_results_fixture(
        results_path,
        cis_links_df,
        cis_idx_dic,
        cis_coeff_dic,
        ctrl_coeff_dic,
    )

    def fake_binned_mcpval(cis_pairs_dic, ctrl_pairs_dic, b=None, flag_interp=False, flag_zp=False):
        del ctrl_pairs_dic, b, flag_interp, flag_zp
        return (
            {key: np.asarray(value) + 1000.0 for key, value in cis_pairs_dic.items()},
            {key: np.asarray(value) + 2000.0 for key, value in cis_pairs_dic.items()},
        )

    monkeypatch.setattr(ctar.method, "binned_mcpval", fake_binned_mcpval)

    CLI_ctar.main(make_args(results_path))

    output_df = pd.read_csv(results_path / "cis_links_df.csv", index_col=0)

    expected_regular = np.array([1011.0, 1015.0, 1012.0, 1020.0, 1016.0])
    expected_pooled = np.array([2011.0, 2015.0, 2012.0, 2020.0, 2016.0])
    expected_studentized = np.array([1011.0, 1003.0, 1006.0, 1002.0, 1004.0])
    expected_studentized_pooled = np.array([2011.0, 2003.0, 2006.0, 2002.0, 2004.0])

    assert np.allclose(output_df[f"{BIN_CONFIG}_mcpval"].to_numpy(), expected_regular)
    assert np.allclose(output_df[f"{BIN_CONFIG}_ppval"].to_numpy(), expected_pooled)
    assert np.allclose(output_df[f"{BIN_CONFIG}_mcpval_z"].to_numpy(), expected_studentized)
    assert np.allclose(output_df[f"{BIN_CONFIG}_ppval_z"].to_numpy(), expected_studentized_pooled)
