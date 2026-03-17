import argparse
import pickle

import numpy as np
import pandas as pd

import CLI_ctar


BINNED_CONFIG = "1.1.1.1.2"
COMBINED_BIN_COL = "combined_bin_1.1.1.1"


class FakeAnnData:
    def __init__(self, X, var):
        self.X = np.asarray(X)
        self.var = var.copy()
        self.layers = {}

    @property
    def shape(self):
        return self.X.shape

    @property
    def var_names(self):
        return self.var.index

    @var_names.setter
    def var_names(self, values):
        self.var.index = pd.Index(values)

    def copy(self):
        copied = FakeAnnData(self.X.copy(), self.var.copy())
        copied.layers = {key: np.array(value, copy=True) for key, value in self.layers.items()}
        return copied

    def __getitem__(self, key):
        rows, cols = key
        if rows != slice(None):
            raise NotImplementedError("Tests only support full-row slicing.")

        cols_arr = np.asarray(cols)
        if cols_arr.dtype == bool:
            col_idx = np.flatnonzero(cols_arr)
        else:
            col_idx = cols_arr

        sliced = FakeAnnData(self.X[:, col_idx], self.var.iloc[col_idx].copy())
        sliced.layers = {
            name: np.asarray(value)[:, col_idx].copy()
            for name, value in self.layers.items()
        }
        return sliced


def make_args(job, target_path, results_path=None, flag_se=False, job_id="smoke"):
    return argparse.Namespace(
        job=job,
        job_id=job_id,
        multiome_file="toy.h5ad",
        batch_size="10",
        links_file=None,
        target_path=str(target_path),
        results_path=None if results_path is None else str(results_path),
        genome_file="hg38",
        binning_config=BINNED_CONFIG,
        binning_type="mean_gc",
        pybedtools_path="bedtools",
        covar_file=None,
        array_idx=None,
        flag_se="True" if flag_se else "False",
        flag_ll="False",
        n_cores="1",
    )


def make_fake_multiome():
    X = np.array(
        [
            [1, 0, 3, 2],
            [0, 2, 1, 4],
            [1, 1, 0, 5],
        ],
        dtype=float,
    )
    var = pd.DataFrame(
        {
            "feature_types": ["ATAC", "ATAC", "GEX", "GEX"],
            "gene_id": ["", "", "ENSG000001", "ENSG000002"],
        },
        index=["chr1:10-20", "chr1:30-40", "gene_a", "gene_b"],
    )
    return FakeAnnData(X, var)


def fake_cis_links_df():
    return pd.DataFrame(
        {
            "peak": ["chr1:10-20", "chr1:30-40", "chr1:10-20"],
            "gene": ["ENSG000001", "ENSG000002", "ENSG000002"],
            "index_x": [0, 1, 0],
            "index_y": [0, 1, 1],
        },
        index=["pair_0", "pair_1", "pair_2"],
    )


def fake_atac_bins_df():
    return pd.DataFrame(
        {
            "peak": ["chr1:10-20", "chr1:30-40"],
            "mean_gc_bin": [0, 1],
        }
    )


def fake_rna_bins_df():
    return pd.DataFrame(
        {
            "gene": ["ENSG000001", "ENSG000002"],
            "mean_var_bin": [0, 1],
        }
    )


def fake_ctrl_links_dic():
    return {
        "0_0": np.array([[1, 0], [0, 0]]),
        "1_1": np.array([[0, 1], [1, 1]]),
        "0_1": np.array([[1, 1], [0, 1]]),
    }


def fake_get_bins(adata, num_bins, type, col, layer, genome_file=None):
    del adata, num_bins, layer, genome_file
    if col == "peak" and type == "mean_gc":
        return fake_atac_bins_df().copy()
    if col == "gene" and type == "mean_var":
        return fake_rna_bins_df().copy()
    raise AssertionError(f"Unexpected get_bins request: col={col}, type={type}")


def patch_common(monkeypatch):
    monkeypatch.setattr(CLI_ctar.ctar.util, "get_cli_head", lambda: "")
    monkeypatch.setattr(CLI_ctar.pybedtools.helpers, "set_bedtools_path", lambda path: None)
    monkeypatch.setattr(CLI_ctar.ad, "read_h5ad", lambda path: make_fake_multiome())
    monkeypatch.setattr(CLI_ctar.ctar.data_loader, "get_gene_coords", lambda df, add_tss=False: df.copy())
    monkeypatch.setattr(CLI_ctar.ctar.data_loader, "peak_to_gene", lambda *args, **kwargs: fake_cis_links_df().copy())
    monkeypatch.setattr(CLI_ctar.ctar.method, "create_ctrl_pairs", lambda *args, **kwargs: (fake_ctrl_links_dic().copy(), fake_atac_bins_df().copy(), fake_rna_bins_df().copy()))
    monkeypatch.setattr(CLI_ctar.ctar.method, "get_bins", fake_get_bins)


def fake_multiprocess_poisson_irls(links_dict, flag_se=False, **kwargs):
    result = {}
    for key, arr in links_dict.items():
        arr = np.asarray(arr)
        beta = arr[:, 0].astype(float) + arr[:, 1].astype(float) / 10.0
        if flag_se:
            result[key] = np.column_stack([np.full(arr.shape[0], 0.5), beta])
        else:
            result[key] = beta
    return result


def patch_parallel(monkeypatch):
    monkeypatch.setattr(CLI_ctar.ctar.parallel, "multiprocess_poisson_irls", fake_multiprocess_poisson_irls)
    monkeypatch.setattr(
        CLI_ctar.ctar.parallel,
        "multiprocess_poisson_irls_chunked",
        fake_multiprocess_poisson_irls,
        raising=False,
    )


def seed_results_path(results_path):
    results_path.mkdir(parents=True, exist_ok=True)
    cis_links_df = fake_cis_links_df().copy()
    cis_links_df["atac_mean_gc_bin_1.1"] = [0, 1, 0]
    cis_links_df["rna_mean_var_bin_1.1"] = [0, 1, 1]
    cis_links_df[COMBINED_BIN_COL] = ["0_0", "1_1", "0_1"]
    cis_links_df.to_csv(results_path / "cis_links_df.csv")


def test_generate_links_job_binned_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)

    args = make_args("generate_links", target_path=tmp_path, job_id="generate_links_binned")
    CLI_ctar.main(args)

    results_dir = tmp_path / "generate_links_binned_results"
    assert (results_dir / "cis_idx_dic.pkl").exists()
    assert (results_dir / "ctrl_links_dic.pkl").exists()
    output_df = pd.read_csv(results_dir / "cis_links_df.csv", index_col=0)
    assert COMBINED_BIN_COL in output_df.columns


def test_generate_controls_job_binned_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    results_path = tmp_path / "generate_controls_binned_results"
    seed_results_path(results_path)

    args = make_args("generate_controls", target_path=tmp_path, results_path=results_path, job_id="generate_controls_binned")
    CLI_ctar.main(args)

    assert (results_path / "ctrl_links_dic.pkl").exists()
    output_df = pd.read_csv(results_path / "cis_links_df.csv", index_col=0)
    assert len(output_df) == 3


def test_compute_cis_only_job_binned_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    patch_parallel(monkeypatch)
    results_path = tmp_path / "compute_cis_only_binned_results"
    seed_results_path(results_path)

    args = make_args("compute_cis_only", target_path=tmp_path, results_path=results_path, flag_se=True, job_id="compute_cis_only_binned")
    CLI_ctar.main(args)

    with open(results_path / "cis_coeff_dic.pkl", "rb") as handle:
        cis_coeff_dic = pickle.load(handle)
    with open(results_path / "cis_idx_dic.pkl", "rb") as handle:
        cis_idx_dic = pickle.load(handle)

    assert set(cis_coeff_dic.keys()) == {"0_0", "0_1", "1_1"}
    assert set(cis_idx_dic.keys()) == {"0_0", "0_1", "1_1"}


def test_compute_ctrl_only_job_binned_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    patch_parallel(monkeypatch)
    results_path = tmp_path / "compute_ctrl_only_binned_results"
    seed_results_path(results_path)

    args = make_args("compute_ctrl_only", target_path=tmp_path, results_path=results_path, flag_se=True, job_id="compute_ctrl_only_binned")
    CLI_ctar.main(args)

    with open(results_path / "ctrl_coeff_dic.pkl", "rb") as handle:
        ctrl_coeff_dic = pickle.load(handle)

    assert set(ctrl_coeff_dic.keys()) == {"0_0", "0_1", "1_1"}


def test_compute_pval_job_binned_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    results_path = tmp_path / "compute_pval_binned_results"
    seed_results_path(results_path)

    cis_coeff_dic = {
        "0_0": np.array([0.2]),
        "1_1": np.array([0.4]),
        "0_1": np.array([0.1]),
    }
    cis_idx_dic = {
        "0_0": [0],
        "1_1": [1],
        "0_1": [2],
    }
    ctrl_coeff_dic = {
        "0_0": np.array([0.0, 0.3]),
        "1_1": np.array([0.2, 0.5]),
        "0_1": np.array([-0.1, 0.2]),
    }

    with open(results_path / "cis_coeff_dic.pkl", "wb") as handle:
        pickle.dump(cis_coeff_dic, handle)
    with open(results_path / "cis_idx_dic.pkl", "wb") as handle:
        pickle.dump(cis_idx_dic, handle)
    with open(results_path / "ctrl_coeff_dic.pkl", "wb") as handle:
        pickle.dump(ctrl_coeff_dic, handle)

    args = make_args("compute_pval", target_path=tmp_path, results_path=results_path, job_id="compute_pval_binned")
    CLI_ctar.main(args)

    output_df = pd.read_csv(results_path / "cis_links_df.csv", index_col=0)
    assert f"{BINNED_CONFIG}_mcpval" in output_df.columns
    assert f"{BINNED_CONFIG}_ppval" in output_df.columns


def test_compute_ctar_job_binned_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    patch_parallel(monkeypatch)

    args = make_args("compute_ctar", target_path=tmp_path, job_id="compute_ctar_binned")
    CLI_ctar.main(args)

    results_dir = tmp_path / "compute_ctar_binned_results"
    with open(results_dir / "cis_coeff_dic.pkl", "rb") as handle:
        cis_coeff_dic = pickle.load(handle)
    with open(results_dir / "ctrl_coeff_dic.pkl", "rb") as handle:
        ctrl_coeff_dic = pickle.load(handle)
    output_df = pd.read_csv(results_dir / "cis_links_df.csv", index_col=0)

    assert set(cis_coeff_dic.keys()) == {"0_0", "0_1", "1_1"}
    assert set(ctrl_coeff_dic.keys()) == {"0_0", "0_1", "1_1"}
    assert "poissonb" in output_df.columns
