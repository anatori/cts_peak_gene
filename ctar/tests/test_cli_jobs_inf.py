import argparse
import numpy as np
import pandas as pd

import CLI_ctar


INF_BIN_CONFIG = "1.1.inf.inf.2"


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
        binning_config=INF_BIN_CONFIG,
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
            "peak": ["chr1:10-20", "chr1:30-40"],
            "gene": ["ENSG000001", "ENSG000002"],
        },
        index=["pair_0", "pair_1"],
    )


def expected_ctrl_peaks():
    return np.array([[1, 0], [0, 1]], dtype=int)


def patch_common(monkeypatch):
    monkeypatch.setattr(CLI_ctar.ctar.util, "get_cli_head", lambda: "")
    monkeypatch.setattr(CLI_ctar.pybedtools.helpers, "set_bedtools_path", lambda path: None)
    monkeypatch.setattr(CLI_ctar.ad, "read_h5ad", lambda path: make_fake_multiome())
    monkeypatch.setattr(CLI_ctar.ctar.data_loader, "get_gene_coords", lambda df, add_tss=False: df.copy())
    monkeypatch.setattr(CLI_ctar.ctar.data_loader, "peak_to_gene", lambda *args, **kwargs: fake_cis_links_df().copy())
    monkeypatch.setattr(CLI_ctar.ctar.method, "create_ctrl_peaks", lambda *args, **kwargs: expected_ctrl_peaks().copy())


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


def seed_results_path(results_path, include_ctrl_peaks=False):
    results_path.mkdir(parents=True, exist_ok=True)
    fake_cis_links_df().to_csv(results_path / "cis_links_df.csv")
    if include_ctrl_peaks:
        np.save(results_path / "ctrl_peaks.npy", expected_ctrl_peaks())


def test_generate_links_job_inf_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)

    args = make_args("generate_links", target_path=tmp_path, job_id="generate_links")
    CLI_ctar.main(args)

    results_dir = tmp_path / "generate_links_results"
    assert np.array_equal(np.load(results_dir / "ctrl_peaks.npy"), expected_ctrl_peaks())
    output_df = pd.read_csv(results_dir / "cis_links_df.csv", index_col=0)
    assert output_df.index.tolist() == ["pair_0", "pair_1"]


def test_generate_controls_job_inf_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    results_path = tmp_path / "generate_controls_results"
    seed_results_path(results_path)

    args = make_args("generate_controls", target_path=tmp_path, results_path=results_path, job_id="generate_controls")
    CLI_ctar.main(args)

    assert np.array_equal(np.load(results_path / "ctrl_peaks.npy"), expected_ctrl_peaks())
    output_df = pd.read_csv(results_path / "cis_links_df.csv", index_col=0)
    assert output_df.index.tolist() == ["pair_0", "pair_1"]


def test_compute_cis_only_job_inf_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    monkeypatch.setattr(CLI_ctar.ctar.parallel, "multiprocess_poisson_irls", fake_multiprocess_poisson_irls)
    results_path = tmp_path / "compute_cis_only_results"
    seed_results_path(results_path)

    args = make_args("compute_cis_only", target_path=tmp_path, results_path=results_path, flag_se=True, job_id="compute_cis_only")
    CLI_ctar.main(args)

    cis_coeff = np.load(results_path / "cis_coeff.npy")
    link_ids = np.load(results_path / "cis_link_ids.npy", allow_pickle=True)

    expected = np.array([[0.5, 0.0], [0.5, 1.1]], dtype=np.float32)
    assert np.allclose(cis_coeff, expected)
    assert link_ids.tolist() == ["pair_0", "pair_1"]


def test_compute_ctrl_only_job_inf_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    monkeypatch.setattr(CLI_ctar.ctar.parallel, "multiprocess_poisson_irls", fake_multiprocess_poisson_irls)
    results_path = tmp_path / "compute_ctrl_only_results"
    seed_results_path(results_path, include_ctrl_peaks=True)

    args = make_args("compute_ctrl_only", target_path=tmp_path, results_path=results_path, flag_se=True, job_id="compute_ctrl_only")
    CLI_ctar.main(args)

    ctrl_coeff = np.load(results_path / "ctrl_coeff.npy")
    link_ids = np.load(results_path / "ctrl_link_ids.npy", allow_pickle=True)

    expected = np.array(
        [
            [[0.5, 1.0], [0.5, 0.0]],
            [[0.5, 0.1], [0.5, 1.1]],
        ],
        dtype=np.float32,
    )
    assert np.allclose(ctrl_coeff, expected)
    assert link_ids.tolist() == ["pair_0", "pair_1"]


def test_compute_pval_job_inf_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    results_path = tmp_path / "compute_pval_results"
    seed_results_path(results_path)

    cis_coeff = np.array([[0.5, 2.0], [1.0, 0.5]], dtype=float)
    ctrl_coeff = np.array(
        [
            [[0.5, 1.0], [0.5, 1.5]],
            [[1.0, 0.0], [1.0, 1.0]],
        ],
        dtype=float,
    )
    np.save(results_path / "cis_coeff.npy", cis_coeff)
    np.save(results_path / "ctrl_coeff.npy", ctrl_coeff)

    args = make_args("compute_pval", target_path=tmp_path, results_path=results_path, job_id="compute_pval")
    CLI_ctar.main(args)

    output_df = pd.read_csv(results_path / "cis_links_df.csv", index_col=0)
    assert "poissonb" in output_df.columns
    assert "poissonb_se" in output_df.columns
    assert f"{INF_BIN_CONFIG}_mcpval" in output_df.columns
    assert f"{INF_BIN_CONFIG}_ppval" in output_df.columns


def test_compute_ctar_job_inf_mode(tmp_path, monkeypatch):
    patch_common(monkeypatch)
    monkeypatch.setattr(CLI_ctar.ctar.parallel, "multiprocess_poisson_irls", fake_multiprocess_poisson_irls)

    args = make_args("compute_ctar", target_path=tmp_path, flag_se=True, job_id="compute_ctar")
    CLI_ctar.main(args)

    results_dir = tmp_path / "compute_ctar_results"
    cis_coeff = np.load(results_dir / "cis_coeff.npy")
    ctrl_coeff = np.load(results_dir / "ctrl_coeff.npy")
    output_df = pd.read_csv(results_dir / "cis_links_df.csv", index_col=0)

    assert np.array_equal(np.load(results_dir / "ctrl_peaks.npy"), expected_ctrl_peaks())
    assert cis_coeff.shape == (2, 2)
    assert ctrl_coeff.shape == (2, 2, 2)
    assert "poissonb" in output_df.columns
    assert "poissonb_se" in output_df.columns
