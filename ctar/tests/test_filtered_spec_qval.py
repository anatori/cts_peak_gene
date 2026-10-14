import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sps
from statsmodels.stats.multitest import multipletests

from ctar.celltype_specificity import celltype_tables, coexp_context, coexp_specificity
from ctar.method import add_filtered_spec_qval
from ctar.parallel import multiprocess_joint_entropy_specificity, multiprocess_link_specificity


def test_filtered_spec_qval_is_bh_on_selected_and_nan_elsewhere(capsys):
    df = pd.DataFrame({
        "q_value_fixed_dist_rmean_rvar": [0.01, 0.5, 0.1, 0.19, 0.2, np.nan],
        "q_value": [0.9] * 6,
        "mcp": [0.02, 0.001, 0.5, 0.01, 0.01, 0.01],
    })
    out = add_filtered_spec_qval(df.copy(), "mcp", "filt", "coexp_log1p", 0.2)
    selected = np.array([True, False, True, True, False, False])   # strict < 0.2; NaN association q not selected
    assert out.loc[~selected, "filt"].isna().all()
    np.testing.assert_allclose(out.loc[selected, "filt"], multipletests(df.loc[selected, "mcp"], method="fdr_bh")[1])
    assert "WARNING" not in capsys.readouterr().out


def test_filtered_spec_qval_falls_back_to_q_value_and_warns_for_joint(capsys):
    df = pd.DataFrame({"q_value": [0.05, 0.3], "mcp": [0.01, 0.01]})
    out = add_filtered_spec_qval(df.copy(), "mcp", "filt", "joint", 0.2)
    assert out["filt"].notna().tolist() == [True, False]
    assert "WARNING" in capsys.readouterr().out


def test_filtered_spec_qval_skips_without_association_q():
    df = pd.DataFrame({"mcp": [0.01, 0.2]})
    out = add_filtered_spec_qval(df.copy(), "mcp", "filt", "coexp_log1p", 0.2)
    assert "filt" not in out.columns


def _adata(counts, labels):
    obs = pd.DataFrame({"celltype": labels}, index=[f"c{i}" for i in range(counts.shape[0])])
    return ad.AnnData(X=sps.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"f{i}" for i in range(counts.shape[1])]))


def test_multiprocess_link_specificity_dispatches():
    labels = ["a", "a", "b", "b", None]
    rna = _adata(np.array([[4, 0, 1], [2, 0, 3], [0, 5, 0], [1, 7, 2], [9, 9, 9]]), labels)
    atac = _adata(np.array([[1, 0], [3, 2], [0, 4], [2, 6], [5, 5]]), labels)
    links = np.array([[0, 0], [1, 1], [0, 2]])
    rna_table, atac_table = celltype_tables(rna, atac, layer=None)
    context = coexp_context(rna, atac, layer=None)

    spec, dom = multiprocess_link_specificity(links, "joint", rna_table, atac_table, n_workers=1)
    assert dom is None
    np.testing.assert_allclose(spec, multiprocess_joint_entropy_specificity(links, rna_table, atac_table, n_workers=1))

    spec, dom = multiprocess_link_specificity(links, "coexp_log1p", rna_table, atac_table, context, n_workers=1)
    expected_spec, expected_dom = coexp_specificity(context, links[:, 1], links[:, 0])
    np.testing.assert_allclose(spec, expected_spec)
    np.testing.assert_array_equal(dom, expected_dom)
