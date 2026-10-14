import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from ctar.celltype_specificity import (
    celltype_tables, dominant_celltypes, entropy_specificity, joint_entropy_specificity, marginal_specificity,
)
from ctar.parallel import multiprocess_joint_entropy_specificity

COUNTS = np.array([[4, 0, 1], [2, 0, 3], [0, 5, 0], [1, 7, 2], [9, 9, 9]], dtype=np.int32)
LABELS = ["a", "a", "b", "b", None]


def _adata(counts=COUNTS, dense=False):
    obs = pd.DataFrame({"celltype": LABELS}, index=[f"c{i}" for i in range(counts.shape[0])])
    matrix = counts.copy() if dense else sps.csr_matrix(counts)
    return ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(counts.shape[1])]))


def test_pseudobulk_table_matches_brute_force():
    adata = _adata()
    rna_table, atac_table = celltype_tables(adata, adata, layer=None, method="pseudobulk")
    summed = np.array([[6, 0, 4], [1, 12, 2]], dtype=float)   # NaN-labelled cell dropped
    totals = summed.sum(axis=1)
    expected = (summed / totals[:, None] + 1.0 / totals.mean()).T
    assert list(rna_table.columns) == ["a", "b"] and list(rna_table.index) == ["g0", "g1", "g2"]
    np.testing.assert_allclose(rna_table.to_numpy(), expected)
    np.testing.assert_allclose(atac_table.to_numpy(), expected)


def test_mean_table_matches_brute_force():
    adata = _adata()
    rna_table, _ = celltype_tables(adata, adata, layer=None, method="mean")
    depth = COUNTS.sum(axis=1).astype(float)                   # depth over ALL cells, NaN label included
    normalized = COUNTS / depth[:, None] * 1e4
    means = np.vstack([normalized[[0, 1]].mean(axis=0), normalized[[2, 3]].mean(axis=0)]).T
    pseudocount = (1e4 / np.median(depth)) / 4                 # 4 labelled cells
    np.testing.assert_allclose(rna_table.to_numpy(), means + pseudocount, rtol=1e-5)


def test_dense_input_and_groups_override():
    adata = _adata()
    dense = _adata(dense=True)
    from_sparse, _ = celltype_tables(adata, adata, layer=None)
    from_dense, _ = celltype_tables(dense, dense, layer=None)
    from_groups, _ = celltype_tables(adata, adata, layer=None, groups=np.array(LABELS, dtype=object))
    np.testing.assert_allclose(from_sparse.to_numpy(), from_dense.to_numpy())
    np.testing.assert_allclose(from_sparse.to_numpy(), from_groups.to_numpy())


def test_celltype_tables_validation():
    adata = _adata()
    with pytest.raises(ValueError, match="method"):
        celltype_tables(adata, adata, layer=None, method="nope")
    with pytest.raises(ValueError, match="same order"):
        celltype_tables(adata, adata[::-1].copy(), layer=None)
    with pytest.raises(ValueError, match="not found"):
        celltype_tables(adata, adata, celltype_col="missing", layer=None)


def test_joint_entropy_specificity_matches_entropy_of_joint_vector():
    adata = _adata()
    rna_table, atac_table = celltype_tables(adata, adata, layer=None)
    gene_idx, peak_idx = np.array([0, 1, 2, 1]), np.array([1, 2, 0, 1])
    specificity = joint_entropy_specificity(rna_table, atac_table, gene_idx, peak_idx, chunk_size=3)
    joint = rna_table.to_numpy()[gene_idx] * atac_table.to_numpy()[peak_idx]
    np.testing.assert_allclose(specificity, entropy_specificity(joint, eps=0.0), atol=1e-12)


def test_multiprocess_dict_and_array_agree():
    adata = _adata()
    rna_table, atac_table = celltype_tables(adata, adata, layer=None)
    links = np.array([[0, 1], [1, 2], [2, 0], [1, 1], [0, 0]])   # (atac_idx, rna_idx)
    from_array = multiprocess_joint_entropy_specificity(links, rna_table, atac_table, array_chunk_size=2, n_workers=2)
    from_dict = multiprocess_joint_entropy_specificity({"x": links[:3], "y": links[3:]}, rna_table, atac_table, n_workers=2)
    direct = joint_entropy_specificity(rna_table, atac_table, links[:, 1], links[:, 0])
    np.testing.assert_allclose(from_array, direct)
    np.testing.assert_allclose(np.concatenate([from_dict["x"], from_dict["y"]]), direct)
    assert multiprocess_joint_entropy_specificity(np.empty((0, 2), dtype=int), rna_table, atac_table).shape == (0,)


def test_marginal_specificity_matches_per_feature_entropy():
    rna = _adata()
    atac = _adata(counts=COUNTS[:, [2, 0, 1]])   # differently ordered features so the two tables differ
    rna_table, atac_table = celltype_tables(rna, atac, layer=None)
    gene_specificity, peak_specificity = marginal_specificity(rna_table, atac_table)
    for table, specificity in ((rna_table, gene_specificity), (atac_table, peak_specificity)):
        assert specificity.shape == (table.shape[0],)
        for row, value in zip(table.to_numpy(), specificity):
            shares = row / row.sum()
            expected = 1.0 + (shares * np.log(shares)).sum() / np.log(len(shares))
            assert value == pytest.approx(expected)
    assert not np.allclose(gene_specificity, peak_specificity)


def test_marginal_specificity_flat_and_concentrated_profiles():
    flat = pd.DataFrame([[1.0, 1.0, 1.0]])
    concentrated = pd.DataFrame([[1.0, 1e-9, 1e-9]])
    gene_specificity, peak_specificity = marginal_specificity(flat, concentrated)
    assert gene_specificity[0] == pytest.approx(0.0, abs=1e-12)
    assert peak_specificity[0] > 0.99


def test_dominant_celltypes_argmax_constant_profiles_and_concordance():
    cols = ["a", "b", "c"]
    rna = pd.DataFrame([[3.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.5, 4.0, 0.1]], index=["g0", "g1", "g2"], columns=cols)
    atac = pd.DataFrame([[1.0, 5.0, 1.0], [2.0, 2.0, 2.0], [0.2, 0.1, 9.0], [4.0, 1.0, 1.0]],
                        index=["p0", "p1", "p2", "p3"], columns=cols)
    gene_idx = np.array([0, 1, 2, 0, -1, 0])
    peak_idx = np.array([0, 0, 2, 2, 1, 3])
    gene_dom, peak_dom, link_dom, concordant = dominant_celltypes(rna, atac, gene_idx, peak_idx, chunk_size=2)
    # gene g1 is flat and p1 is flat -> no dominant cell type; a negative index marks a missing link
    assert list(gene_dom) == ["a", None, "b", "a", None, "a"]
    assert list(peak_dom) == ["b", "b", "c", "c", None, "a"]
    # joint products: g0*p0=[3,5,2]->b, g1*p0=[1,5,1]->b (flat gene, specific peak), g2*p2=[.1,.4,.9]->c,
    #                 g0*p2=[.6,.1,18]->c, missing gene->None, g0*p3=[12,1,2]->a
    assert list(link_dom) == ["b", "b", "c", "c", None, "a"]
    np.testing.assert_array_equal(concordant, [0.0, np.nan, 0.0, 0.0, np.nan, 1.0])
