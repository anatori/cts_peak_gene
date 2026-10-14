import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps

from ctar.celltype_specificity import (
    coexp_context, coexp_specificity, entropy_specificity,
)
from ctar.parallel import multiprocess_coexp_specificity

RNA = np.array([[4, 0, 1], [2, 0, 3], [0, 5, 0], [1, 7, 2], [9, 9, 9]], dtype=np.int32)
ATAC = np.array([[1, 0], [3, 2], [0, 4], [2, 6], [5, 5]], dtype=np.int32)
LABELS = ["a", "a", "b", "b", None]


def _adata(counts):
    obs = pd.DataFrame({"celltype": LABELS}, index=[f"c{i}" for i in range(counts.shape[0])])
    return ad.AnnData(X=sps.csr_matrix(counts), obs=obs,
                      var=pd.DataFrame(index=[f"f{i}" for i in range(counts.shape[1])]))


def _brute_force(gene, peak):
    """P_t and pseudocount computed cell by cell from the definition in coexp_context (default median target)."""
    labeled = [0, 1, 2, 3]
    rna_target, atac_target = np.median(RNA.sum(axis=1)), np.median(ATAC.sum(axis=1))   # median over ALL cells
    log_rna = np.log1p(RNA / RNA.sum(axis=1, keepdims=True) * rna_target)
    log_atac = np.log1p(ATAC / ATAC.sum(axis=1, keepdims=True) * atac_target)
    eps_rna = np.log(2) / len(labeled)                                       # one count at median depth -> log1p(1)
    eps_atac = np.log(2) / len(labeled)
    product = log_rna[:, gene] * log_atac[:, peak]
    profile = np.array([product[[0, 1]].mean(), product[[2, 3]].mean()]) + eps_rna * eps_atac
    return profile, eps_rna * eps_atac


def test_coexp_context_reuses_precomputed_matrices():
    rna, atac = _adata(RNA), _adata(ATAC)
    precomputed = []
    for adata in (_adata(RNA), _adata(ATAC)):
        adata.X = adata.X.astype(np.float64)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        precomputed.append(adata.X)
    built = coexp_context(rna, atac, layer=None)
    reused = coexp_context(rna, atac, layer=None, log_rna=precomputed[0], log_atac=precomputed[1])
    np.testing.assert_allclose(reused["log_rna"].toarray(), built["log_rna"].toarray(), rtol=1e-6)
    np.testing.assert_allclose(reused["log_atac"].toarray(), built["log_atac"].toarray(), rtol=1e-6)
    assert reused["pseudocount"] == built["pseudocount"]


def test_coexp_context_pseudocount_and_groups():
    context = coexp_context(_adata(RNA), _adata(ATAC), layer=None)
    _, pseudocount = _brute_force(0, 0)
    assert list(context["celltypes"]) == ["a", "b"]
    np.testing.assert_allclose(context["group_sizes"], [2, 2])
    np.testing.assert_allclose(context["pseudocount"], pseudocount, rtol=1e-12)
    assert context["log_rna"].shape == (4, 3) and context["log_atac"].shape == (4, 2)   # NaN-labelled cell dropped


def test_coexp_specificity_matches_brute_force():
    context = coexp_context(_adata(RNA), _adata(ATAC), layer=None)
    gene_idx = np.array([0, 1, 2, 2, 0])
    peak_idx = np.array([0, 1, 0, 1, 1])
    specificity, dominant = coexp_specificity(context, gene_idx, peak_idx, chunk_size=2)
    for i, (g, p) in enumerate(zip(gene_idx, peak_idx)):
        profile, _ = _brute_force(g, p)
        np.testing.assert_allclose(specificity[i], entropy_specificity(profile, eps=0.0), rtol=1e-5)
        assert dominant[i] == np.argmax(profile)


def test_gene_sorted_chunks_return_input_order():
    context = coexp_context(_adata(RNA), _adata(ATAC), layer=None)
    gene_idx, peak_idx = np.array([2, 0, 1, 0]), np.array([1, 0, 1, 1])     # unsorted genes, one link per chunk
    specificity, dominant = coexp_specificity(context, gene_idx, peak_idx, chunk_size=1)
    for i, (g, p) in enumerate(zip(gene_idx, peak_idx)):
        profile, _ = _brute_force(g, p)
        np.testing.assert_allclose(specificity[i], entropy_specificity(profile, eps=0.0), rtol=1e-5)
        assert dominant[i] == np.argmax(profile)


def test_multiprocess_matches_serial_for_array_and_dict():
    context = coexp_context(_adata(RNA), _adata(ATAC), layer=None)
    links = np.array([[0, 0], [1, 1], [0, 2], [1, 2], [1, 0]])   # ATAC index in column 0, RNA in column 1
    expected_spec, expected_dom = coexp_specificity(context, links[:, 1], links[:, 0])

    spec, dom = multiprocess_coexp_specificity(links, context, n_workers=2, array_chunk_size=2)
    np.testing.assert_allclose(spec, expected_spec)
    np.testing.assert_array_equal(dom, expected_dom)

    spec_dic, dom_dic = multiprocess_coexp_specificity({"x": links[:3], "y": links[3:]}, context, n_workers=2)
    np.testing.assert_allclose(np.concatenate([spec_dic["x"], spec_dic["y"]]), expected_spec)
    np.testing.assert_array_equal(np.concatenate([dom_dic["x"], dom_dic["y"]]), expected_dom)
