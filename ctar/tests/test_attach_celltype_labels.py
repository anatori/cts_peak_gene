import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ctar.data_loader import attach_celltype_labels


def _adata():
    obs = pd.DataFrame(index=["c0", "c1", "c2", "c3"])
    return ad.AnnData(X=np.ones((4, 2)), obs=obs)


def test_labels_are_matched_on_barcode_and_missing_cells_are_nan(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"pgboost_ct": ["b", "a", None, "z"]}, index=["c3", "c0", "c1", "other"]).to_csv(path)
    adata = _adata()
    n_matched = attach_celltype_labels(adata, path, "pgboost_ct")
    assert n_matched == 3                                     # c0, c1, c3 are in the file ("other" is not an obs name)
    labels = adata.obs["pgboost_ct"].to_numpy()
    assert labels[0] == "a" and labels[3] == "b"              # matched by barcode, not by row order
    assert pd.isna(labels[1]) and pd.isna(labels[2])          # NaN in the file, and absent from the file


def test_missing_column_none_matched_and_duplicates_raise(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"pgboost_ct": ["a"]}, index=["c0"]).to_csv(path)
    with pytest.raises(ValueError, match="not found"):
        attach_celltype_labels(_adata(), path, "celltype")

    pd.DataFrame({"pgboost_ct": ["a"]}, index=["zzz"]).to_csv(path)
    with pytest.raises(ValueError, match="None of the"):
        attach_celltype_labels(_adata(), path, "pgboost_ct")

    pd.DataFrame({"pgboost_ct": ["a", "b"]}, index=["c0", "c0"]).to_csv(path)
    with pytest.raises(ValueError, match="unique"):
        attach_celltype_labels(_adata(), path, "pgboost_ct")
