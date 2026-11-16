import numpy as np
import pandas as pd
import ctar


def _toy():
    genes = pd.DataFrame({
        'gene': ['A', 'B'], 'chr': ['chr1', 'chr1'], 'start': [10000, 50000], 'end': [20000, 60000],
        'strand': [1, -1], 'tss': [10000, 60000],
    })
    peaks = pd.DataFrame({'peak': ['chr1:9800-10300', 'chr1:5000-5500', 'chr1:30000-30500', 'chr1:39800-40300']})
    return peaks, genes


def test_peak_to_gene_distance_is_tss_distance_not_overlap():
    ''' `distance` is the signed peak-to-TSS distance, consistent with abs_distance.
    '''
    peaks, genes = _toy()
    links = ctar.data_loader.peak_to_gene(peaks, genes, distance=20000)
    links = links.set_index(['peak', 'gene'])
    expected = {
        ('chr1:9800-10300', 'A'): 0, # TSS inside the peak
        ('chr1:5000-5500', 'A'): 5500 - 10000, # upstream in genomic coordinates
        ('chr1:30000-30500', 'A'): 30000 - 10000,
        ('chr1:39800-40300', 'A'): 39800 - 10000, # only 200 bp of the peak lie inside the window of A
        ('chr1:30000-30500', 'B'): 30500 - 60000,
        ('chr1:39800-40300', 'B'): 40300 - 60000,
    }
    for key, dist in expected.items():
        assert links.loc[key, 'distance'] == dist, key
        assert links.loc[key, 'abs_distance'] == abs(dist), key
    assert not (links['distance'] == 500).all()
    assert (links['abs_distance'] == links['distance'].abs()).all()


def test_peak_to_gene_without_tss_has_no_distance():
    peaks, genes = _toy()
    links = ctar.data_loader.peak_to_gene(peaks, genes.drop(columns='tss'), distance=20000)
    assert links['distance'].isna().all()
    assert 'abs_distance' not in links.columns
