import numpy as np
import ctar


def test_initial_and_pooled_mcpval_z():
    cis_coeff = np.array(
        [
            [0.5, 2.0],
            [1.0, 0.5],
        ],
        dtype=float,
    )
    ctrl_coeff = np.array(
        [
            [[0.5, 1.0], [0.5, 1.5], [0.5, 2.5]],
            [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]],
        ],
        dtype=float,
    )

    ctrl_beta = ctrl_coeff[:, :, 1]
    ctrl_se = ctrl_coeff[:, :, 0]
    cis_beta = cis_coeff[:, 1]
    cis_se = cis_coeff[:, 0]

    observed_mcpval = ctar.method.initial_mcpval(ctrl_beta, cis_beta)
    observed_ppval = ctar.method.pooled_mcpval(ctrl_beta, cis_beta)

    cis_studentized = cis_beta / cis_se
    ctrl_studentized = ctrl_beta / ctrl_se
    observed_mcpval_z = ctar.method.initial_mcpval(ctrl_studentized, cis_studentized)
    observed_ppval_z = ctar.method.pooled_mcpval(ctrl_studentized, cis_studentized)

    expected_mcpval = np.array([
        0.5,   # row 1: (1 + 1 control >= 2.0) / (1 + 3)
        0.75,  # row 2: (1 + 2 controls >= 0.5) / (1 + 3)
    ])
    expected_ppval = np.array([
        3.0 / 7.0,  # centered 0.5345 has 2 pooled controls >= it
        4.0 / 7.0,  # centered 0.0 has 3 pooled controls >= it
    ])

    # In this toy example, studentizing preserves the same ordering and pooled positions.
    expected_mcpval_z = expected_mcpval
    expected_ppval_z = expected_ppval

    assert np.allclose(observed_mcpval, expected_mcpval)
    assert np.allclose(observed_ppval, expected_ppval)
    assert np.allclose(observed_mcpval_z, expected_mcpval_z)
    assert np.allclose(observed_ppval_z, expected_ppval_z)


def test_binned_mcpval():
    cis_pairs_dic = {
        "bin_b": np.array([2.0, np.nan, 0.5]),
        "bin_a": np.array([1.5, -0.5]),
    }
    ctrl_pairs_dic = {
        "bin_b": np.array([1.0, 1.5, 2.5, np.nan]),
        "bin_a": np.array([0.0, 0.5, 1.0]),
    }

    observed_mcpval_dic, observed_ppval_dic = ctar.method.binned_mcpval(
        cis_pairs_dic,
        ctrl_pairs_dic,
        b=3,
    )

    expected_mcpval_dic = {
        "bin_a": np.array([
            0.25,  # (1 + 0 controls >= 1.5) / (1 + 3)
            1.0,   # (1 + 3 controls >= -0.5) / (1 + 3)
        ]),
        "bin_b": np.array([
            0.5,   # (1 + 1 control >= 2.0) / (1 + 3)
            np.nan,
            1.0,   # (1 + 3 controls >= 0.5) / (1 + 3)
        ]),
    }
    expected_ppval_dic = {
        "bin_a": np.array([
            1.0 / 7.0,  # centered 2.4495 exceeds all pooled controls
            1.0,        # centered -2.4495 is below all pooled controls
        ]),
        "bin_b": np.array([
            3.0 / 7.0,  # centered 0.5345 has 2 pooled controls >= it
            np.nan,
            1.0,        # centered -1.8708 is below all pooled controls
        ]),
    }

    for key in sorted(cis_pairs_dic.keys()):
        assert np.allclose(
            observed_mcpval_dic[key],
            expected_mcpval_dic[key],
            equal_nan=True,
        )
        assert np.allclose(
            observed_ppval_dic[key],
            expected_ppval_dic[key],
            equal_nan=True,
        )
