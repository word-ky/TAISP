import importlib.util
from pathlib import Path

import pytest
import numpy as np


def test_taylor_sign_formula_and_cluster_bootstrap_on_known_relation():
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("linear_analysis", Path("scripts/analyze_t002_linear.py"))
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    rows = []
    for image_id, value in enumerate([-4., -2., -1., 1., 2., 4., 5., 8.]):
        for offset in (0., 0.1):
            gradient = value + offset
            rows.append({"image_id": image_id, "g_det": [1., 0.], "g_sem": [gradient, 0.],
                         "det_loss_delta_sem1": -0.1 * gradient,
                         "gradient_cosine": float(np.sign(gradient))})
    result, predicted = analysis.describe(rows, 0.1, replicates=100)
    np.testing.assert_allclose(predicted, [r["det_loss_delta_sem1"] for r in rows])
    assert result["observations"] == 16 and result["image_clusters"] == 8
    assert result["sign_agreement"]["estimate"] == 1
    assert result["spearman_linear_observed"]["estimate"] == pytest.approx(1)
    assert result["pearson_linear_observed"]["estimate"] == pytest.approx(1)
