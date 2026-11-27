import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5개초과 10개이하", 7.5),
        ("10건초과 20건이하", 15.0),
        ("50건 초과", 55.0),
        ("2개", 2.0),
        ("0건", 0.0),
        ("  3개 ", 3.0),
    ],
)
def test_parse_bucket_value(notebook_functions, text, expected):
    assert notebook_functions["parse_bucket_value"](text) == pytest.approx(expected)


@pytest.mark.parametrize("missing", [np.nan, None, "", "없음", "nan"])
def test_parse_bucket_value_missing(notebook_functions, missing):
    assert np.isnan(notebook_functions["parse_bucket_value"](missing))


def test_clean_amount(notebook_functions):
    result = notebook_functions["clean_amount"](pd.Series(["1,234.5", "12", "abc", 0.5]))
    np.testing.assert_allclose(result.to_numpy(dtype=float), [1234.5, 12.0, np.nan, 0.5])


def test_metrics_on_small_ranking(notebook_functions):
    f = notebook_functions
    actual = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])

    assert f["precision_at_k"](actual, scores, 2) == pytest.approx(0.5)
    assert f["recall_at_k"](actual, scores, 2) == pytest.approx(0.5)
    assert f["hit_rate_at_k"](actual, scores, 2) == 1.0
    # 정답이 1위와 3위: DCG = 1/log2(2), IDCG = 1/log2(2) + 1/log2(3)
    assert f["ndcg_at_k"](actual, scores, 2) == pytest.approx(1 / (1 + 1 / np.log2(3)))


def test_metrics_perfect_ranking(notebook_functions):
    f = notebook_functions
    actual = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])

    assert f["precision_at_k"](actual, scores, 2) == 1.0
    assert f["ndcg_at_k"](actual, scores, 2) == pytest.approx(1.0)


def test_metrics_k_larger_than_candidates(notebook_functions):
    f = notebook_functions
    actual = np.array([1, 0, 0])
    scores = np.array([0.1, 0.2, 0.3])

    assert f["precision_at_k"](actual, scores, 10) == pytest.approx(1 / 3)
    assert f["recall_at_k"](actual, scores, 10) == 1.0


def test_metrics_without_relevant_items(notebook_functions):
    f = notebook_functions
    actual = np.array([0, 0, 0])
    scores = np.array([0.3, 0.2, 0.1])

    assert f["hit_rate_at_k"](actual, scores, 2) == 0.0
    assert np.isnan(f["recall_at_k"](actual, scores, 2))
    assert np.isnan(f["ndcg_at_k"](actual, scores, 2))


@pytest.mark.parametrize("metric", ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"])
def test_metrics_reject_length_mismatch(notebook_functions, metric):
    with pytest.raises(ValueError):
        notebook_functions[metric](np.array([1, 0, 1]), np.array([0.5, 0.4]), 2)
