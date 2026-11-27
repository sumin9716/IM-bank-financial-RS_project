"""합성 데이터로 노트북 사본을 끝까지 실행한다. 저장소의 노트북과 저장된 출력은 바꾸지 않는다."""

import json
import shutil

import nbformat
import pytest
from nbclient import NotebookClient
from synthetic_data import NOTEBOOK_PATH, SYNTHETIC_REGIONS, write_synthetic_data

SUMMARY_CELL = """
import json
print(json.dumps({
    "n_regions": len(feature_df),
    "metrics": {name: float(value) for name, value in metrics.items()},
    "recommended_provinces": sorted(set(recommendations["사업장_시도"])),
    "similar_regions": similar_regions["권역"].tolist(),
    "compare_scalers": sorted(set(compare_table["사용스케일러"])),
    "compare_rows": len(compare_table),
    "rule_weight_sum": sum(rule_weights.values()),
}, ensure_ascii=False))
"""


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    project = tmp_path_factory.mktemp("project")
    write_synthetic_data(project / "data")
    notebook_dir = project / "notebooks"
    notebook_dir.mkdir()
    shutil.copy(NOTEBOOK_PATH, notebook_dir / NOTEBOOK_PATH.name)

    nb = nbformat.read(notebook_dir / NOTEBOOK_PATH.name, as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(SUMMARY_CELL))
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(notebook_dir)}}
    )
    client.execute()

    stdout = [out["text"] for out in nb.cells[-1].outputs if out.get("name") == "stdout"]
    return json.loads("".join(stdout))


def test_all_regions_are_scored(summary):
    assert summary["n_regions"] == len(SYNTHETIC_REGIONS)


def test_metrics_are_in_unit_interval(summary):
    assert set(summary["metrics"]) == {"Precision@10", "Recall@10", "HitRate@10", "NDCG@10"}
    assert all(0.0 <= value <= 1.0 for value in summary["metrics"].values())


def test_recommendations_are_limited_to_metro_area(summary):
    assert set(summary["recommended_provinces"]) <= {"서울특별시", "경기도", "인천광역시"}


def test_similar_regions_exclude_daegu_gyeongbuk_for_daegu_input(summary):
    assert len(summary["similar_regions"]) == 5
    assert not any(region.startswith(("대구광역시", "경상북도")) for region in summary["similar_regions"])


def test_scaler_comparison_covers_six_scalers(summary):
    assert summary["compare_scalers"] == sorted(["minmax", "standard", "robust", "maxabs", "quantile", "power-yeo"])
    assert summary["compare_rows"] == 6 * 3


def test_rule_weights_sum_to_one(summary):
    assert summary["rule_weight_sum"] == pytest.approx(1.0)
