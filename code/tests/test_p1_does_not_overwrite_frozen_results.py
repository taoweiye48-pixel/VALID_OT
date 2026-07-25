from pathlib import Path


def test_p1_does_not_overwrite_frozen_results():
    root = Path(__file__).resolve().parents[2]
    result_root = (root / "results" / "p1_scale_regularization").resolve()
    forbidden = [
        (root / "results.json").resolve(),
        (root / "analysis_freeze" / "p0-pre-reanalysis").resolve(),
        (root / "code" / "protocol" / "frozen_config.json").resolve(),
    ]
    assert all(result_root != path and path not in result_root.parents for path in forbidden)
