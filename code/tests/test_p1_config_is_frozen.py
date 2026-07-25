import hashlib
from pathlib import Path


EXPECTED = "dd72774f6a525ec380d8bf32a5fc3c6e5edbde4687a16f569abccb892687e3b1"


def test_p1_config_is_frozen():
    root = Path(__file__).resolve().parents[2]
    config = root / "configs" / "p1_scale_regularization_v1.yaml"
    assert hashlib.sha256(config.read_bytes()).hexdigest() == EXPECTED
    snapshot = root / "analysis" / "p1_scale_regularization" / "config_snapshot.sha256"
    if snapshot.is_file():
        snapshot_hash = snapshot.read_text(encoding="utf-8").split()[0]
        assert snapshot_hash == EXPECTED
