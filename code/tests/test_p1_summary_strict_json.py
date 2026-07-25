import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_nonapplicable_reproduction_summary_uses_json_null_not_nan():
    source = (ROOT / "code" / "run_p1_scale_regularization.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    finalize = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "finalize")
    text = ast.get_source_segment(source, finalize)
    assert 'reproduction_max = None' in text
    assert 'reproduction_values' in text
    assert '"arm_r_reproduction_max_abs_difference"' in text
