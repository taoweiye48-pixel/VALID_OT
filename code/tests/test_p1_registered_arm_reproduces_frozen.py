import os
from pathlib import Path

import numpy as np
import pytest

from validot.io import load_pair
from validot.metrics import exact_row_response
from validot.p1 import P1Parameters, arm_weights, mixed_cost, solve_p1
from validot.solvers import cost_components


SOURCE = Path(
    os.environ.get(
        "VALIDOT_LEGACY_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "legacy_reference",
    )
)


def test_p1_registered_arm_reproduces_frozen():
    if not SOURCE.exists():
        pytest.skip(
            "Set VALIDOT_LEGACY_ROOT to the separately archived legacy reference "
            "installation to run this immutable P0 reproduction test."
        )
    pair_id = "STAR_8M_D1_D2__reverse"
    pair, _ = load_pair(SOURCE / "03_data_processed" / "external_pairs" / f"{pair_id}.npz")
    components = cost_components(pair.source_x, pair.target_x, pair.source_xy, pair.target_xy)
    a = np.full(len(pair.source_x), 1 / len(pair.source_x))
    b = np.full(len(pair.target_x), 1 / len(pair.target_x))
    parameters = P1Parameters("row_softmax", 0.25)
    base = solve_p1(mixed_cost(components["expression"], components["spatial_cross"], (0.5, 0.5)), a, b, parameters)
    with np.load(SOURCE / "10_E6_real_external" / pair_id / "row_softmax" / "row_responses.npz") as frozen:
        for intervention in ("I_EXPR", "I_SPATIAL"):
            endpoint = solve_p1(mixed_cost(components["expression"], components["spatial_cross"], arm_weights("R", intervention, 1.0)), a, b, parameters)
            fd = solve_p1(mixed_cost(components["expression"], components["spatial_cross"], arm_weights("R", intervention, 0.01)), a, b, parameters)
            np.testing.assert_allclose(exact_row_response(base.plan, endpoint.plan), frozen[f"exact_{intervention}"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(exact_row_response(base.plan, fd.plan) / 0.01, frozen[f"endpoint_{intervention}"], rtol=0, atol=1e-12)
