"""파이프 프로파일(원통) RANSAC 피팅/검증 단위 테스트 - 합성 점군(노이즈 + 이상치 + 이웃 형상)."""

from __future__ import annotations

import numpy as np
import pytest

from plugins.poseDeterminator.PipeProfileFitting import (
    PipeProfileParams, fit_metrics, ransac_cylinder, refine_cylinder, validate_fit)

RADIUS = 0.0375


def _pipe_cloud(rng, n=20000, sigma=0.001, sweep_deg=360.0, axis_tilt=(0.3, 0.2, 1.0), outliers=0.3):
    """반지름 RADIUS, 임의 방향 축의 원통 + 가우시안 노이즈 + 임의 이상치 + 평면(이웃 형상)."""
    axis = np.asarray(axis_tilt, float)
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.radians(rng.uniform(0.0, sweep_deg, n))
    along = rng.uniform(-0.2, 0.2, n)
    radial = np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)
    pts = np.outer(along, axis) + RADIUS * radial + rng.normal(0, sigma, (n, 3))
    nrm = radial + rng.normal(0, 0.02, (n, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    n_out = int(n * outliers)
    out_pts = rng.uniform(-0.25, 0.25, (n_out, 3))
    out_nrm = rng.normal(size=(n_out, 3))
    out_nrm /= np.linalg.norm(out_nrm, axis=1, keepdims=True)
    return np.vstack([pts, out_pts]), np.vstack([nrm, out_nrm]), axis, pts[0]


def test_ransac_recovers_radius_and_axis_with_noise_and_outliers():
    rng = np.random.default_rng(1)
    pts, nrm, axis, anchor = _pipe_cloud(rng)
    params = PipeProfileParams()
    model = ransac_cylinder(pts, nrm, anchor, params, np.random.default_rng(0))
    model = refine_cylinder(pts, nrm, model, params)
    assert abs(model["radius"] - RADIUS) < 0.002
    assert abs(abs(float(model["axis"] @ axis)) - 1.0) < 1e-3
    metrics = fit_metrics(pts, model, anchor)
    assert not validate_fit(model["radius"], metrics, len(pts), params)
    assert metrics["angular_coverage_deg"] >= 350


def test_single_view_scan_still_validates_with_partial_coverage():
    rng = np.random.default_rng(2)
    pts, nrm, _axis, anchor = _pipe_cloud(rng, sweep_deg=150.0, outliers=0.1)
    params = PipeProfileParams()
    model = refine_cylinder(pts, nrm, ransac_cylinder(pts, nrm, anchor, params, np.random.default_rng(0)), params)
    assert abs(model["radius"] - RADIUS) < 0.004
    assert not validate_fit(model["radius"], fit_metrics(pts, model, anchor), len(pts), params)


def test_fixed_seed_is_deterministic():
    rng = np.random.default_rng(3)
    pts, nrm, _axis, anchor = _pipe_cloud(rng, n=6000)
    params = PipeProfileParams(seed=7)
    a = ransac_cylinder(pts, nrm, anchor, params, np.random.default_rng(params.seed))
    b = ransac_cylinder(pts, nrm, anchor, params, np.random.default_rng(params.seed))
    assert a["radius"] == b["radius"] and np.allclose(a["axis"], b["axis"])


def test_flat_patch_has_no_cylinder_hypothesis():
    rng = np.random.default_rng(4)
    pts = np.c_[rng.uniform(-0.2, 0.2, (5000, 2)), rng.normal(0, 0.0005, 5000)]
    nrm = np.tile([0.0, 0.0, 1.0], (5000, 1))
    assert ransac_cylinder(pts, nrm, pts[0], PipeProfileParams(), np.random.default_rng(0)) is None


def test_validation_reports_every_reason():
    params = PipeProfileParams()
    metrics = {"n_inliers": 20, "angular_coverage_deg": 30.0, "anchor_residual": 0.05}
    reasons = validate_fit(0.0024, metrics, 10000, params)
    text = " | ".join(reasons)
    for needle in ("outside", "min_inliers", "inlier ratio", "angular coverage", "off the fitted surface"):
        assert needle in text
    mismatch = validate_fit(0.037, {"n_inliers": 500, "angular_coverage_deg": 200.0, "anchor_residual": 0.001},
                            1000, params, estimated_radius=0.002)
    assert any("disagrees with cluster estimate" in r for r in mismatch)


def test_params_from_config_new_section_overrides_legacy_flat_keys():
    cfg = {
        "sampling_size_for_calculating_normal": 0.02,
        "radius_offset_for_sampling_points_in_sphere": 0.004,
        "pipe_profile": {"method": "cluster", "radius_max": 0.2, "cluster_height_range": [-0.05, 0.2],
                         "sphere_radius_offset": 0.005, "typo_key": 1},
    }
    params = PipeProfileParams.from_config(cfg)
    assert params.method == "cluster" and params.radius_max == 0.2
    assert params.normal_sampling_size == 0.02          # legacy flat key still honored
    assert params.sphere_radius_offset == 0.005         # new section wins
    assert params.cluster_height_range == (-0.05, 0.2)
    assert PipeProfileParams.unknown_keys(cfg) == ["typo_key"]


def test_invalid_params_are_rejected():
    with pytest.raises(ValueError):
        PipeProfileParams.from_config({"pipe_profile": {"method": "nope"}})
    with pytest.raises(ValueError):
        PipeProfileParams.from_config({"pipe_profile": {"radius_min": 0.5, "radius_max": 0.1}})
