"""The imagery pipeline is VISTA's distinct contribution - test that it is a
REAL pixel pipeline: features must be RECOVERED from rendered pixels and must
track the ground-truth scene latents (monotone response), not be passthroughs.
"""

import numpy as np

from vista.imagery import (
    IMAGE_FEATURES,
    SceneLatents,
    extract_features,
    render_tile,
)
from vista.config import TILE_PX


def _base_latents(**kw):
    d = dict(canopy_density=0.4, canopy_height=0.4, row_intrusion=0.3,
             growth_rate=0.3, pole_lean_deg=4.0, canopy_roughness=0.4,
             defoliation=0.1)
    d.update(kw)
    return SceneLatents(**d)


def test_render_shapes_and_bands():
    rng = np.random.default_rng(0)
    t0, t1 = render_tile(_base_latents(), rng)
    assert t0.shape == (2, TILE_PX, TILE_PX)
    assert t1.shape == (2, TILE_PX, TILE_PX)
    assert np.isfinite(t0).all() and np.isfinite(t1).all()
    assert t0.min() >= 0.0 and t0.max() <= 1.0


def test_feature_extractor_returns_full_stack():
    rng = np.random.default_rng(1)
    t0, t1 = render_tile(_base_latents(), rng)
    f = extract_features(t0, t1)
    assert set(f.keys()) == set(IMAGE_FEATURES)
    for v in f.values():
        assert np.isfinite(v)


def test_canopy_density_increases_recovered_canopy():
    """More canopy in the scene -> extractor recovers more canopy fraction.
    This proves the feature is read FROM PIXELS, not the latent."""
    rng = np.random.default_rng(2)
    lo = extract_features(*render_tile(_base_latents(canopy_density=0.10), rng))
    rng = np.random.default_rng(2)
    hi = extract_features(*render_tile(_base_latents(canopy_density=0.85), rng))
    assert hi["img_canopy_frac"] > lo["img_canopy_frac"] + 0.05


def test_row_intrusion_increases_recovered_row_canopy():
    rng = np.random.default_rng(3)
    lo = extract_features(*render_tile(_base_latents(row_intrusion=0.05), rng))
    rng = np.random.default_rng(3)
    hi = extract_features(*render_tile(_base_latents(row_intrusion=0.95), rng))
    assert hi["img_row_canopy"] > lo["img_row_canopy"]


def test_pole_lean_is_recovered_by_hough():
    """The Hough-style estimator must recover a near-vertical pole as ~0 deg
    and a strongly leaning pole as a clearly larger angle - geometry from
    pixels, independent of the latent value."""
    rng = np.random.default_rng(4)
    straight = extract_features(*render_tile(
        _base_latents(pole_lean_deg=0.0, canopy_density=0.2), rng))
    rng = np.random.default_rng(4)
    leaning = extract_features(*render_tile(
        _base_latents(pole_lean_deg=15.0, canopy_density=0.2), rng))
    assert straight["img_pole_lean_deg"] < 4.0
    assert leaning["img_pole_lean_deg"] > straight["img_pole_lean_deg"] + 4.0


def test_change_detection_sees_growth():
    """Higher growth_rate -> larger recovered bi-temporal RoW NDVI change."""
    rng = np.random.default_rng(5)
    lo = extract_features(*render_tile(_base_latents(growth_rate=0.05), rng))
    rng = np.random.default_rng(5)
    hi = extract_features(*render_tile(_base_latents(growth_rate=0.95), rng))
    assert hi["img_row_growth"] >= lo["img_row_growth"]
    assert hi["img_row_growth"] > 0.0


def test_extractor_is_deterministic():
    rng = np.random.default_rng(7)
    a = extract_features(*render_tile(_base_latents(), rng))
    rng = np.random.default_rng(7)
    b = extract_features(*render_tile(_base_latents(), rng))
    assert a == b
