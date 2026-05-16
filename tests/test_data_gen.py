"""Fleet generation: determinism, real-data anchoring, and the crucial
NO-LEAKAGE property (the latent hazard probability is never a model feature,
and image scene-latents are decoupled from pole age)."""

import numpy as np

from vista.data_gen import ALL_FEATURES, IMAGE_FEATURES, generate_fleet


def test_fleet_deterministic():
    a = generate_fleet()
    b = generate_fleet()
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.p_latent, b.p_latent)


def test_fleet_shape_and_label_balance():
    fd = generate_fleet()
    assert fd.X.shape[0] == len(fd.y) == 1400
    assert fd.X.shape[1] == len(ALL_FEATURES)
    assert 0.10 < fd.y.mean() < 0.35, "failure base rate must be realistic"
    assert len(set(fd.segment_id)) == 130


def test_image_features_present_in_matrix():
    fd = generate_fleet()
    for name in IMAGE_FEATURES:
        assert name in fd.feature_names
    n_img = sum(1 for n in fd.feature_names if n.startswith("img_"))
    assert n_img == 9


def test_no_latent_leakage_into_features():
    """p_latent must NOT appear (even numerically) as any feature column."""
    fd = generate_fleet()
    for j in range(fd.X.shape[1]):
        col = fd.X[:, j]
        # identical or trivially-scaled copy of p_latent would be leakage
        if col.std() > 0:
            r = abs(np.corrcoef(col, fd.p_latent)[0, 1])
            assert r < 0.97, (
                f"feature {fd.feature_names[j]} corr {r:.3f} with latent "
                "is implausibly high - possible leakage")


def test_imagery_decoupled_from_age():
    """VISTA's thesis requires the imagery modality to carry signal an
    age-only cycle cannot proxy. Verify image RoW/lean features are weakly
    correlated with pole age (independent ecological/structural process)."""
    fd = generate_fleet()
    fi = {n: j for j, n in enumerate(fd.feature_names)}
    age = fd.X[:, fi["age_years"]]
    for f in ("img_row_canopy", "img_row_growth", "img_canopy_frac"):
        r = abs(np.corrcoef(age, fd.X[:, fi[f]])[0, 1])
        assert r < 0.30, f"{f} too correlated with age ({r:.3f})"


def test_weather_anchored_to_real_noaa_range():
    """Synthetic weather columns must sit in the REAL NOAA normal ranges
    (they are inherited from the nearest real station)."""
    fd = generate_fleet()
    fi = {n: j for j, n in enumerate(fd.feature_names)}
    snow = fd.X[:, fi["snow_norm_in"]]
    precip = fd.X[:, fi["precip_norm_in"]]
    assert 30.0 < snow.mean() < 55.0
    assert 28.0 < precip.mean() < 42.0
