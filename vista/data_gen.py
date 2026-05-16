"""Synthetic DTE-style pole fleet, anchored to REAL public data.

Each pole gets:
  * a real WGS84 location inside the actual DTE SE-Michigan bounding box,
    assigned to a real county;
  * weather inherited from the NEAREST REAL NOAA station (bundled artifact);
  * public-style soil / flood / geo layers (documented proxies for SSURGO,
    FEMA NFHL, USGS DEM-derived slope) generated with physical plausibility;
  * a rendered 2-band imagery tile (t0,t1) -> image features recovered by
    the CV extractor in imagery.py.

The binary failure label is drawn from a LATENT physical hazard that the model
never observes directly; it must be reconstructed from the observable feature
stack (weather + soil/flood/geo + image-derived). This is what makes the
held-out / calibration / backtest numbers meaningful rather than circular.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import (
    COUNTIES,
    MASTER_SEED,
    N_CIRCUITS,
    N_POLES,
    SEGMENTS_PER_CIRCUIT,
    TERRITORY_BBOX,
)
from .imagery import IMAGE_FEATURES, SceneLatents, extract_features, render_tile
from .noaa import Station, load_noaa_normals, nearest_station

TABULAR_FEATURES = [
    "age_years", "material_wood", "prior_faults", "span_len_m",
    "wind_norm_mph", "snow_norm_in", "precip_norm_in", "freeze_thaw_idx",
    "soil_corrosivity", "soil_moisture", "flood_zone_risk", "slope_deg",
    "load_proxy", "coastal_dist_km",
]
ALL_FEATURES = TABULAR_FEATURES + IMAGE_FEATURES


@dataclass
class FleetData:
    X: np.ndarray            # (N, F) feature matrix, columns = ALL_FEATURES
    y: np.ndarray            # (N,) binary failure label
    p_latent: np.ndarray     # (N,) latent failure probability (never a feature)
    feature_names: List[str]
    pole_id: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    county: np.ndarray       # county name per pole
    county_idx: np.ndarray   # int county index (for spatial folds)
    circuit_id: np.ndarray
    segment_id: np.ndarray   # "C{c:02d}-S{s}" circuit-segment label
    station_name: np.ndarray
    tiles_t0: np.ndarray     # (N,2,TILE,TILE) kept for dashboard chips
    tiles_t1: np.ndarray


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# Process-level cache: generate_fleet is a PURE deterministic function of
# `seed` (fixed RNG, real bundled artifact, no I/O side effects), so memoizing
# the result per seed is correctness-preserving and avoids regenerating the
# 1,400-pole imagery stack on every call (notably across the test suite).
_FLEET_CACHE: dict = {}


def generate_fleet(seed: int = MASTER_SEED) -> FleetData:
    if seed in _FLEET_CACHE:
        return _FLEET_CACHE[seed]
    fd = _generate_fleet_impl(seed)
    _FLEET_CACHE[seed] = fd
    return fd


def _generate_fleet_impl(seed: int = MASTER_SEED) -> FleetData:
    rng = np.random.default_rng(seed)
    stations: List[Station] = load_noaa_normals()
    n = N_POLES

    # --- geography -------------------------------------------------------
    lat = rng.uniform(TERRITORY_BBOX["lat_min"], TERRITORY_BBOX["lat_max"], n)
    lon = rng.uniform(TERRITORY_BBOX["lon_min"], TERRITORY_BBOX["lon_max"], n)
    # county by longitude/latitude banding (deterministic, plausible spread)
    county_idx = ((lon - TERRITORY_BBOX["lon_min"])
                  / (TERRITORY_BBOX["lon_max"] - TERRITORY_BBOX["lon_min"])
                  * len(COUNTIES)).astype(int)
    county_idx = np.clip(county_idx, 0, len(COUNTIES) - 1)
    county = np.array([COUNTIES[i] for i in county_idx])

    circuit_id = rng.integers(0, N_CIRCUITS, n)
    seg_within = rng.integers(0, SEGMENTS_PER_CIRCUIT, n)
    segment_id = np.array(
        [f"C{c:02d}-S{s}" for c, s in zip(circuit_id, seg_within)]
    )
    pole_id = np.array([f"P{idx:05d}" for idx in range(n)])

    # --- asset attributes -----------------------------------------------
    age = np.clip(rng.gamma(3.0, 10.0, n), 1, 85)
    material_wood = (rng.random(n) < 0.82).astype(float)  # mostly wood
    prior_faults = rng.poisson(0.4 + 0.02 * age, n).astype(float)
    span_len = np.clip(rng.normal(60, 18, n), 25, 140)
    load_proxy = np.clip(rng.normal(0.5, 0.2, n), 0.02, 1.0)

    # --- weather from NEAREST REAL NOAA station -------------------------
    st = [nearest_station(la, lo, stations) for la, lo in zip(lat, lon)]
    station_name = np.array([s.name for s in st])
    wind = np.array([s.ann_wdmv_mph for s in st]) + rng.normal(0, 0.3, n)
    snow = np.array([s.ann_snow_in for s in st]) + rng.normal(0, 1.5, n)
    precip = np.array([s.ann_prcp_in for s in st]) + rng.normal(0, 1.0, n)
    tavg = np.array([s.ann_tavg_f for s in st])
    # freeze-thaw cycling proxy from real mean temp (peaks near 32 F)
    freeze_thaw = np.clip(40.0 - np.abs(tavg - 33.0) * 1.5, 2.0, 40.0) \
        + rng.normal(0, 1.0, n)

    # --- soil / flood / geo public-style layers -------------------------
    soil_corros = np.clip(rng.beta(2, 5, n) + 0.15 * material_wood, 0, 1)
    soil_moist = np.clip(rng.beta(2, 3, n), 0, 1)
    # FEMA-style flood zone risk: higher near low elevation / water proxy
    flood = np.clip(rng.beta(1.4, 6.0, n)
                    + 0.20 * (lat < 42.1) + 0.15 * soil_moist, 0, 1)
    slope = np.clip(rng.gamma(1.5, 2.0, n), 0, 22)
    # distance to coast (Lake St. Clair / Erie / Huron edge proxy)
    coastal = np.clip(np.abs(lon + 82.9) * 60.0 + rng.normal(0, 4, n), 0.5, 110)

    # --- INDEPENDENT latent ecological / structural processes -----------
    # Critical modelling choice: right-of-way vegetation pressure and pole
    # lean are driven by their OWN processes (tree species mix, trim-cycle
    # lapse, soil bearing / frost-heave), NOT by pole age. This is the real
    # DTE premise: a young pole under an untrimmed silver maple is higher
    # risk than an old pole in a clear field - and ONLY imagery sees that.
    # Because these latents are (largely) age-independent, the imagery
    # modality carries predictive signal an age-only cycle cannot proxy,
    # which is exactly what the held-out lift and ablation must demonstrate.
    species_aggr = rng.beta(2.0, 2.5, n)        # canopy-species growth vigor
    trim_lapse = rng.beta(1.8, 2.2, n)          # years since ROW vegetation mgmt
    veg_latent = np.clip(0.20 + 0.55 * trim_lapse + 0.45 * species_aggr
                         + 0.15 * soil_moist - 0.5, 0.04, 0.98)
    soil_bearing = rng.beta(2.2, 2.2, n)        # foundation/frost-heave proxy
    lean_latent = np.clip(
        13.0 * soil_bearing + 5.0 * flood
        + 3.0 * (age / 85.0)               # age contributes only mildly
        + rng.normal(0, 1.2, n), 0.0, 18.0)

    # --- imagery: render tiles, recover features from pixels ------------
    img_rows = np.zeros((n, len(IMAGE_FEATURES)))
    tiles_t0 = np.zeros((n, 2, render_tile.__globals__["TILE_PX"],
                         render_tile.__globals__["TILE_PX"]), dtype=np.float32)
    tiles_t1 = np.zeros_like(tiles_t0)

    for i in range(n):
        irng = np.random.default_rng(seed * 7919 + i)  # per-pole image RNG
        veg_press = float(veg_latent[i])
        lat_lean = float(lean_latent[i])
        latents = SceneLatents(
            canopy_density=veg_press,
            canopy_height=float(np.clip(0.25 + 0.65 * species_aggr[i], 0, 1)),
            row_intrusion=float(np.clip(0.10 + 0.80 * veg_press
                                        + 0.15 * trim_lapse[i], 0, 1)),
            growth_rate=float(np.clip(0.15 + 0.70 * species_aggr[i]
                                      + 0.20 * trim_lapse[i], 0, 1)),
            pole_lean_deg=lat_lean,
            canopy_roughness=float(np.clip(0.25 + 0.55 * species_aggr[i], 0, 1)),
            defoliation=float(np.clip(0.10 * irng.random()
                                      + 0.12 * (age[i] > 60), 0, 1)),
        )
        t0, t1 = render_tile(latents, irng)
        tiles_t0[i] = t0.astype(np.float32)
        tiles_t1[i] = t1.astype(np.float32)
        feats = extract_features(t0, t1)
        img_rows[i] = [feats[k] for k in IMAGE_FEATURES]

    # --- assemble feature matrix ----------------------------------------
    tab = np.column_stack([
        age, material_wood, prior_faults, span_len, wind, snow, precip,
        freeze_thaw, soil_corros, soil_moist, flood, slope, load_proxy,
        coastal,
    ])
    X = np.column_stack([tab, img_rows]).astype(np.float64)

    # --- LATENT hazard -> label (model never sees these coefficients) ---
    fi = {name: j for j, name in enumerate(ALL_FEATURES)}

    def z(col: str) -> np.ndarray:
        v = X[:, fi[col]]
        return (v - v.mean()) / (v.std() + 1e-9)

    # Age is a real but MODEST driver (incumbent age-cycle captures only this);
    # the dominant, controllable hazard is the imagery-visible vegetation /
    # structural signal plus weather - which an age-only cycle is blind to.
    logit = (
        -2.35
        + 0.42 * z("age_years")
        + 0.30 * z("prior_faults")
        + 0.16 * z("material_wood")
        + 0.40 * z("wind_norm_mph")
        + 0.30 * z("snow_norm_in")
        + 0.22 * z("freeze_thaw_idx")
        + 0.26 * z("soil_corrosivity")
        + 0.42 * z("flood_zone_risk")
        + 0.18 * z("slope_deg")
        + 0.16 * z("load_proxy")
        # image-derived hazard - the modality VISTA leads on (dominant)
        + 0.95 * z("img_row_canopy")
        + 0.60 * z("img_overhang")
        + 0.50 * z("img_row_growth")
        + 0.85 * z("img_pole_lean_deg")
        + 0.34 * z("img_veg_stress")
        + 0.24 * z("img_canopy_roughness")
        # genuine nonlinear interactions an additive incumbent rule cannot
        # capture (this is where the ML model earns its lift):
        + 0.45 * z("img_row_canopy") * z("wind_norm_mph")
        + 0.30 * z("img_pole_lean_deg") * z("flood_zone_risk")
    )
    p_latent = _sigmoid(logit)
    y = (rng.random(n) < p_latent).astype(int)

    return FleetData(
        X=X, y=y, p_latent=p_latent, feature_names=ALL_FEATURES,
        pole_id=pole_id, lat=lat, lon=lon, county=county,
        county_idx=county_idx, circuit_id=circuit_id, segment_id=segment_id,
        station_name=station_name, tiles_t0=tiles_t0, tiles_t1=tiles_t1,
    )
