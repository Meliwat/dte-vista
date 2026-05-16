"""Central configuration. Every random draw in VISTA derives from MASTER_SEED,
so the entire pipeline - data, imagery tiles, model, validation, dashboard - is
byte-for-byte reproducible on any machine, offline."""

from __future__ import annotations

import os

# --- determinism -----------------------------------------------------------
MASTER_SEED = 20260515  # Hack Michigan build date; single source of all RNG.

# Make hash-based ordering deterministic too (matplotlib / numpy do not depend
# on PYTHONHASHSEED but we set it for any downstream dict ordering).
os.environ.setdefault("PYTHONHASHSEED", "0")

# --- fleet scale -----------------------------------------------------------
N_POLES = 1_400          # synthetic DTE-style distribution fleet
N_CIRCUITS = 26          # feeder circuits
SEGMENTS_PER_CIRCUIT = 5  # circuit-segment granularity -> 130 segments

# --- imagery tile pipeline -------------------------------------------------
TILE_PX = 48             # each pole gets a TILE_PX x TILE_PX raster chip
TILE_M_PER_PX = 1.0      # ground sample distance proxy: 1 m / pixel
NIR_RED_BANDS = ("nir", "red")  # bands synthesized for NDVI

# --- geography (real DTE / SE-Michigan counties & bounding box) ------------
# DTE Electric service territory is Southeast Michigan. Bounding box below is
# the real approximate WGS84 extent of that territory.
TERRITORY_BBOX = {
    "lat_min": 41.70,
    "lat_max": 43.55,
    "lon_min": -84.30,
    "lon_max": -82.30,
}
COUNTIES = [
    "Wayne", "Oakland", "Macomb", "Washtenaw", "Livingston",
    "St. Clair", "Lapeer", "Genesee", "Monroe", "Ingham",
    "Saginaw", "Lenawee",
]

# --- model / validation ----------------------------------------------------
TEST_FRACTION = 0.20         # held-out test fold (seen exactly once)
VAL_FRACTION = 0.20          # validation fold for threshold tuning
N_SPATIAL_FOLDS = 6          # leave-county-group-out spatial backtest folds
N_TEMPORAL_STORMS = 8        # independent storm replays (temporal backtest)
CALIBRATION_BINS = 10        # reliability-table bins

# --- economics (all figures externally cited in README) --------------------
# Cost per truck-roll inspection and avoided outage value per prevented
# pole-failure are grounded in published utility figures (see README sources).
COST_PER_INSPECTION_USD = 220.0     # field truck-roll + crew (industry range)
AVOIDED_COST_PER_FAILURE_USD = 6_800.0  # avg distribution pole-failure event
INSPECTION_BUDGET_FRACTION = 0.20   # utility can inspect ~20% of fleet / cycle

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
NOAA_NORMALS_CSV = os.path.join(DATA_DIR, "noaa_climate_normals_mi.csv")
