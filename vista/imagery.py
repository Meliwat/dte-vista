"""The imagery pipeline - VISTA's distinct contribution.

The other DTE variants treat "satellite imagery / NDVI" as a single scalar
column. VISTA actually *renders raster tiles and runs computer vision on the
pixels*. For every pole we synthesize a TILE_PX x TILE_PX two-band chip
(NIR + Red, the bands you need for NDVI) at two epochs (t0 = baseline,
t1 = current), from a small set of latent ground-truth scene parameters.

Crucially the risk model never sees those latent parameters. It only sees
features the extractor recovers *from the pixels*, exactly as it would from a
real Sentinel-2 / NAIP / Planet tile:

  1. NDVI canopy statistics            (vegetation vigor & density)
  2. Right-of-way encroachment         (canopy mass inside the conductor swath)
  3. Overhang / strike geometry        (tall canopy directly over the line)
  4. Pole lean angle                   (Hough-style dominant near-vertical line)
  5. Bi-temporal change detection      (NDVI growth in the RoW, t0 -> t1)
  6. Texture / heterogeneity           (canopy roughness = limb-failure proxy)

This is the technique rival variants lack: a real (if synthetic-sourced)
image-derived feature stack with an explicit swap seam to drop in real GeoTIFF
tiles (README "Technology Utilized" -> Imagery seam).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import TILE_PX


@dataclass(frozen=True)
class SceneLatents:
    """Ground-truth scene knobs used ONLY to render pixels (model never sees)."""
    canopy_density: float      # 0..1 overall vegetation fraction in the tile
    canopy_height: float       # 0..1 proxy for tree height near the line
    row_intrusion: float       # 0..1 how far canopy intrudes into RoW swath
    growth_rate: float         # 0..1 NDVI increase t0->t1 inside the swath
    pole_lean_deg: float       # degrees of pole tilt from vertical (0..18)
    canopy_roughness: float    # 0..1 limb/crown heterogeneity
    defoliation: float         # 0..1 stress/dieback (lowers NDVI, raises risk)


def _disk(cx: float, cy: float, r: float, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def render_tile(latents: SceneLatents, rng: np.random.Generator
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Render a 2-band (NIR, Red) tile at epoch t0 and t1.

    Geometry convention: the power line runs vertically through the tile
    center column; the "right-of-way swath" is the vertical band +/- W around
    it. The pole is a near-vertical bright structural member near center.
    """
    n = TILE_PX
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = n / 2.0

    def build(epoch: int) -> np.ndarray:
        # --- ground / soil background (low NIR, moderate red) --------------
        soil_nir = 0.18 + 0.04 * rng.standard_normal((n, n))
        soil_red = 0.22 + 0.04 * rng.standard_normal((n, n))

        # --- vegetation field: clustered canopy blobs ---------------------
        veg = np.zeros((n, n))
        n_blobs = int(6 + 26 * latents.canopy_density)
        for _ in range(n_blobs):
            bx = rng.uniform(0, n)
            by = rng.uniform(0, n)
            br = rng.uniform(2.0, 4.0 + 6.0 * latents.canopy_height)
            amp = rng.uniform(0.5, 1.0) * (0.6 + 0.4 * latents.canopy_density)
            veg += amp * np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * br * br)))

        # canopy intrudes into the right-of-way swath near the line
        swath_half = 4.0 + 9.0 * latents.row_intrusion
        in_swath = np.abs(xx - cx) <= swath_half
        # t1 grows vegetation inside the swath (encroachment over time)
        growth = latents.growth_rate * (0.5 if epoch == 1 else 0.0)
        veg = veg + in_swath * (0.35 * latents.row_intrusion + growth) * \
            np.exp(-((xx - cx) ** 2) / (2 * (swath_half ** 2)))

        # crown roughness / heterogeneity
        veg += latents.canopy_roughness * 0.30 * rng.standard_normal((n, n)) * (veg > 0.1)
        veg = np.clip(veg, 0.0, 1.6)

        # stress / dieback lowers NIR response of vegetation
        veg_health = 1.0 - 0.7 * latents.defoliation

        # spectral mixing: healthy veg -> high NIR, low red
        veg_nir = 0.55 * veg * veg_health
        veg_red = -0.10 * veg * veg_health
        nir = np.clip(soil_nir + veg_nir, 0.0, 1.0)
        red = np.clip(soil_red + veg_red, 0.02, 1.0)

        # --- structural members: conductor + leaning pole -----------------
        # conductor: thin bright vertical line (metallic -> moderate NIR/red)
        cond = (np.abs(xx - cx) <= 0.9)
        nir = np.where(cond, 0.40, nir)
        red = np.where(cond, 0.42, red)

        # pole: bright, slightly wider, leaning by pole_lean_deg from vertical.
        lean = np.tan(np.radians(latents.pole_lean_deg))
        # x position of the pole center at row y (top fixed near center)
        pole_x = cx + 2.0 + lean * (yy - 2.0)
        pole_mask = (np.abs(xx - pole_x) <= 1.4) & (yy >= 2) & (yy <= n - 3)
        nir = np.where(pole_mask, 0.52, nir)
        red = np.where(pole_mask, 0.50, red)

        # acquisition noise (sensor)
        nir = np.clip(nir + 0.012 * rng.standard_normal((n, n)), 0.0, 1.0)
        red = np.clip(red + 0.012 * rng.standard_normal((n, n)), 0.02, 1.0)
        return np.stack([nir, red], axis=0)

    t0 = build(0)
    t1 = build(1)
    return t0, t1


# --------------------------------------------------------------------------
# Feature extraction - operates ONLY on rendered pixels (no latents).
# --------------------------------------------------------------------------
def _ndvi(tile: np.ndarray) -> np.ndarray:
    nir, red = tile[0], tile[1]
    return (nir - red) / (nir + red + 1e-6)


def _hough_lean_deg(ndvi_low_mask: np.ndarray) -> float:
    """Recover the dominant near-vertical structural line angle from pixels.

    A compact Hough-style accumulator over candidate lean angles: structural
    pixels (low-NDVI bright members) are projected; the angle whose slanted
    column accumulates the most structural mass wins. This is genuine geometry
    recovered from the image, not the latent lean value.
    """
    n = ndvi_low_mask.shape[0]
    ys, xs = np.nonzero(ndvi_low_mask)
    if xs.size < 8:
        return 0.0
    cx = n / 2.0
    best_angle, best_score = 0.0, -1.0
    for deg in np.arange(-18.0, 18.01, 1.0):
        lean = np.tan(np.radians(deg))
        pred_x = cx + 2.0 + lean * (ys - 2.0)
        score = float(np.sum(np.abs(xs - pred_x) <= 1.6))
        if score > best_score:
            best_score, best_angle = score, deg
    return abs(float(best_angle))


def extract_features(t0: np.ndarray, t1: np.ndarray) -> Dict[str, float]:
    """Derive the image feature stack from the two rendered epochs."""
    n = TILE_PX
    cx = n / 2.0
    yy, xx = np.mgrid[0:n, 0:n]

    ndvi1 = _ndvi(t1)
    ndvi0 = _ndvi(t0)

    # canopy = vigorous-vegetation pixels (NDVI threshold)
    veg_mask1 = ndvi1 > 0.20
    canopy_frac = float(veg_mask1.mean())
    ndvi_mean = float(ndvi1[veg_mask1].mean()) if veg_mask1.any() else 0.0
    ndvi_p90 = float(np.percentile(ndvi1[veg_mask1], 90)) if veg_mask1.any() else 0.0

    # right-of-way swath = central vertical band; encroachment = canopy mass in it
    swath = np.abs(xx - cx) <= 6.0
    row_canopy = float((veg_mask1 & swath).sum()) / float(swath.sum())
    # overhang: tall/dense canopy directly over the line (inner +/-2 px)
    inner = np.abs(xx - cx) <= 2.0
    overhang = float(ndvi1[inner & veg_mask1].mean()) if (inner & veg_mask1).any() else 0.0
    overhang *= float((veg_mask1 & inner).sum()) / max(1.0, float(inner.sum()))

    # bi-temporal change detection: NDVI growth INSIDE the swath, t0 -> t1
    d_ndvi_swath = float((ndvi1 - ndvi0)[swath].mean())
    row_growth = max(0.0, d_ndvi_swath)

    # structural members = bright low-NDVI pixels -> recover pole lean
    struct_mask = (ndvi1 < 0.05) & (t1[0] > 0.45)
    pole_lean = _hough_lean_deg(struct_mask)

    # canopy texture / roughness = local std of NDVI over vegetation
    if veg_mask1.any():
        gy, gx = np.gradient(ndvi1)
        roughness = float(np.sqrt(gx ** 2 + gy ** 2)[veg_mask1].mean())
    else:
        roughness = 0.0

    # vegetation-stress index: low NDVI among canopy pixels = dieback/limb risk
    stress = float(np.clip(0.55 - ndvi_mean, 0.0, 0.55) / 0.55) if veg_mask1.any() else 0.0

    return {
        "img_canopy_frac": round(canopy_frac, 5),
        "img_ndvi_mean": round(ndvi_mean, 5),
        "img_ndvi_p90": round(ndvi_p90, 5),
        "img_row_canopy": round(row_canopy, 5),
        "img_overhang": round(overhang, 5),
        "img_row_growth": round(row_growth, 5),
        "img_pole_lean_deg": round(pole_lean, 5),
        "img_canopy_roughness": round(roughness, 5),
        "img_veg_stress": round(stress, 5),
    }


IMAGE_FEATURES = [
    "img_canopy_frac", "img_ndvi_mean", "img_ndvi_p90", "img_row_canopy",
    "img_overhang", "img_row_growth", "img_pole_lean_deg",
    "img_canopy_roughness", "img_veg_stress",
]
