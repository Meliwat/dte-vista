"""Load the small REAL bundled public artifact: NOAA 1991-2020 Climate Normals
for Michigan stations in/near DTE Electric territory.

This is the single non-synthetic input wired into VISTA. It is genuine
public-domain U.S. Government data (NOAA NCEI). Every synthetic pole is
geographically snapped to its nearest real NOAA station and inherits that
station's REAL annual precipitation / snowfall / temperature / wind normals,
so the weather layer of the risk model is anchored to measured climatology
rather than invented numbers.

Swap seam: to use the full live product instead of this bundled annual sample,
replace `load_noaa_normals()` with an NCEI `access/services` query (documented
in README "Technology Utilized" -> Data seam). The downstream interface
(`nearest_station_features`) is unchanged.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import List

from .config import NOAA_NORMALS_CSV


@dataclass(frozen=True)
class Station:
    station_id: str
    name: str
    county: str
    lat: float
    lon: float
    ann_prcp_in: float
    ann_snow_in: float
    ann_tavg_f: float
    ann_wdmv_mph: float


def load_noaa_normals() -> List[Station]:
    """Parse the bundled real NOAA climate-normals CSV (comment lines start #)."""
    stations: List[Station] = []
    with open(NOAA_NORMALS_CSV, "r", encoding="utf-8") as fh:
        rows = [ln for ln in fh if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(rows)
    for r in reader:
        stations.append(
            Station(
                station_id=r["station_id"],
                name=r["name"],
                county=r["county"],
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                ann_prcp_in=float(r["ann_prcp_in"]),
                ann_snow_in=float(r["ann_snow_in"]),
                ann_tavg_f=float(r["ann_tavg_f"]),
                ann_wdmv_mph=float(r["ann_wdmv_mph"]),
            )
        )
    if not stations:
        raise RuntimeError("No NOAA stations parsed - bundled artifact missing.")
    return stations


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_station(lat: float, lon: float, stations: List[Station]) -> Station:
    """Nearest real NOAA station to an arbitrary pole location (great-circle)."""
    return min(stations, key=lambda s: _haversine_km(lat, lon, s.lat, s.lon))


def normals_summary(stations: List[Station]) -> dict:
    """Aggregate stats over the real artifact (used in README / provenance)."""
    n = len(stations)
    return {
        "n_stations": n,
        "mean_ann_prcp_in": round(sum(s.ann_prcp_in for s in stations) / n, 2),
        "mean_ann_snow_in": round(sum(s.ann_snow_in for s in stations) / n, 2),
        "mean_ann_tavg_f": round(sum(s.ann_tavg_f for s in stations) / n, 2),
        "counties": sorted({s.county for s in stations}),
    }
