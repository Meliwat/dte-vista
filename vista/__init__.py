"""VISTA - imagery-led predictive utility-pole risk profiling for the DTE grid.

VISTA (Vegetation & Imagery Structural Threat Analytics) is the imagery-modality
reading of the DTE "Utility Pole Risk Profiling" brief: it builds a real
computer-vision-style feature pipeline over (synthetic-but-physical) raster
tiles - NDVI canopy encroachment, pole lean from edge geometry, and bi-temporal
right-of-way change detection - fuses those image-derived features with public
weather / flood / soil / geo layers (including a small REAL bundled NOAA
climate-normals artifact), and produces an explainable, calibrated per-pole and
per-segment failure-risk score on a map-based dashboard.

Everything is offline, deterministic and key-free.
"""

__version__ = "1.0.0"
