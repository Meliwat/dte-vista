# PoleProof Public Data Slice

Wayne County is the real public-data proof slice. The statewide pole inventory remains synthetic for demo coverage.

## Embedded Evidence

- NOAA NCEI Storm Events bulk CSV: 75 Wayne County records from 2023 through January 2026.
- FEMA NFHL MapServer layer 28: 4,397 flood-hazard zone features intersect the Detroit/Wayne working envelope.
- USDA NRCS Soil Data Access: SSURGO area `MI163`, Wayne County, Michigan; drainage table includes 313 somewhat poorly drained, 83 poorly drained, 54 hydric poorly drained, and 21 hydric very poorly drained components.
- USGS 3DEP TNM API: two 1/3 arc-second GeoTIFF tiles cover the Wayne proof envelope, `n43w083` and `n43w084`.
- Public imagery stack: Landsat, NAIP, and Sentinel are represented as the vegetation/canopy source family; the demo stores a canopy proxy, not image tiles.
- OSM Overpass: 49,058 open power pole/tower/line features returned in the Wayne bounding box; used as open infrastructure context, not DTE asset truth.

## Honest Submission Wording

Statewide pole inventory is synthetic for demo coverage. One county slice demonstrates the same receipt pipeline with real public data. DTE asset, outage, and inspection data would convert this into a production prioritization layer.
