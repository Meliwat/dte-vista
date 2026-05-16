"""The bundled REAL public artifact must parse and be sane."""

from vista.noaa import (
    load_noaa_normals,
    nearest_station,
    normals_summary,
)


def test_real_noaa_artifact_loads():
    st = load_noaa_normals()
    assert len(st) >= 10, "expected >=10 real Michigan NOAA stations"
    # Detroit Metro is a real station with a well-known normal precip ~34 in.
    dtw = [s for s in st if s.station_id == "USW00094847"]
    assert dtw, "Detroit Metro (USW00094847) must be present"
    assert 25.0 < dtw[0].ann_prcp_in < 45.0
    assert 30.0 < dtw[0].ann_snow_in < 70.0
    assert 40.0 < dtw[0].ann_tavg_f < 55.0


def test_real_noaa_values_are_physical():
    for s in load_noaa_normals():
        assert 41.0 < s.lat < 44.5          # Michigan latitude band
        assert -85.0 < s.lon < -82.0        # SE-Michigan longitude band
        assert 20.0 < s.ann_prcp_in < 50.0
        assert 25.0 < s.ann_snow_in < 80.0
        assert 40.0 < s.ann_tavg_f < 56.0
        assert 6.0 < s.ann_wdmv_mph < 14.0


def test_nearest_station_is_great_circle_nearest():
    st = load_noaa_normals()
    # a point right at Detroit Metro should snap to Detroit Metro
    near = nearest_station(42.2313, -83.3308, st)
    assert near.station_id == "USW00094847"


def test_normals_summary_aggregates():
    s = normals_summary(load_noaa_normals())
    assert s["n_stations"] >= 10
    assert 25.0 < s["mean_ann_prcp_in"] < 45.0
    assert len(s["counties"]) >= 8
