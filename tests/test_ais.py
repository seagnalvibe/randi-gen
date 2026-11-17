import numpy as np
import pytest

from randi_gen.ais import ShipCatalog, load_ais, synthetic_catalog

BBOX = (41.8, 5.5, 43.4, 9.6)


def test_synthetic_catalog_deterministic_and_in_bbox():
    c1 = synthetic_catalog(bbox=BBOX, seed=3)
    c2 = synthetic_catalog(bbox=BBOX, seed=3)
    assert len(c1) > 100
    assert np.array_equal(c1.lat, c2.lat)
    assert c1.lat.min() >= BBOX[0] - 0.5
    assert c1.lat.max() <= BBOX[2] + 0.5
    assert c1.lon.min() >= BBOX[1] - 0.5
    assert c1.lon.max() <= BBOX[3] + 0.5
    assert set(np.unique(c1.kind)) <= {"tanker", "cargo", "passenger", "fishing", "other"}
    assert np.all(c1.speed_kn > 0)


def test_load_ais_csv(tmp_path):
    csv = tmp_path / "ais.csv"
    csv.write_text(
        "longitude,latitude,ship_type,sog\n"
        "6.10,43.00,Ferry,18\n"
        "7.20,42.50,cargo,14\n"
        "bad,row,ignored,\n"
    )
    cat = load_ais(file=str(csv), bbox=BBOX)
    assert len(cat) == 2
    assert set(cat.kind) == {"passenger", "cargo"}
    assert float(cat.speed_kn[0]) == 18.0


def test_load_ais_geojson_url_like(tmp_path):
    gj = tmp_path / "ais.geojson"
    gj.write_text(
        '{"type":"FeatureCollection","features":['
        '{"type":"Feature","geometry":{"type":"Point","coordinates":[8.0,42.9]},"properties":{"type":"tanker","sog":10}}]}'
    )
    cat = load_ais(file=str(gj), bbox=BBOX)
    assert len(cat) == 1
    assert cat.kind[0] == "tanker"


def test_load_ais_unknown_source():
    with pytest.raises(ValueError):
        load_ais(source="nulle-part")


def test_filter_bbox():
    cat = ShipCatalog(lat=np.array([0.0, 43.0]), lon=np.array([0.0, 6.0]),
                      kind=np.array(["other", "cargo"]), speed_kn=np.array([10.0, 10.0]), source="t")
    f = cat.filter_bbox(*BBOX)
    assert len(f) == 1
    assert f.kind[0] == "cargo"
