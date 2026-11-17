import os

import numpy as np
import pytest

from randi_gen import density as dens
from randi_gen.density import (DensityRaster, load_density_raster, sample_catalog_from_density,
                               synthetic_density_raster, write_ascii_grid, write_geotiff)

BBOX = (41.8, 5.5, 43.4, 9.6)


def _tiny_raster():
    lats = np.array([42.02, 42.01, 42.00])
    lons = np.array([6.98, 6.99, 7.00])
    values = np.array([[0.0, 0.0, 0.0],
                       [0.0, 7200.0, 0.0],   # 7200 h/km²/mois sur ~0,9 km² -> ~9 navires dans la cellule centrale
                       [0.0, 0.0, 0.0]])
    area = dens._make_area(lats, 3, 0.01, 0.01)
    return DensityRaster(values=values, lats=lats, lons=lons, area_km2=area,
                         dlat_deg=0.01, dlon_deg=0.01, crs="EPSG:4326", source="tiny")


def test_ascii_roundtrip(tmp_path):
    r = synthetic_density_raster(bbox=(42.0, 6.0, 43.0, 8.0), cellsize_deg=0.05, seed=1)
    p = write_ascii_grid(r, tmp_path / "d.asc")
    r2 = dens.from_ascii_grid(p)
    assert r2.shape == r.shape
    assert np.allclose(r2.lats, r.lats, atol=1e-9)
    assert np.allclose(r2.lons, r.lons, atol=1e-9)
    assert np.allclose(np.nan_to_num(r2.values), np.nan_to_num(r.values), atol=1e-3)


def test_geotiff_roundtrip(tmp_path):
    pytest.importorskip("rasterio")
    r = synthetic_density_raster(bbox=(42.0, 6.0, 43.0, 8.0), cellsize_deg=0.05, seed=1)
    p = write_geotiff(r, tmp_path / "d.tif")
    r2 = dens.from_geotiff(p)
    assert r2.shape == r.shape
    assert np.allclose(r2.lats, r.lats, atol=1e-6)
    assert np.allclose(r2.lons, r.lons, atol=1e-6)
    assert np.allclose(np.nan_to_num(r2.values), np.nan_to_num(r.values), atol=1e-2)


def test_synthetic_raster_has_lanes_and_ports():
    r = synthetic_density_raster(bbox=BBOX, cellsize_deg=0.02, seed=3)
    assert r.shape[0] > 50 and r.shape[1] > 100
    assert np.nanmax(r.values) > 100  # ports
    # la densité le long de la ligne Toulon-Ajaccio est bien au-dessus du fond
    assert np.nanmax(r.values) > 10 * np.nanmedian(r.values)


def test_sample_catalog_concentration_and_determinism():
    r = _tiny_raster()
    c1 = sample_catalog_from_density(r, seed=5, hours_per_month=720.0, max_ships=10**9)
    c2 = sample_catalog_from_density(r, seed=5, hours_per_month=720.0, max_ships=10**9)
    assert len(c1) > 0 and len(c1) == len(c2)
    assert np.array_equal(c1.lat, c2.lat)
    # tous les navires sont dans la cellule centrale (lat ~42.01, lon ~6.99)
    assert np.all(np.abs(c1.lat - 42.01) <= 0.005)
    assert np.all(np.abs(c1.lon - 6.99) <= 0.005)
    # densité x10 -> environ x10 navires
    c10 = sample_catalog_from_density(r, seed=5, hours_per_month=720.0, scale=10.0, max_ships=10**9)
    assert len(c10) > 3 * len(c1)


def test_sample_max_ships_cap():
    r = _tiny_raster()
    c = sample_catalog_from_density(r, seed=5, scale=1000.0, max_ships=50)
    assert len(c) <= 50


def test_sample_fixed_type():
    r = _tiny_raster()
    c = sample_catalog_from_density(r, seed=5, fixed_type="cargo")
    assert set(np.unique(c.kind)) == {"cargo"}
    assert np.all((c.speed_kn >= 2.0) & (c.speed_kn <= 25.0))


def test_clip_bbox():
    r = synthetic_density_raster(bbox=BBOX, cellsize_deg=0.05, seed=1)
    c = r.clip_bbox(42.5, 6.5, 43.0, 7.5)
    assert c.lats.min() >= 42.5 and c.lats.max() <= 43.0
    assert c.lons.min() >= 6.5 and c.lons.max() <= 7.5


def test_load_density_raster_local_ascii(tmp_path):
    r = synthetic_density_raster(bbox=BBOX, cellsize_deg=0.1, seed=1)
    p = write_ascii_grid(r, tmp_path / "d.asc")
    r2, note = load_density_raster(str(p), bbox=BBOX, log=lambda *a: None)
    assert note == "file"
    assert r2.shape == r.shape


def test_load_density_raster_emodnet_fallback(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("réseau indisponible (test)")

    monkeypatch.setattr(dens, "fetch_emodnet_vessel_density", boom)
    r, note = load_density_raster("emodnet", bbox=BBOX, outdir=tmp_path, seed=2, log=lambda *a: None)
    assert note == "synthetic-fallback"
    assert "synthetic" in r.source
    assert (tmp_path / "emodnet_vesseldensity_TEST.asc").exists()
    assert np.nanmax(r.values) > 0


@pytest.mark.skipif(os.environ.get("RANDI_GEN_ONLINE") != "1", reason="test en ligne (RANDI_GEN_ONLINE=1)")
def test_fetch_emodnet_online():
    r = dens.fetch_emodnet_vessel_density((42.5, 6.5, 43.0, 7.5), layer="vesseldensity_allavg")
    assert r.shape[0] > 10 and r.shape[1] > 10
    assert np.nanmax(r.values) >= 0


def test_resolve_coverage_id_exact_per_type():
    ids = ["emodnet__vesseldensity_allavg", "emodnet__vesseldensity_09avg",
           "emodnet__vesseldensity_10avg", "emodnet__vesseldensity_01avg", "emodnet__autre_couche"]
    # nom complet -> exact
    assert dens._resolve_coverage_id(ids, "emodnet__vesseldensity_09avg") == "emodnet__vesseldensity_09avg"
    # nom court -> suffixe, JAMAIS le repli allavg (bug : cargo chargeait all-types)
    assert dens._resolve_coverage_id(ids, "vesseldensity_09avg") == "emodnet__vesseldensity_09avg"
    assert dens._resolve_coverage_id(ids, "vesseldensity_10avg") == "emodnet__vesseldensity_10avg"
    assert dens._resolve_coverage_id(ids, "vesseldensity_allavg") == "emodnet__vesseldensity_allavg"
    with pytest.raises(RuntimeError):
        dens._resolve_coverage_id(ids, "vesseldensity_99avg")


def test_combine_rasters():
    import copy

    r1 = _tiny_raster()
    r2 = copy.copy(r1)
    r2.values = np.where(np.isfinite(r1.values), r1.values * 2.0, np.nan)
    comb = dens.combine_rasters([r1, r2])
    assert comb is not None
    assert np.allclose(comb.values, np.where(np.isfinite(r1.values), r1.values * 3.0, np.nan), equal_nan=True)
    # NaN partout -> NaN conservé
    r3 = copy.copy(r1)
    r3.values = np.full(r1.shape, np.nan)
    comb2 = dens.combine_rasters([r1, r3])
    assert np.allclose(comb2.values, r1.values, equal_nan=True)
    # grilles différentes -> None
    r4 = copy.copy(r1)
    r4.lats = r1.lats + 0.001
    assert dens.combine_rasters([r1, r4]) is None
    assert dens.combine_rasters([]) is None


def test_emodnet_type_layer_all_codes():
    # les 13 codes EMODnet 00-12 ont leur couche dédiée
    assert len(dens.EMODNET_TYPE_LAYER) == len(dens.EMODNET_TYPE_CODE) == 13
    assert dens.EMODNET_TYPE_LAYER["cargo"] == "vesseldensity_09avg"
    assert dens.EMODNET_TYPE_LAYER["tanker"] == "vesseldensity_10avg"
    assert dens.EMODNET_TYPE_LAYER["passenger"] == "vesseldensity_08avg"
    assert dens.EMODNET_TYPE_LAYER["fishing"] == "vesseldensity_01avg"
    assert dens.EMODNET_TYPE_LAYER["other"] == "vesseldensity_00avg"


def test_plot_density_raster(tmp_path):
    from randi_gen.visualize import plot_density_raster

    r = synthetic_density_raster(bbox=BBOX, cellsize_deg=0.05, seed=1)
    out = tmp_path / "density.png"
    fig, _ = plot_density_raster(r, out)
    assert out.exists() and out.stat().st_size > 10_000
    import matplotlib.pyplot as plt

    plt.close(fig)
