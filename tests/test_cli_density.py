"""Intégration CLI : mode --ais-density (raster local + repli 'emodnet' hors-ligne)."""

import numpy as np
import pytest
import xarray as xr

from randi_gen.cli import main
from randi_gen.density import synthetic_density_raster, write_ascii_grid

BBOX = ["41.8", "5.5", "43.4", "9.6"]


@pytest.fixture(scope="module")
def density_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("density")
    r = synthetic_density_raster(bbox=(41.8, 5.5, 43.4, 9.6), cellsize_deg=0.05, seed=11)
    return write_ascii_grid(r, d / "vessel_density_test.asc")


def test_run_with_local_density_raster(tmp_path, density_file):
    rc = main(["run", "--bbox", *BBOX, "--res", "0.2", "--freq", "1000", "--az-step", "10",
               "--ais-density", str(density_file), "--max-ships", "3000",
               "--out", str(tmp_path)])
    assert rc == 0
    f = tmp_path / "randi_map_f1000Hz.nc"
    assert f.exists()
    assert (tmp_path / "density_used.tif").exists() or (tmp_path / "density_used.asc").exists()
    assert (tmp_path / "density_used.png").exists()  # affichage des données brutes
    with xr.open_dataset(f, engine="netcdf4") as ds:
        assert ds["noise_level"].dims == ("lat", "lon", "azimuth")
        assert np.isfinite(ds["noise_level"].values).all()
        assert "vessel-density" in ds.attrs["ais_source"]
        assert int(ds.attrs["n_ships"]) > 100
        assert float(ds["directionality_index"].max()) >= 2.0


def test_density_plot_command(tmp_path, density_file):
    png = tmp_path / "dens.png"
    assert main(["density-plot", str(density_file), "-o", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 10_000


def test_html_with_density_map(tmp_path, density_file):
    rc = main(["run", "--bbox", *BBOX, "--res", "0.25", "--freq", "1000",
               "--ais-density", str(density_file), "--max-ships", "1500",
               "--out", str(tmp_path / "run")])
    assert rc == 0
    nc = tmp_path / "run" / "randi_map_f1000Hz.nc"
    html = tmp_path / "viewer.html"
    assert main(["html", str(nc), "--density", str(density_file), "-o", str(html)]) == 0
    content = html.read_bytes()
    assert b"Vessel density" in content  # carte des données brutes présente


def test_run_emodnet_fallback_offline(tmp_path, monkeypatch):
    from randi_gen import density as dens

    def boom(*a, **k):
        raise RuntimeError("WCS indisponible (test hors-ligne)")

    monkeypatch.setattr(dens, "fetch_emodnet_vessel_density", boom)
    rc = main(["run", "--bbox", *BBOX, "--res", "0.25", "--freq", "1000",
               "--ais-density", "emodnet", "--max-ships", "2000", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "emodnet_vesseldensity_TEST.asc").exists()
    f = tmp_path / "randi_map_f1000Hz.nc"
    with xr.open_dataset(f, engine="netcdf4") as ds:
        assert "synthetic" in ds.attrs["ais_source"] or "vessel-density" in ds.attrs["ais_source"]
        assert int(ds.attrs["n_ships"]) > 50


def test_run_density_per_type(tmp_path, density_file):
    rc = main(["run", "--bbox", *BBOX, "--res", "0.25", "--freq", "1000",
               "--ais-density-type", f"cargo={density_file}",
               "--ais-density-type", f"passenger={density_file}",
               "--max-ships", "2000", "--out", str(tmp_path)])
    assert rc == 0
    with xr.open_dataset(tmp_path / "randi_map_f1000Hz.nc", engine="netcdf4") as ds:
        assert int(ds.attrs["n_ships"]) > 50
        assert "per-type" in ds.attrs["ais_source"]
    # raster combiné (somme des types) sauvegardé et affiché comme en mode --ais-density
    assert (tmp_path / "density_used.tif").exists() or (tmp_path / "density_used.asc").exists()
    assert (tmp_path / "density_used.png").exists()


def test_run_density_per_type_normalizes_kinds(tmp_path, density_file):
    from randi_gen.density import load_density_raster, sample_catalog_from_density

    raster, _ = load_density_raster(str(density_file), bbox=tuple(float(v) for v in BBOX), log=lambda *a: None)
    cat = sample_catalog_from_density(raster, bbox=tuple(float(v) for v in BBOX), fixed_type="sailing")
    assert set(np.unique(cat.kind)) == {"other"}  # normalisé vers les 5 classes du modèle
