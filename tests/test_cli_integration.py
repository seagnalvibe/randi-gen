"""Test d'intégration : cas test entre Toulon et la Corse (bbox par défaut)."""

import math

import numpy as np
import pytest
import xarray as xr

from randi_gen.cli import main

BBOX = ["41.8", "5.5", "43.4", "9.6"]
PORTS = [(43.1244, 5.9299), (43.2965, 5.3698), (41.9267, 8.7386), (42.7028, 9.4501)]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("toulon_corse")
    rc = main([
        "run",
        "--bbox", *BBOX,
        "--res", "0.2",
        "--freq", "1000",
        "--az-step", "10",
        "--ais", "synthetic",
        "--seed", "7",
        "--out", str(out),
        "--figures",
        "--html",
    ])
    assert rc == 0
    return out


def test_netcdf_3d_structure(run_dir):
    files = list(run_dir.glob("randi_map_f1000Hz.nc"))
    assert len(files) == 1
    with xr.open_dataset(files[0], engine="netcdf4") as ds:
        assert ds["noise_level"].dims == ("lat", "lon", "azimuth")
        assert ds.sizes["azimuth"] == 36
        assert ds.sizes["lat"] == 9    # 41.8 -> 43.4 par pas de 0.2
        assert ds.sizes["lon"] == 21   # 5.5 -> 9.6
        assert np.isfinite(ds["noise_level"].values).all()
        assert float(ds.attrs["frequency_hz"]) == 1000.0
        assert int(ds.attrs["n_ships"]) > 50
        assert "synthetic" in ds.attrs["ais_source"]


def test_directionality_and_hotspot(run_dir):
    f = run_dir / "randi_map_f1000Hz.nc"
    with xr.open_dataset(f, engine="netcdf4") as ds:
        di = ds["directionality_index"].values
        omni = ds["omnidirectional_level"].values
        assert float(di.max()) >= 2.0  # la carte est bien directive
        assert float(omni.max()) > float(omni.min()) + 5.0
        i, j = np.unravel_index(int(np.argmax(omni)), omni.shape)
        lat, lon = float(ds["lat"][i]), float(ds["lon"][j])
        dport = min(math.hypot(lat - p[0], lon - p[1]) for p in PORTS)
        assert dport < 0.4  # le max omnidirectionnel est près d'un port (Toulon/Marseille/Ajaccio/Bastia)


def test_figures_and_html(run_dir):
    figs = list(run_dir.glob("randi_map_f1000Hz_figures/*.png"))
    assert len(figs) >= 6
    html = run_dir / "viewer.html"
    assert html.exists()
    assert "plotly" in html.read_text(encoding="utf-8", errors="ignore").lower()


def test_plot_and_html_commands(run_dir, tmp_path):
    f = str(run_dir / "randi_map_f1000Hz.nc")
    png = tmp_path / "map.png"
    assert main(["plot", f, "--azimuth", "90", "--polar", "43.0", "6.5", "--3d", "-o", str(png)]) == 0
    assert png.exists()
    assert (tmp_path / "map_polar.png").exists()
    assert (tmp_path / "map_3d.png").exists()
    html = tmp_path / "v.html"
    assert main(["html", f, "-o", str(html)]) == 0
    assert html.exists() and html.stat().st_size > 100_000
