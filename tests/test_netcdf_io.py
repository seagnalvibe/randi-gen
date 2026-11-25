import numpy as np
import pytest
import xarray as xr

from randi_gen.ais import ShipCatalog
from randi_gen.model import ModelParams, compute_directional_noise
from randi_gen.netcdf_io import field_to_dataset, output_path, read_field, write_field


def _field():
    ships = ShipCatalog(lat=np.array([43.0, 42.0]), lon=np.array([7.0, 8.5]),
                        kind=np.array(["cargo", "passenger"]), speed_kn=np.array([15.0, 18.0]),
                        source="test")
    f = compute_directional_noise([43.0, 42.5], [6.0, 7.0], ships, 1000.0, ModelParams(n_azimuth=36))
    f.meta.update(ais_source=ships.source, bbox=[41.8, 5.5, 43.4, 9.6], resolution_deg=0.5)
    return f


def test_dataset_structure_3d():
    ds = field_to_dataset(_field())
    assert ds["noise_level"].dims == ("lat", "lon", "azimuth")
    assert ds["noise_level"].shape == (2, 2, 36)
    assert ds.sizes["azimuth"] == 36
    assert ds["omnidirectional_level"].dims == ("lat", "lon")
    assert ds["directionality_index"].dims == ("lat", "lon")
    assert float(ds.attrs["frequency_hz"]) == 1000.0
    assert ds.attrs["Conventions"] == "CF-1.8"
    assert ds["noise_level"].attrs["units"] == "dB re 1 uPa^2/Hz"
    assert ds["azimuth"].attrs["units"] == "degree"
    assert ds["lat"].attrs["units"] == "degrees_north"


def test_roundtrip(tmp_path):
    f = _field()
    p = write_field(f, tmp_path / "map.nc")
    assert p.exists()
    g = read_field(p)
    assert g.noise_level.shape == f.noise_level.shape
    assert np.allclose(g.noise_level, f.noise_level, atol=1e-4)  # float32
    assert g.frequency_hz == f.frequency_hz
    assert np.allclose(g.azimuth, f.azimuth)
    assert g.meta["ais_source"] == "test"


def test_output_path():
    assert output_path(1000.0, "out").name == "randi_map_f1000Hz.nc"
    assert output_path(12500.5, "out").name == "randi_map_f12500.5Hz.nc"


def test_netcdf_file_is_valid_netcdf(tmp_path):
    p = write_field(_field(), tmp_path / "m.nc")
    with xr.open_dataset(p, engine="netcdf4") as ds:
        assert "noise_level" in ds
        assert np.isfinite(ds["noise_level"].values).all()
