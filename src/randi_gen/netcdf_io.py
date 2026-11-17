"""Écriture/lecture NetCDF : carte 3D (dimensions lat, lon, azimuth)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from . import __version__
from .model import DirectionalField

LEVEL_UNITS = "dB re 1 uPa^2/Hz"


def output_path(freq_hz, outdir):
    return Path(outdir) / f"randi_map_f{float(freq_hz):g}Hz.nc"


def field_to_dataset(field: DirectionalField) -> xr.Dataset:
    ds = xr.Dataset(
        data_vars={
            "noise_level": (("lat", "lon", "azimuth"), np.asarray(field.noise_level, dtype=np.float32)),
            "omnidirectional_level": (("lat", "lon"), np.asarray(field.omnidirectional, dtype=np.float32)),
            "directionality_index": (("lat", "lon"), np.asarray(field.directionality_index, dtype=np.float32)),
        },
        coords={
            "lat": ("lat", np.asarray(field.lat, float),
                    {"standard_name": "latitude", "units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", np.asarray(field.lon, float),
                    {"standard_name": "longitude", "units": "degrees_east", "long_name": "longitude"}),
            "azimuth": ("azimuth", np.asarray(field.azimuth, float),
                        {"units": "degree", "long_name": "azimuth from receiver toward noise source, 0 = north, clockwise"}),
        },
    )
    ds["noise_level"].attrs = {
        "units": LEVEL_UNITS,
        "long_name": "directional ambient noise spectrum level (RANDI-type prediction)",
    }
    ds["omnidirectional_level"].attrs = {"units": LEVEL_UNITS, "long_name": "omnidirectional ambient noise spectrum level"}
    ds["directionality_index"].attrs = {"units": "dB", "long_name": "directionality index = 10*log10(max/min over azimuth)"}

    meta = dict(field.meta)
    bbox = meta.get("bbox", [])
    ds.attrs = {
        "Conventions": "CF-1.8",
        "title": "Directional underwater ambient noise map (RANDI-type, AIS-driven)",
        "summary": "3D map (lat, lon, azimuth) of predicted directional ambient noise spectrum level",
        "frequency_hz": float(field.frequency_hz),
        "frequency_units": "Hz",
        "model": "RANDI-type: incoherent sum of AIS ship sources per azimuth sector with range-dependent transmission loss",
        "level_units": LEVEL_UNITS,
        "ais_source": str(meta.get("ais_source", "unknown")),
        "n_ships": int(meta.get("n_ships", 0)),
        "bbox_south_west_north_east": [float(v) for v in bbox] if bbox else [],
        "grid_resolution_deg": float(meta.get("resolution_deg", np.nan)),
        "azimuth_step_deg": float(meta.get("azimuth_step_deg", np.nan)),
        "max_range_km": float(meta.get("max_range_km", np.nan)),
        "spreading_exponent": float(meta.get("spreading", np.nan)),
        "min_range_m": float(meta.get("min_range_m", np.nan)),
        "references": "Wenz (1962); Ross (1976); Thorp (1967); Ainslie & McColm (1998)",
        "history": f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} created by randi-gen {__version__}",
        "generator": f"randi-gen {__version__}",
    }
    return ds


def write_field(field: DirectionalField, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "azimuth_step_deg" not in field.meta and field.azimuth.size > 1:
        field.meta["azimuth_step_deg"] = float(field.azimuth[1] - field.azimuth[0])
    field_to_dataset(field).to_netcdf(path, engine="netcdf4")
    return path


def read_field(path) -> DirectionalField:
    with xr.open_dataset(path, engine="netcdf4") as ds:
        ds = ds.load()
    return DirectionalField(
        lat=ds["lat"].values,
        lon=ds["lon"].values,
        azimuth=ds["azimuth"].values,
        noise_level=ds["noise_level"].values,
        frequency_hz=float(ds.attrs["frequency_hz"]),
        omnidirectional=ds["omnidirectional_level"].values,
        directionality_index=ds["directionality_index"].values,
        meta=dict(ds.attrs),
    )
