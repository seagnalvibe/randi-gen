"""randi-gen — cartes 3D (lat, lon, azimut) de bruit ambiant sous-marin directif (type RANDI)."""

__version__ = "0.1.0"

from .model import DirectionalField, ModelParams, compute_directional_noise, make_grid
from .ais import ShipCatalog, load_ais, synthetic_catalog
from .density import (DensityRaster, fetch_emodnet_vessel_density, load_density_raster,
                      sample_catalog_from_density, synthetic_density_raster)
from .netcdf_io import read_field, write_field

__all__ = [
    "__version__",
    "DirectionalField",
    "ModelParams",
    "compute_directional_noise",
    "make_grid",
    "ShipCatalog",
    "load_ais",
    "synthetic_catalog",
    "DensityRaster",
    "fetch_emodnet_vessel_density",
    "load_density_raster",
    "sample_catalog_from_density",
    "synthetic_density_raster",
    "read_field",
    "write_field",
]
