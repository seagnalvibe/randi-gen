"""Modèle de bruit ambiant directif de type RANDI.

Pour chaque cellule (lat, lon) réceptrice et chaque secteur d'azimut :

    NL(lat, lon, az, f) = 10*log10( sum_navires(az) 10^((SL(f) - TL(r, f))/10) + P_fond(f) )

avec SL niveau de source du navire (Ross/Wenz), TL perte de transmission
(étalement + absorption de Thorp), r distance navire-récepteur, P_fond niveau
de fond diffus. Les navires sont répartis dans les secteurs selon leur cap
vu depuis le récepteur (0 = Nord, sens horaire).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .acoustics import background_level_db, source_level_db, transmission_loss_db
from .geo import bearing_deg, haversine_km


@dataclass
class ModelParams:
    azimuth_step_deg: float = 10.0
    max_range_km: float = 300.0
    min_range_m: float = 500.0
    spreading: float = 1.5  # 1 = cylindrique, 2 = sphérique
    chunk_cells: int = 256


@dataclass
class DirectionalField:
    """Champ 3D : noise_level[lat, lon, azimuth] en dB re 1 µPa²/Hz."""

    lat: np.ndarray
    lon: np.ndarray
    azimuth: np.ndarray
    noise_level: np.ndarray
    frequency_hz: float
    omnidirectional: np.ndarray
    directionality_index: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.noise_level.shape


def make_grid(bbox, resolution_deg):
    """Grille régulière à partir de bbox = (sud, ouest, nord, est)."""
    south, west, north, east = (float(v) for v in bbox)
    res = float(resolution_deg)
    if not (south < north and west < east and res > 0):
        raise ValueError(f"bbox/résolution invalides: {bbox}, {resolution_deg}")
    lats = np.arange(south, north + res * 0.5, res)
    lons = np.arange(west, east + res * 0.5, res)
    return lats, lons


def compute_directional_noise(lats, lons, ships, frequency_hz, params=None):
    """Calcule le champ de bruit directif pour une fréquence donnée.

    lats, lons : vecteurs 1D (degrés) ; ships : ShipCatalog ; frequency_hz : float.
    Retourne un DirectionalField avec noise_level de shape (len(lats), len(lons), n_azimuts).
    """
    params = params or ModelParams()
    step = float(params.azimuth_step_deg)
    if step <= 0 or not np.isclose(360.0 / step, round(360.0 / step)):
        raise ValueError(f"azimuth_step_deg doit diviser 360 (reçu: {step})")
    naz = int(round(360.0 / step))
    azimuth = (np.arange(naz) + 0.5) * step

    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    clat = lat2d.ravel()
    clon = lon2d.ravel()
    ncells = clat.size

    sl = source_level_db(frequency_hz, ships.kind, ships.speed_kn)
    bg_lin = 10.0 ** (background_level_db(frequency_hz) / 10.0)

    acc = np.zeros(ncells * naz, dtype=np.float64)
    slat = np.asarray(ships.lat, float)
    slon = np.asarray(ships.lon, float)
    for c0 in range(0, ncells, params.chunk_cells):
        m = min(params.chunk_cells, ncells - c0)
        la = clat[c0:c0 + m][:, None]
        lo = clon[c0:c0 + m][:, None]
        r_km = haversine_km(la, lo, slat[None, :], slon[None, :])
        brg = bearing_deg(la, lo, slat[None, :], slon[None, :])
        azbin = np.floor(brg / step).astype(np.int64) % naz
        tl = transmission_loss_db(r_km * 1000.0, frequency_hz, params.spreading, params.min_range_m)
        energy = np.where(r_km <= params.max_range_km, 10.0 ** ((sl[None, :] - tl) / 10.0), 0.0)
        idx = (np.arange(m)[:, None] * naz + azbin).ravel()
        acc[c0 * naz:(c0 + m) * naz] += np.bincount(idx, weights=energy.ravel(), minlength=m * naz)

    ship_lin = acc.reshape(lats.size, lons.size, naz)
    noise_level = 10.0 * np.log10(ship_lin + bg_lin)
    omnidirectional = 10.0 * np.log10(ship_lin.sum(axis=2) + bg_lin)
    directionality_index = noise_level.max(axis=2) - noise_level.min(axis=2)
    meta = {
        "azimuth_step_deg": step,
        "max_range_km": float(params.max_range_km),
        "spreading": float(params.spreading),
        "min_range_m": float(params.min_range_m),
        "n_ships": int(len(ships)),
    }
    return DirectionalField(
        lat=lats,
        lon=lons,
        azimuth=azimuth,
        noise_level=noise_level,
        frequency_hz=float(frequency_hz),
        omnidirectional=omnidirectional,
        directionality_index=directionality_index,
        meta=meta,
    )
