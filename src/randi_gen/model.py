"""Modèle de bruit ambiant directif de type RANDI.

Pour chaque cellule (lat, lon) réceptrice et chaque secteur d'azimut :

    NL(lat, lon, az, f) = 10*log10( sum_navires w(az, cap) 10^((SL(f) - TL(r, f))/10) + P_fond(f) )

avec SL niveau de source du navire (Ross/Wenz), TL perte de transmission
(étalement + absorption de Thorp), r distance navire-récepteur, P_fond niveau
de fond diffus. Chaque navire contribue aux secteurs avec un poids w(az, cap) fonction du cap
du navire vu depuis le récepteur (0 = Nord, sens horaire), deux modes :

- ouverture **gaussienne** : w(az, cap) = exp(-4*ln2 * ((az - cap)/ouverture)^2),
  ouverture (FWHM) = `azimuth_aperture_deg`, égale par défaut à 360/`n_azimuth` ;
- **directivité tabulée** (`azimuth_pattern`, fichier .mat) : w(az, cap) = valeur du motif
  interpolée (linéairement, périodiquement sur 360°) au cap `cap` ; le nombre d'azimuts et
  l'échantillonnage angulaire sont extraits du fichier (voir pattern.py).

Les poids sont normalisés par navire (somme sur les secteurs = 1) : l'énergie totale
est conservée et le niveau omnidirectionnel est inchangé, seules les cartes directives
sont lissées (plus de tout-ou-rien entre secteurs adjacents).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .acoustics import background_level_db, source_level_db, transmission_loss_db
from .geo import bearing_deg, haversine_km
from .pattern import AzimuthPattern, load_pattern


@dataclass
class ModelParams:
    n_azimuth: int = 36
    azimuth_aperture_deg: float | None = None  # FWHM de l'ouverture gaussienne ; None -> 360/n_azimuth
    azimuth_pattern: str | AzimuthPattern | None = None  # fichier .mat de directivité tabulée par azimut
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
    pattern = None
    if params.azimuth_pattern is not None:
        pattern = (params.azimuth_pattern if isinstance(params.azimuth_pattern, AzimuthPattern)
                   else load_pattern(params.azimuth_pattern, n_azimuth_hint=params.n_azimuth))
        azimuth = pattern.azimuth
        naz = azimuth.size
        step = 360.0 / naz
        aperture = None
        sigma = None
    else:
        naz = int(params.n_azimuth)
        if naz < 1:
            raise ValueError(f"n_azimuth doit être >= 1 (reçu: {params.n_azimuth})")
        step = 360.0 / naz
        azimuth = (np.arange(naz) + 0.5) * step
        aperture = float(params.azimuth_aperture_deg) if params.azimuth_aperture_deg is not None else step
        if aperture <= 0:
            raise ValueError(f"azimuth_aperture_deg doit être > 0 (reçu: {aperture})")
        sigma = aperture / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM = aperture

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
    nships = slat.size
    if pattern is not None:
        offsets = range(naz)  # motif tabulé : tous les secteurs peuvent contribuer

        def gen_weights(brg, azbin):
            for k in offsets:
                yield k, np.maximum(pattern.weight(k, brg), 0.0)
    else:
        half = int(np.ceil(6.0 * sigma / step))  # queue gaussienne négligeable au-delà de 6 sigma
        offsets = range(-half, half + 1) if 2 * half + 1 < naz else range(naz)

        def gen_weights(brg, azbin):
            for dk in offsets:
                k = (azbin + dk) % naz
                d = (azimuth[k] - brg + 180.0) % 360.0 - 180.0
                yield k, np.exp(-0.5 * (d / sigma) ** 2)

    for c0 in range(0, ncells, params.chunk_cells):
        m = min(params.chunk_cells, ncells - c0)
        la = clat[c0:c0 + m][:, None]
        lo = clon[c0:c0 + m][:, None]
        r_km = haversine_km(la, lo, slat[None, :], slon[None, :])
        brg = bearing_deg(la, lo, slat[None, :], slon[None, :])
        tl = transmission_loss_db(r_km * 1000.0, frequency_hz, params.spreading, params.min_range_m)
        energy = np.where(r_km <= params.max_range_km, 10.0 ** ((sl[None, :] - tl) / 10.0), 0.0)
        azbin = np.floor(brg / step).astype(np.int64) % naz
        wsum = np.zeros((m, nships), dtype=np.float64)
        for _k, w in gen_weights(brg, azbin):
            wsum += w
        wsum = np.where(wsum > 0.0, wsum, 1.0)
        base = np.arange(m, dtype=np.int64)[:, None] * naz
        for k, w in gen_weights(brg, azbin):
            idx = np.broadcast_to(base + k, energy.shape).ravel()
            acc[c0 * naz:(c0 + m) * naz] += np.bincount(idx, weights=(energy * w / wsum).ravel(), minlength=m * naz)

    ship_lin = acc.reshape(lats.size, lons.size, naz)
    noise_level = 10.0 * np.log10(ship_lin + bg_lin)
    omnidirectional = 10.0 * np.log10(ship_lin.sum(axis=2) + bg_lin)
    directionality_index = noise_level.max(axis=2) - noise_level.min(axis=2)
    meta = {
        "n_azimuth": int(naz),
        "azimuth_step_deg": float(step),
        "azimuth_aperture_deg": float(aperture) if aperture is not None else float("nan"),
        "azimuth_sigma_deg": float(sigma) if sigma is not None else float("nan"),
        "azimuth_pattern": pattern.source if pattern is not None else "",
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
