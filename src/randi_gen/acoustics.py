"""Niveaux de source des navires, absorption et perte de transmission.

Niveaux exprimés en niveaux spectraux : dB re 1 µPa²/Hz (source à 1 m pour SL).
Formules usuelles : courbes de Wenz (1962) pour le bruit de trafic, Ross (1976)
pour les niveaux de source, Thorp (1967) pour l'absorption, Ainslie & McColm (1998).
"""

from __future__ import annotations

import numpy as np

F_REF_HZ = 100.0
SL_100HZ_DB = {
    "tanker": 185.0,
    "cargo": 182.0,
    "passenger": 178.0,
    "fishing": 168.0,
    "other": 165.0,
}
SL_SLOPE_DB_PER_DECADE = 17.0
SPEED_REF_KN = 12.0
SPEED_CORR_DB = 5.0
SPEED_CORR_CLIP_DB = 6.0
BG_100HZ_DB = 60.0
BG_SLOPE_DB_PER_DECADE = 12.0

_KIND_ALIASES = {
    "ferry": "passenger",
    "passenger ship": "passenger",
    "pleasure": "other",
    "pleasure craft": "other",
    "sailing": "other",
    "sailboat": "other",
    "military": "other",
    "tanker": "tanker",
    "cargo": "cargo",
    "passenger": "passenger",
    "fishing": "fishing",
}


def normalize_kind(kind):
    """Normalise un libellé de type de navire vers une des 5 classes du modèle."""
    key = str(kind).strip().lower()
    return _KIND_ALIASES.get(key, "other")


def source_level_db(freq_hz, kind="cargo", speed_kn=12.0):
    """Niveau de source d'un navire (dB re 1 µPa @ 1 m) en fonction de la fréquence.

    SL(f) = SL_100Hz(type) - 17*log10(f/100 Hz) + clip(5*log10(v/12 kn), -6, +6)
    ``kind`` peut être un scalaire ou un tableau de libellés.
    """
    f = np.asarray(freq_hz, dtype=float)
    kinds = np.asarray(kind, dtype=str)
    if kinds.ndim == 0:
        base = SL_100HZ_DB[normalize_kind(str(kinds))]
    else:
        base = np.array([SL_100HZ_DB[normalize_kind(str(k))] for k in kinds.ravel()]).reshape(kinds.shape)
    sl = base - SL_SLOPE_DB_PER_DECADE * np.log10(f / F_REF_HZ)
    v = np.clip(np.asarray(speed_kn, dtype=float), 1.0, None)
    corr = np.clip(SPEED_CORR_DB * np.log10(v / SPEED_REF_KN), -SPEED_CORR_CLIP_DB, SPEED_CORR_CLIP_DB)
    return sl + corr


def thorp_absorption_db_km(freq_hz):
    """Absorption de Thorp (dB/km), fréquence en Hz."""
    f = np.asarray(freq_hz, dtype=float) / 1000.0  # kHz
    return 0.11 * f**2 / (1.0 + f**2) + 44.0 * f**2 / (4100.0 + f**2) + 2.75e-4 * f**2 + 0.003


def transmission_loss_db(range_m, freq_hz, spreading=1.5, min_range_m=500.0):
    """Perte de transmission (dB) : étalement géométrique + absorption.

    TL(r, f) = 10*n*log10(r) + alpha(f) * r/1000, avec r plafonnée à min_range_m
    (n = exposant d'étalement : 1 cylindrique, 2 sphérique, 1.5 intermédiaire).
    """
    r = np.clip(np.asarray(range_m, dtype=float), float(min_range_m), None)
    return 10.0 * spreading * np.log10(r) + thorp_absorption_db_km(freq_hz) * (r / 1000.0)


def background_level_db(freq_hz):
    """Niveau de fond diffus (non directionnel), dB re 1 µPa²/Hz."""
    return BG_100HZ_DB - BG_SLOPE_DB_PER_DECADE * np.log10(np.asarray(freq_hz, dtype=float) / F_REF_HZ)
