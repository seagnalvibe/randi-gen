"""Fonctions géodésiques sur sphère terrestre (rayon moyen 6371,0088 km)."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance orthodromique en km (broadcast numpy)."""
    lat1, lon1, lat2, lon2 = (np.asarray(a, dtype=float) for a in (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Cap initial (0 = Nord, sens horaire) du point 1 vers le point 2, en degrés [0, 360)."""
    p1, p2 = np.radians(np.asarray(lat1, dtype=float)), np.radians(np.asarray(lat2, dtype=float))
    dlmb = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    y = np.sin(dlmb) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def destination_point(lat, lon, bearing, dist_km):
    """Point d'arrivée depuis (lat, lon) le long d'un cap et d'une distance (grande sphère)."""
    d = np.asarray(dist_km, dtype=float) / EARTH_RADIUS_KM
    brg = np.radians(np.asarray(bearing, dtype=float))
    p1 = np.radians(np.asarray(lat, dtype=float))
    l1 = np.radians(np.asarray(lon, dtype=float))
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(brg))
    l2 = l1 + np.arctan2(
        np.sin(brg) * np.sin(d) * np.cos(p1),
        np.cos(d) - np.sin(p1) * np.sin(p2),
    )
    return np.degrees(p2), (np.degrees(l2) + 540.0) % 360.0 - 180.0


def interpolate_lane(lat1, lon1, lat2, lon2, n):
    """Échantillonne n points régulièrement espacés le long de la route (lat1,lon1)->(lat2,lon2)."""
    total = haversine_km(lat1, lon1, lat2, lon2)
    brg = bearing_deg(lat1, lon1, lat2, lon2)
    frac = (np.arange(n) + 0.5) / n
    return destination_point(lat1, lon1, brg, total * frac)
