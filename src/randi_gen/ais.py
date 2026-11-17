"""Catalogue de navires AIS.

Trois sources :
- ``synthetic`` : climatologie AIS réaliste de la Méditerranée nord-ouest
  (lignes Toulon-Corse, approches de Marseille/Toulon, côtes corses), utilisée
  hors-ligne et pour les tests ;
- un fichier local CSV/GeoJSON de points AIS ;
- une URL http(s) vers un CSV/GeoJSON de points AIS (ex. export de statistiques
  AIS publiques type EMODnet Human Activities, MarineTraffic, marinecadastre...).

CSV attendu : colonnes lon/lat (alias longitude/lng, latitude) + optionnel
type/kind/ship_type et speed/sog.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .acoustics import normalize_kind
from .geo import interpolate_lane


@dataclass
class ShipCatalog:
    """Points navires AIS : lat/lon (degrés), type, vitesse (nœuds)."""

    lat: np.ndarray
    lon: np.ndarray
    kind: np.ndarray
    speed_kn: np.ndarray
    source: str

    def __len__(self):
        return int(self.lat.size)

    def filter_bbox(self, south, west, north, east, margin_deg=0.0):
        m = (
            (self.lat >= south - margin_deg)
            & (self.lat <= north + margin_deg)
            & (self.lon >= west - margin_deg)
            & (self.lon <= east + margin_deg)
        )
        return ShipCatalog(self.lat[m], self.lon[m], self.kind[m], self.speed_kn[m], self.source)

    @classmethod
    def concat(cls, catalogs):
        cats = [c for c in catalogs if len(c) > 0]
        if not cats:
            raise ValueError("concat de catalogues vides")
        return cls(
            lat=np.concatenate([c.lat for c in cats]),
            lon=np.concatenate([c.lon for c in cats]),
            kind=np.concatenate([c.kind for c in cats]).astype("<U16"),
            speed_kn=np.concatenate([c.speed_kn for c in cats]).astype(float),
            source="+".join(sorted({c.source for c in cats})),
        )


# (nom, lat1, lon1, lat2, lon2, navires/jour, type, vitesse nœuds)
LANES = [
    ("Toulon-Ajaccio", 43.10, 5.93, 41.93, 8.74, 7, "passenger", 18.0),
    ("Toulon-Bastia", 43.10, 5.93, 42.70, 9.45, 5, "passenger", 18.0),
    ("Toulon-IleRousse", 43.10, 5.93, 42.60, 8.81, 3, "passenger", 17.0),
    ("Toulon-Propriano", 43.10, 5.93, 41.69, 8.90, 2, "passenger", 16.0),
    ("Marseille-Ajaccio", 43.30, 5.37, 41.93, 8.74, 4, "cargo", 15.0),
    ("Marseille-Bastia", 43.30, 5.37, 42.70, 9.45, 3, "passenger", 18.0),
    ("Marseille-Nice-cotier", 43.30, 5.37, 43.70, 7.27, 6, "cargo", 12.0),
    ("Cote-est-corse", 42.70, 9.45, 41.38, 9.52, 8, "cargo", 13.0),
    ("Cote-ouest-corse", 42.57, 8.76, 41.93, 8.74, 4, "cargo", 12.0),
    ("Gênes-Corse", 44.41, 8.93, 42.70, 9.45, 3, "cargo", 14.0),
]

# (nom, lat, lon, nombre de navires à quai/approche)
PORT_HOTSPOTS = [
    ("Toulon", 43.10, 5.93, 60),
    ("Marseille", 43.30, 5.37, 50),
    ("Ajaccio", 41.93, 8.74, 25),
    ("Bastia", 42.70, 9.45, 25),
    ("Calvi", 42.57, 8.76, 8),
    ("Nice", 43.70, 7.26, 10),
]


def synthetic_catalog(bbox=None, seed=42, ships_scale=1.0):
    """Snapshot AIS synthétique de la zone Toulon-Corse (générateur aléatoire fixé)."""
    rng = np.random.default_rng(seed)
    lats, lons, kinds, speeds = [], [], [], []
    for _name, la1, lo1, la2, lo2, per_day, kind, spd in LANES:
        n = max(1, int(round(per_day * ships_scale)))
        la, lo = interpolate_lane(la1, lo1, la2, lo2, n)
        la = la + rng.normal(0.0, 0.02, n)
        lo = lo + rng.normal(0.0, 0.02, n)
        lats.append(la)
        lons.append(lo)
        kinds.append(np.full(n, kind, dtype="<U16"))
        speeds.append(np.clip(rng.normal(spd, 1.5, n), 5.0, 25.0))
    for _name, la, lo, count in PORT_HOTSPOTS:
        n = max(1, int(round(count * ships_scale)))
        lats.append(la + rng.normal(0.0, 0.05, n))
        lons.append(lo + rng.normal(0.0, 0.05, n))
        kinds.append(rng.choice(["cargo", "tanker", "passenger", "fishing", "other"], size=n,
                                 p=[0.35, 0.20, 0.20, 0.15, 0.10]).astype("<U16"))
        speeds.append(rng.uniform(2.0, 8.0, n))
    cat = ShipCatalog(
        lat=np.concatenate(lats),
        lon=np.concatenate(lons),
        kind=np.concatenate(kinds).astype("<U16"),
        speed_kn=np.concatenate(speeds).astype(float),
        source=f"synthetic-nw-mediterranean(seed={seed})",
    )
    if bbox is not None:
        cat = cat.filter_bbox(*bbox, margin_deg=0.5)
    return cat


def load_csv_points(text):
    """Parse un CSV de points AIS -> (lat, lon, kind, speed_kn)."""
    reader = csv.DictReader(io.StringIO(text))
    cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    clon, clat = pick("lon", "longitude", "lng"), pick("lat", "latitude")
    if clon is None or clat is None:
        raise ValueError("CSV AIS : colonnes lon/lat requises (lon,longitude,lng / lat,latitude)")
    ckind, cspeed = pick("kind", "ship_type", "type", "vessel_type"), pick("speed", "sog", "speed_kn")
    lat, lon, kind, speed = [], [], [], []
    for row in reader:
        try:
            lo, la = float(row[clon]), float(row[clat])
        except (TypeError, ValueError):
            continue
        lat.append(la)
        lon.append(lo)
        kind.append(normalize_kind(row.get(ckind, "other")) if ckind else "other")
        try:
            speed.append(float(row[cspeed]) if cspeed and row[cspeed] not in (None, "") else 12.0)
        except (TypeError, ValueError):
            speed.append(12.0)
    if not lat:
        raise ValueError("CSV AIS : aucune ligne valide")
    return (np.asarray(lat, float), np.asarray(lon, float),
            np.asarray(kind, dtype="<U16"), np.asarray(speed, float))


def load_geojson_points(text):
    """Parse un GeoJSON de points AIS -> (lat, lon, kind, speed_kn)."""
    gj = json.loads(text)
    lat, lon, kind, speed = [], [], [], []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lo, la = float(coords[0]), float(coords[1])
        props = feat.get("properties") or {}
        sp = props.get("speed", props.get("sog", 12.0))
        lat.append(la)
        lon.append(lo)
        kind.append(normalize_kind(props.get("type", props.get("kind", "other"))))
        speed.append(float(sp) if sp is not None else 12.0)
    if not lat:
        raise ValueError("GeoJSON AIS : aucun Point trouvé")
    return (np.asarray(lat, float), np.asarray(lon, float),
            np.asarray(kind, dtype="<U16"), np.asarray(speed, float))


def load_ais(source="synthetic", file=None, url=None, bbox=None, seed=42, ships_scale=1.0):
    """Charge un catalogue de navires.

    Priorité : ``file`` > ``url`` > ``source`` ('synthetic' ou chemin de fichier).
    ``source`` peut aussi être directement une URL http(s).
    """
    if file is None and url is None and source and str(source).lower().startswith(("http://", "https://")):
        url = source
    if file is not None:
        text = Path(file).read_text(encoding="utf-8")
        if str(file).lower().endswith((".geojson", ".json")):
            lat, lon, kind, speed = load_geojson_points(text)
        else:
            lat, lon, kind, speed = load_csv_points(text)
        src = f"file:{file}"
    elif url is not None:
        import requests

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        text = resp.text
        src = f"url:{url}"
        if url.lower().split("?")[0].endswith((".geojson", ".json")) or text.lstrip().startswith("{"):
            lat, lon, kind, speed = load_geojson_points(text)
        else:
            lat, lon, kind, speed = load_csv_points(text)
    elif str(source).lower() in ("synthetic", "syn", "demo"):
        return synthetic_catalog(bbox=bbox, seed=seed, ships_scale=ships_scale)
    elif source and os.path.exists(str(source)):
        return load_ais(file=str(source), bbox=bbox)
    else:
        raise ValueError(
            f"source AIS inconnue: {source!r} — utiliser 'synthetic', un fichier CSV/GeoJSON local ou une URL http(s)"
        )
    cat = ShipCatalog(lat=np.asarray(lat, float), lon=np.asarray(lon, float),
                      kind=np.asarray(kind, dtype="<U16"), speed_kn=np.asarray(speed, float), source=src)
    if bbox is not None:
        cat = cat.filter_bbox(*bbox, margin_deg=0.5)
    if len(cat) == 0:
        raise ValueError("catalogue AIS vide après filtrage sur la bbox")
    return cat
