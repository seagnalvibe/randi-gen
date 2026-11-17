"""Mode « vessel density » (raster de densité de navires, ex. EMODnet Human Activities).

Charge un raster de densité AIS (GeoTIFF / ESRI ASCII / NetCDF / zip, local ou URL,
ou directement via le WCS EMODnet), puis échantillonne des points navires dont la
densité spatiale est proportionnelle au raster :

    navires moyens par cellule = densité(h/km²/mois) * aire_cellule(km²) / heures_par_mois

Si le téléchargement EMODnet échoue (réseau indisponible, service en erreur), un
raster de test synthétique (format EMODnet, zone Méditerranée NO) est généré et saved
pour transparence.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .acoustics import normalize_kind
from .ais import LANES, PORT_HOTSPOTS, ShipCatalog
from .config import DEFAULT_BBOX

EMODNET_WCS_URL = "https://ows.emodnet-humanactivities.eu/wcs"
EMODNET_WMS_URL = "https://ows.emodnet-humanactivities.eu/wms"
EMODNET_DEFAULT_LAYER = "vesseldensity_allavg"

# codes de type de navire EMODnet (couches vesseldensity_XX)
EMODNET_TYPE_CODE = {
    "other": "00", "fishing": "01", "service": "02", "dredging": "03", "sailing": "04",
    "pleasure": "05", "highspeed": "06", "tug": "07", "passenger": "08", "cargo": "09",
    "tanker": "10", "military": "11", "unknown": "12",
}
EMODNET_TYPE_LAYER = {t: f"vesseldensity_{code}avg" for t, code in EMODNET_TYPE_CODE.items()}

DEFAULT_TYPE_MIX = {"cargo": 0.35, "tanker": 0.15, "passenger": 0.20, "fishing": 0.20, "other": 0.10}
TYPE_SPEED_KN = {"cargo": 14.0, "tanker": 13.0, "passenger": 18.0, "fishing": 9.0, "other": 10.0}

KM_PER_DEG_LAT = 111.32


@dataclass
class DensityRaster:
    """Raster de densité : values[nrows, ncols] (ligne 0 = nord), centres de cellules lats/lons."""

    values: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    area_km2: np.ndarray  # (nrows, ncols)
    dlat_deg: float
    dlon_deg: float
    crs: str = "EPSG:4326"
    source: str = ""

    @property
    def shape(self):
        return self.values.shape

    def clip_bbox(self, south, west, north, east):
        ri = np.nonzero((self.lats >= south) & (self.lats <= north))[0]
        cj = np.nonzero((self.lons >= west) & (self.lons <= east))[0]
        if ri.size == 0 or cj.size == 0:
            raise ValueError("bbox hors du raster de densité")
        return DensityRaster(
            values=self.values[np.ix_(ri, cj)],
            lats=self.lats[ri], lons=self.lons[cj],
            area_km2=self.area_km2[np.ix_(ri, cj)],
            dlat_deg=self.dlat_deg, dlon_deg=self.dlon_deg,
            crs=self.crs, source=self.source,
        )


def _make_area(lats, ncols, dlat_deg, dlon_deg, cell_area_km2_scalar=None):
    if cell_area_km2_scalar is not None:
        return np.full((lats.size, ncols), float(cell_area_km2_scalar))
    area_row = (dlat_deg * KM_PER_DEG_LAT) * (dlon_deg * KM_PER_DEG_LAT * np.cos(np.radians(lats)))
    return np.repeat(area_row[:, None], ncols, axis=1)


def from_geotiff(source, source_name=""):
    """Charge un GeoTIFF (chemin ou bytes) -> DensityRaster (centres de cellules en EPSG:4326)."""
    import rasterio
    from rasterio.io import MemoryFile

    if isinstance(source, (bytes, bytearray)):
        mem = MemoryFile()
        mem.write(bytes(source))
        src = mem.open()
        close = mem.close
    else:
        src = rasterio.open(str(source))
        close = src.close
    try:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        nrows, ncols = src.shape
        tr = src.transform
        crs = src.crs
        pw, ph = tr.a, tr.e
        xs = tr.c + (np.arange(ncols) + 0.5) * pw
        ys = tr.f + (np.arange(nrows) + 0.5) * ph
        crs_str = str(crs) if crs else "EPSG:4326"
        if crs is not None and crs.to_epsg() not in (None, 4326):
            from rasterio.warp import transform as warp_transform

            lons, _ = warp_transform(crs, "EPSG:4326", xs.tolist(), [ys[0]] * ncols)
            _, lats = warp_transform(crs, "EPSG:4326", [xs[0]] * nrows, ys.tolist())
            lons = np.asarray(lons, float)
            lats = np.asarray(lats, float)
            if lons[0] > lons[-1]:
                lons = lons[::-1]
                arr = arr[:, ::-1]
            if lats[0] < lats[-1]:
                lats = lats[::-1]
                arr = arr[::-1, :]
            dlon = float(np.mean(np.diff(lons)))
            dlat = float(abs(np.mean(np.diff(lats))))
            area = _make_area(lats, ncols, dlat, dlon)
        else:
            lons = xs
            lats = ys
            if lats[0] < lats[-1]:  # ligne 0 doit être le nord
                lats = lats[::-1]
                arr = arr[::-1, :]
            dlon = float(abs(pw))
            dlat = float(abs(ph))
            if abs(pw) > 1.0:  # CRS projeté en mètres
                area = _make_area(lats, ncols, dlat, dlon, cell_area_km2_scalar=abs(pw * ph) / 1e6)
            else:
                area = _make_area(lats, ncols, dlat, dlon)
        return DensityRaster(values=arr, lats=lats, lons=lons, area_km2=area,
                             dlat_deg=dlat, dlon_deg=dlon, crs=crs_str,
                             source=source_name or ("geotiff" if not isinstance(source, (bytes, bytearray)) else "geotiff-bytes"))
    finally:
        close()


def from_ascii_grid(path):
    """Charge une grille ESRI ASCII (.asc/.grd) -> DensityRaster (supposé EPSG:4326)."""
    path = Path(path)
    header = {}
    data_lines = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks = line.split()
            if not toks:
                continue
            if len(toks) == 2 and toks[0][0].isalpha() and toks[0].lower() not in header:
                header[toks[0].lower()] = toks[1]
            else:
                data_lines.append(line)
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cellsize = float(header["cellsize"])
    if "xllcorner" in header:
        xll = float(header["xllcorner"])
    else:
        xll = float(header["xllcenter"]) - 0.5 * cellsize
    if "yllcorner" in header:
        yll = float(header["yllcorner"])
    else:
        yll = float(header["yllcenter"]) - 0.5 * cellsize
    nodata = float(header.get("nodata_value", "-9999"))
    values = np.loadtxt(io.StringIO("".join(data_lines)), dtype=np.float64)
    values = values.reshape(nrows, ncols)
    values = np.where(values == nodata, np.nan, values)
    lons = xll + (np.arange(ncols) + 0.5) * cellsize
    lats = yll + (nrows - 0.5 - np.arange(nrows)) * cellsize  # ligne 0 = nord
    area = _make_area(lats, ncols, cellsize, cellsize)
    return DensityRaster(values=values, lats=lats, lons=lons, area_km2=area,
                         dlat_deg=cellsize, dlon_deg=cellsize, crs="EPSG:4326",
                         source=f"ascii:{path.name}")


def from_netcdf(path):
    """Charge un raster NetCDF 2D (ex. sortie WCS EMODnet format netCDF)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        ds = ds.load()
    lat_name = next(n for n in ds.coords if n.lower() in ("lat", "latitude", "y"))
    lon_name = next(n for n in ds.coords if n.lower() in ("lon", "longitude", "x"))
    var = next(v for v in ds.data_vars if ds[v].dims == (lat_name, lon_name) or ds[v].dims == (lon_name, lat_name))
    da = ds[var]
    if da.dims == (lon_name, lat_name):
        da = da.transpose(lat_name, lon_name)
    lats = np.asarray(da[lat_name].values, float)
    lons = np.asarray(da[lon_name].values, float)
    values = np.asarray(da.values, float)
    nodata = da.attrs.get("_FillValue", da.attrs.get("missing_value"))
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    if lats[0] < lats[-1]:
        lats = lats[::-1]
        values = values[::-1, :]
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        values = values[:, ::-1]
    dlat = float(abs(np.mean(np.diff(lats))))
    dlon = float(abs(np.mean(np.diff(lons))))
    area = _make_area(lats, lons.size, dlat, dlon)
    return DensityRaster(values=values, lats=lats, lons=lons, area_km2=area,
                         dlat_deg=dlat, dlon_deg=dlon, crs="EPSG:4326",
                         source=f"netcdf:{Path(path).name}:{var}")


def write_ascii_grid(raster: DensityRaster, path):
    """Écrit un raster en ESRI ASCII (représentation de secours, sans dépendance)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nrows, ncols = raster.shape
    xll = float(raster.lons[0] - 0.5 * raster.dlon_deg)
    yll = float(raster.lats[-1] - 0.5 * raster.dlat_deg)
    cellsize = float(raster.dlon_deg)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ncols {ncols}\nnrows {nrows}\nxllcorner {xll:.10f}\nyllcorner {yll:.10f}\n")
        f.write(f"cellsize {cellsize:.10f}\nNODATA_value -9999\n")
        vals = np.where(np.isnan(raster.values), -9999.0, raster.values)
        for row in vals:
            f.write(" ".join(f"{v:.6g}" for v in row) + "\n")
    return path


def write_geotiff(raster: DensityRaster, path):
    """Écrit un GeoTIFF EPSG:4326 (nécessite rasterio)."""
    import rasterio
    from rasterio.transform import Affine

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nrows, ncols = raster.shape
    transform = Affine(raster.dlon_deg, 0.0, float(raster.lons[0] - 0.5 * raster.dlon_deg),
                       0.0, -raster.dlat_deg, float(raster.lats[0] + 0.5 * raster.dlat_deg))
    with rasterio.open(path, "w", driver="GTiff", height=nrows, width=ncols, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan) as dst:
        dst.write(raster.values.astype(np.float32), 1)
    return path


def _http_get(url, timeout=60):
    import requests

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "randi-gen/0.1"})
    resp.raise_for_status()
    return resp.content


def _extract_exception(xml_bytes):
    m = re.search(rb"ExceptionText[^>]*>([^<]+)<", xml_bytes)
    return m.group(1).decode("utf-8", "ignore") if m else xml_bytes[:200].decode("utf-8", "ignore")


def _resolve_coverage_id(coverage_ids, layer):
    """Résout un nom de couche vers le CoverageId exact côté serveur.

    Accepte le nom court ('vesseldensity_09avg') ou complet ('emodnet__vesseldensity_09avg').
    Ne replie JAMAIS sur une autre couche : chaque type de navire doit avoir sa propre
    couche (sinon --ais-density-type chargerait silencieusement la couche all-types).
    """
    if layer in coverage_ids:
        return layer
    prefixed = [c for c in coverage_ids if c.endswith("__" + layer)]
    if len(prefixed) == 1:
        return prefixed[0]
    containing = [c for c in coverage_ids if layer in c]
    if len(containing) == 1:
        return containing[0]
    avail = [c for c in coverage_ids if "vesseldensity" in c.lower()][:8]
    raise RuntimeError(
        f"couche {layer!r} introuvable côté serveur ({len(coverage_ids)} couvertures ; "
        f"vesseldensity disponibles ex. {avail})"
    )


def combine_rasters(rasters):
    """Somme plusieurs rasters de densité (couches par type) -> raster combiné.

    Retourne None si les grilles ne coïncident pas. Les cellules NaN partout
    (terre / nodata) restent NaN.
    """
    if not rasters:
        return None
    r0 = rasters[0]
    for r in rasters[1:]:
        if r.shape != r0.shape or not (np.allclose(r.lats, r0.lats) and np.allclose(r.lons, r0.lons)):
            return None
    total = np.zeros(r0.shape)
    all_nan = np.ones(r0.shape, dtype=bool)
    for r in rasters:
        fin = np.isfinite(r.values)
        total += np.where(fin, r.values, 0.0)
        all_nan &= ~fin
    values = np.where(all_nan, np.nan, total)
    return DensityRaster(values=values, lats=r0.lats, lons=r0.lons, area_km2=r0.area_km2,
                         dlat_deg=r0.dlat_deg, dlon_deg=r0.dlon_deg, crs=r0.crs,
                         source="sum-per-type:" + "+".join(r.source for r in rasters))


def _native_epsg_from_describe(describe_xml):
    m = re.search(r'srsName="[^"]*/EPSG/\d+/(\d+)"', describe_xml)
    return f"EPSG:{m.group(1)}" if m else "EPSG:3857"


def fetch_emodnet_vessel_density(bbox, layer=EMODNET_DEFAULT_LAYER, wcs_url=EMODNET_WCS_URL, timeout=120):
    """Télécharge la couche vessel density EMODnet (WCS 2.0.1 GetCoverage, GeoTIFF) sur bbox.

    bbox = (sud, ouest, nord, est) en degrés. Lève RuntimeError si le service échoue.
    """
    import urllib.parse

    from rasterio.warp import transform as warp_transform

    south, west, north, east = (float(v) for v in bbox)
    cap = _http_get(f"{wcs_url}?service=WCS&version=2.0.1&request=GetCapabilities", timeout=timeout).decode("utf-8", "ignore")
    coverage_ids = re.findall(r"<wcs:CoverageId>([^<]+)</wcs:CoverageId>", cap)
    layer = _resolve_coverage_id(coverage_ids, layer)
    desc = _http_get(f"{wcs_url}?service=WCS&version=2.0.1&request=DescribeCoverage&coverageId={layer}", timeout=timeout).decode("utf-8", "ignore")
    native_crs = _native_epsg_from_describe(desc)
    times = re.findall(r"<gml:timePosition[^>]*>([^<]+)</gml:timePosition>", desc)
    xs, ys = warp_transform("EPSG:4326", native_crs, [west, east], [south, north])
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    params = {
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
        "coverageId": layer, "format": "image/tiff",
        "subset": [f"X({xmin},{xmax})", f"Y({ymin},{ymax})"],
    }
    if times:
        params["subset"].append(f'time("{times[-1]}")')
    url = f"{wcs_url}?" + urllib.parse.urlencode(params, doseq=True)
    try:
        data = _http_get(url, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"GetCoverage a échoué: {type(e).__name__}: {str(e)[:120]}") from e
    if data[:4] not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"réponse WCS non GeoTIFF: {_extract_exception(data)}")
    raster = from_geotiff(data, source_name=f"emodnet-wcs:{layer}")
    raster.source = url
    return raster


def synthetic_density_raster(bbox=DEFAULT_BBOX, cellsize_deg=0.01, seed=42):
    """Raster de test au format EMODnet (densité en h/km²/mois, EPSG:4326, ~1 km),
    construit à partir des lignes Toulon-Corse et ports de la Méditerranée NO."""
    from .geo import haversine_km, interpolate_lane

    south, west, north, east = (float(v) for v in bbox)
    lons = np.arange(west, east + cellsize_deg * 0.5, cellsize_deg)
    lats = np.arange(north, south - cellsize_deg * 0.5, -cellsize_deg)
    nrows, ncols = lats.size, lons.size
    values = np.full((nrows, ncols), 0.2)  # fond offshore
    rng = np.random.default_rng(seed)

    def stamp(lat, lon, amplitude, sigma_deg):
        i0 = int(np.argmin(np.abs(lats - lat)))
        j0 = int(np.argmin(np.abs(lons - lon)))
        w = max(2, int(round(3 * sigma_deg / cellsize_deg)))
        i1, i2 = max(0, i0 - w), min(nrows, i0 + w + 1)
        j1, j2 = max(0, j0 - w), min(ncols, j0 + w + 1)
        di = (np.arange(i1, i2) - i0) * cellsize_deg
        dj = (np.arange(j1, j2) - j0) * cellsize_deg
        kernel = amplitude * np.exp(-(di[:, None] ** 2 + dj[None, :] ** 2) / (2 * sigma_deg ** 2))
        values[i1:i2, j1:j2] += kernel

    for _name, la1, lo1, la2, lo2, per_day, _kind, _spd in LANES:
        dist = float(haversine_km(la1, lo1, la2, lo2))
        n_pts = max(2, int(dist / 0.5))
        la, lo = interpolate_lane(la1, lo1, la2, lo2, n_pts)
        amp = 40.0 + 30.0 * per_day  # lignes plus ou moins fréquentées
        for k in range(n_pts):
            stamp(float(la[k]), float(lo[k]), amp, 0.015 + rng.uniform(0, 0.003))
    for _name, la, lo, count in PORT_HOTSPOTS:
        for _ in range(max(1, count // 6)):
            stamp(la + rng.normal(0, 0.01), lo + rng.normal(0, 0.01), 500.0, 0.025)
    area = _make_area(lats, ncols, cellsize_deg, cellsize_deg)
    return DensityRaster(values=values, lats=lats, lons=lons, area_km2=area,
                         dlat_deg=cellsize_deg, dlon_deg=cellsize_deg, crs="EPSG:4326",
                         source="synthetic-emodnet-like-nw-med")


def _load_any(path):
    ext = Path(str(path)).suffix.lower()
    if ext in (".tif", ".tiff"):
        return from_geotiff(path)
    if ext in (".asc", ".grd"):
        return from_ascii_grid(path)
    if ext in (".nc", ".cdf"):
        return from_netcdf(path)
    for loader in (from_ascii_grid, from_geotiff, from_netcdf):
        try:
            return loader(path)
        except Exception:
            continue
    raise ValueError(f"format de raster non reconnu: {path}")


def load_density_raster(spec, bbox=None, outdir=None, seed=42, layer=EMODNET_DEFAULT_LAYER,
                        wcs_url=EMODNET_WCS_URL, timeout=120, log=print):
    """Charge un raster de densité depuis 'emodnet' | URL | chemin local.

    Retourne (DensityRaster, note). Si spec == 'emodnet' et que le WCS échoue,
    génère un raster de test synthétique (sauvegardé dans outdir) comme repli.
    """
    spec = str(spec)
    if spec.lower() == "emodnet":
        try:
            raster = fetch_emodnet_vessel_density(bbox or DEFAULT_BBOX, layer=layer, wcs_url=wcs_url, timeout=timeout)
            return raster, "emodnet-wcs"
        except Exception as e:
            log(f"[randi-gen] WARN: téléchargement EMODnet WCS impossible ({type(e).__name__}: {str(e)[:140]})")
            log("[randi-gen]       -> génération d'un raster de TEST synthétique (format EMODnet) en repli")
            raster = synthetic_density_raster(bbox or DEFAULT_BBOX, seed=seed)
            raster.source = "synthetic-test-raster(fallback; EMODnet WCS indisponible)"
            outdir = Path(outdir or ".")
            outdir.mkdir(parents=True, exist_ok=True)
            p = outdir / "emodnet_vesseldensity_TEST.asc"
            write_ascii_grid(raster, p)
            log(f"[randi-gen] raster de test écrit: {p}")
            return raster, "synthetic-fallback"
    if spec.lower().startswith(("http://", "https://")):
        data = _http_get(spec, timeout=timeout)
        if data[:4] == b"PK\x03\x04" or spec.lower().split("?")[0].endswith(".zip"):
            zf = zipfile.ZipFile(io.BytesIO(data))
            name = next(n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff", ".asc", ".grd", ".nc")))
            return _load_any_from_bytes(zf.read(name), name)
        ext = Path(spec.split("?")[0]).suffix.lower()
        if ext in (".tif", ".tiff"):
            return from_geotiff(data, source_name=spec), "url-geotiff"
        if ext in (".asc", ".grd"):
            return from_ascii_grid_from_bytes(data, spec), "url-ascii"
        if ext in (".nc", ".cdf"):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(data)
                return from_netcdf(tmp.name), "url-netcdf"
        raise ValueError(f"URL non reconnue comme raster: {spec}")
    return _load_any(spec), "file"


def from_ascii_grid_from_bytes(data, name=""):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".asc", delete=False) as tmp:
        tmp.write(data)
        return from_ascii_grid(tmp.name)


def _load_any_from_bytes(data, name):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as tmp:
        tmp.write(data)
        return _load_any(tmp.name)


def sample_catalog_from_density(raster: DensityRaster, bbox=None, seed=42, hours_per_month=720.0,
                                scale=1.0, max_ships=20000, type_mix=None, fixed_type=None):
    """Échantillonne des points navires ~ densité du raster -> ShipCatalog.

    densité EMODnet en h/km²/mois : navires moyens/cellule = densité * aire(km²) / heures_par_mois.
    """
    r = raster.clip_bbox(*bbox) if bbox is not None else raster
    lam = np.nan_to_num(r.values, nan=0.0) * r.area_km2 * float(scale) / float(hours_per_month)
    lam = np.clip(lam, 0.0, None)
    total = float(lam.sum())
    if total > max_ships and total > 0:
        lam = lam * (max_ships / total)
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam)
    ii, jj = np.nonzero(n)
    counts = n[ii, jj]
    if counts.sum() == 0:
        raise ValueError("raster de densité nul sur la bbox (aucun navire échantillonné)")
    lat = np.repeat(r.lats[ii], counts) + (rng.random(counts.sum()) - 0.5) * r.dlat_deg
    lon = np.repeat(r.lons[jj], counts) + (rng.random(counts.sum()) - 0.5) * r.dlon_deg
    if fixed_type is not None:
        kind = np.full(counts.sum(), normalize_kind(fixed_type), dtype="<U16")
    else:
        mix = type_mix or DEFAULT_TYPE_MIX
        keys = list(mix.keys())
        kind = rng.choice(keys, size=counts.sum(), p=[mix[k] for k in keys]).astype("<U16")
    speed = np.array([TYPE_SPEED_KN.get(str(k), 10.0) for k in kind]) + rng.normal(0.0, 1.5, counts.sum())
    speed = np.clip(speed, 2.0, 25.0)
    if lat.size > max_ships:  # vrai cap après tirage de Poisson
        sel = rng.choice(lat.size, size=int(max_ships), replace=False)
        lat, lon, kind, speed = lat[sel], lon[sel], kind[sel], speed[sel]
    return ShipCatalog(
        lat=lat, lon=lon, kind=kind, speed_kn=speed,
        source=f"vessel-density[{r.source}] n={counts.sum()} (h/mois={hours_per_month}, scale={scale}, max={max_ships})",
    )
