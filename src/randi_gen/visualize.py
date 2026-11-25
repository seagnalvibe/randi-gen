"""Visualisations : figures PNG (matplotlib) et viewer HTML interactif (plotly)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

CITIES = {
    "Toulon": (43.1244, 5.9299),
    "Marseille": (43.2965, 5.3698),
    "Nice": (43.7102, 7.2620),
    "Ajaccio": (41.9267, 8.7386),
    "Bastia": (42.7028, 9.4501),
    "Calvi": (42.5675, 8.7566),
}


def _robust_limits(z, lo=1.0, hi=99.5):
    """Limites de couleur robustes (percentiles) : les points chauds isolés
    (ports) n'écrasent pas le reste de la carte (rails de navigation)."""
    zf = np.asarray(z, float)
    zf = zf[np.isfinite(zf)]
    if zf.size == 0:
        return None, None
    vmin, vmax = float(np.percentile(zf, lo)), float(np.percentile(zf, hi))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return vmin, vmax


def nearest_azimuth_index(field, azimuth_deg):
    d = np.abs(((field.azimuth - float(azimuth_deg) + 180.0) % 360.0) - 180.0)
    return int(np.argmin(d))


def nearest_cell(field, lat, lon):
    i = int(np.argmin(np.abs(field.lat - float(lat))))
    j = int(np.argmin(np.abs(field.lon - float(lon))))
    return i, j


def _city_overlay(ax):
    for name, (la, lo) in CITIES.items():
        ax.plot(lo, la, marker="^", color="red", markersize=6, linestyle="None")
        ax.annotate(name, (lo, la), textcoords="offset points", xytext=(4, 4), fontsize=7, color="red")


def plot_map_azimuth(field, azimuth_deg, ax=None, cmap="viridis"):
    """Carte 2D du niveau directif pour un azimut donné."""
    ai = nearest_azimuth_index(field, azimuth_deg)
    fig, ax = (plt.subplots(figsize=(8, 6)) if ax is None else (ax.figure, ax))
    z = field.noise_level[:, :, ai]
    vmin, vmax = _robust_limits(z)
    pcm = ax.pcolormesh(field.lon, field.lat, z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="dB re 1 µPa²/Hz (clip p1–p99,5)")
    _city_overlay(ax)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Niveau de bruit directif — f={field.frequency_hz:g} Hz, azimut {field.azimuth[ai]:g}°")
    return fig, ax


def plot_omni(field, ax=None, cmap="viridis"):
    """Carte du niveau omnidirectionnel."""
    fig, ax = (plt.subplots(figsize=(8, 6)) if ax is None else (ax.figure, ax))
    vmin, vmax = _robust_limits(field.omnidirectional)
    pcm = ax.pcolormesh(field.lon, field.lat, field.omnidirectional, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="dB re 1 µPa²/Hz (clip p1–p99,5)")
    _city_overlay(ax)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Niveau omnidirectionnel — f={field.frequency_hz:g} Hz")
    return fig, ax


def plot_di(field, ax=None, cmap="magma"):
    """Carte de l'indice de directivité (dB)."""
    fig, ax = (plt.subplots(figsize=(8, 6)) if ax is None else (ax.figure, ax))
    vmin, vmax = _robust_limits(field.directionality_index)
    pcm = ax.pcolormesh(field.lon, field.lat, field.directionality_index, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="dB (clip p1–p99,5)")
    _city_overlay(ax)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Indice de directivité — f={field.frequency_hz:g} Hz")
    return fig, ax


def plot_polar(field, lat, lon):
    """Diagramme polaire du niveau en fonction de l'azimut en un point."""
    i, j = nearest_cell(field, lat, lon)
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    width = np.radians(field.azimuth[1] - field.azimuth[0]) if field.azimuth.size > 1 else np.radians(10.0)
    ax.bar(np.radians(field.azimuth), field.noise_level[i, j, :], width=width * 0.9, alpha=0.85)
    ax.set_title(f"Directivité à lat={field.lat[i]:.3f}°, lon={field.lon[j]:.3f}° — f={field.frequency_hz:g} Hz")
    return fig


def plot_surface3d(field, azimuth_deg):
    """Surface 3D (lon, lat, niveau) pour un azimut donné."""
    ai = nearest_azimuth_index(field, azimuth_deg)
    lat2d, lon2d = np.meshgrid(field.lat, field.lon, indexing="ij")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(lon2d, lat2d, field.noise_level[:, :, ai], cmap="viridis")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="dB re 1 µPa²/Hz")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_zlabel("Niveau (dB)")
    ax.set_title(f"Carte 3D — f={field.frequency_hz:g} Hz, azimut {field.azimuth[ai]:g}°")
    return fig


def plot_density_raster(raster, out_path=None, ax=None, cmap="YlOrRd", log_scale=True):
    """Affiche le raster brut de vessel density (h/km²/mois, échelle log par défaut)."""
    from matplotlib.colors import LogNorm

    fig, ax = (plt.subplots(figsize=(8.5, 6.5)) if ax is None else (ax.figure, ax))
    v = np.asarray(raster.values, dtype=float)
    if log_scale:
        vm = np.ma.masked_where(~np.isfinite(v) | (v <= 0), v)
        vmax = float(np.nanmax(v)) if np.isfinite(v).any() else 1.0
        pcm = ax.pcolormesh(raster.lons, raster.lats, vm, shading="auto", cmap=cmap,
                            norm=LogNorm(vmin=max(1e-3, float(np.nanmin(vm)) if vm.count() else 1e-3), vmax=vmax))
        cbl = "densité (h/km²/mois, log)"
    else:
        pcm = ax.pcolormesh(raster.lons, raster.lats, np.nan_to_num(v), shading="auto", cmap=cmap)
        cbl = "densité (h/km²/mois)"
    fig.colorbar(pcm, ax=ax, label=cbl)
    _city_overlay(ax)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    src = (raster.source or "")[:110]
    ax.set_title(f"Vessel density (données brutes) — {src}")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig, ax


def save_demo_figures(field, outdir):
    """Génère un jeu de figures de démonstration dans outdir, retourne les chemins."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    az_max = float(field.azimuth[int(np.argmax(field.noise_level.max(axis=(0, 1))))])
    figures = [
        ("omni", plot_omni(field)[0]),
        ("di", plot_di(field)[0]),
        ("surface3d_azmax", plot_surface3d(field, az_max)),
    ]
    for az in (0.0, 90.0, 180.0, 270.0):
        figures.append((f"az{az:03.0f}", plot_map_azimuth(field, az)[0]))
    hy, hx = np.unravel_index(int(np.argmax(field.omnidirectional)), field.omnidirectional.shape)
    figures.append(("polar_hotspot", plot_polar(field, field.lat[hy], field.lon[hx])))
    for name, fig in figures:
        p = outdir / f"{name}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def build_html_viewer(fields, out_path, max_traces=240, density_raster=None):
    """Viewer HTML autonome (plotly embarqué) : carte + polaire + nuage 3D,
    avec curseur d'azimut et sélecteur de fréquence si plusieurs champs.
    Si density_raster est fourni, sa carte brute est affichée en haut (toujours visible)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    base = fields[0]
    naz = base.azimuth.size
    for f in fields:
        if f.azimuth.size != naz or not np.allclose(f.azimuth, base.azimuth):
            raise ValueError("tous les champs doivent partager la même grille d'azimut")
        if f.lat.size != base.lat.size or f.lon.size != base.lon.size:
            raise ValueError("tous les champs doivent partager la même grille lat/lon")
    nfreq = len(fields)
    step = max(1, int(np.ceil(nfreq * naz / max_traces)))
    az_idx = list(range(0, naz, step))
    if az_idx[-1] != naz - 1:
        az_idx.append(naz - 1)
    n_az = len(az_idx)

    lat2d, lon2d = np.meshgrid(base.lat, base.lon, indexing="ij")
    hy, hx = np.unravel_index(int(np.argmax(base.omnidirectional)), base.omnidirectional.shape)
    sub = max(1, int(np.ceil(max(lat2d.shape) / 40.0)))

    has_density = density_raster is not None
    specs = ([ [{"type": "xy"}] ] if has_density else []) + [
        [{"type": "xy"}], [{"type": "polar"}], [{"type": "scene"}]
    ]
    heights = ([0.28] if has_density else []) + [0.32, 0.20, 0.20]
    titles = (["Vessel density (données brutes, h/km²/mois, log)"] if has_density else []) + [
        "Carte du niveau directif (lat, lon)",
        f"Diagramme polaire au point chaud (lat={base.lat[hy]:.3f}°, lon={base.lon[hx]:.3f}°)",
        "Nuage 3D sous-échantillonné (couleur = niveau)",
    ]
    row_map = 2 if has_density else 1
    fig = make_subplots(rows=len(specs), cols=1, row_heights=heights, specs=specs, subplot_titles=titles)
    first = True
    for fi, f in enumerate(fields):
        for ai in az_idx:
            vis = first
            first = False
            z = f.noise_level[:, :, ai]
            fig.add_trace(
                go.Heatmap(x=f.lon, y=f.lat, z=z, colorscale="Viridis",
                           colorbar=dict(title="dB"), visible=vis,
                           name=f"f={f.frequency_hz:g} Hz, az={f.azimuth[ai]:g}°"),
                row=row_map, col=1,
            )
            fig.add_trace(
                go.Barpolar(r=f.noise_level[hy, hx, :], theta=f.azimuth, visible=vis,
                            marker_color=z[hy, hx]),
                row=row_map + 1, col=1,
            )
            zz = z[::sub, ::sub]
            fig.add_trace(
                go.Scatter3d(
                    x=lon2d[::sub, ::sub].ravel(), y=lat2d[::sub, ::sub].ravel(), z=zz.ravel(),
                    mode="markers",
                    marker=dict(size=2, color=zz.ravel(), colorscale="Viridis", showscale=False),
                    visible=vis,
                ),
                row=row_map + 2, col=1,
            )
    if has_density:
        dv = np.asarray(density_raster.values, dtype=float)
        dv = np.where(np.isfinite(dv) & (dv > 0), dv, np.nan)
        fig.add_trace(
            go.Heatmap(x=density_raster.lons, y=density_raster.lats, z=np.log10(dv),
                       colorscale="YlOrRd", colorbar=dict(title="log10 h/km²/mois"),
                       visible=True, name="vessel density (brut)", zmin=None, zmax=None),
            row=1, col=1,
        )
    fig.update_polars(angularaxis=dict(rotation=90, direction="clockwise"))

    ntraces = len(fig.data)

    steps = []
    for fi in range(nfreq):
        for k, ai in enumerate(az_idx):
            vis = [False] * ntraces
            for j in range(3):
                vis[(fi * n_az + k) * 3 + j] = True
            if has_density:
                vis[-1] = True  # la carte de densité brute reste visible
            label = f"az={base.azimuth[ai]:g}°" if nfreq == 1 else f"f={fields[fi].frequency_hz:g} Hz, az={base.azimuth[ai]:g}°"
            steps.append(dict(label=label, method="update", args=[{"visible": vis}]))

    fig.update_layout(
        sliders=[dict(active=0, steps=steps, pad=dict(t=40))],
        height=1500,
        title="randi-gen — carte 3D (lat, lon, azimut) de bruit ambiant directif",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    return out_path
