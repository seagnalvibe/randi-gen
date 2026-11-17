"""Interface de visualisation Streamlit (lancée en local sur 127.0.0.1).

Usage : randi-gen serve output/randi_map_f1000Hz.nc [autres.nc ...]
ou    streamlit run src/randi_gen/app.py -- fichier.nc
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from randi_gen.netcdf_io import read_field

st.set_page_config(page_title="randi-gen — bruit ambiant directif", layout="wide")


def script_args():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return [a for a in args if not a.startswith("-")]


def script_files():
    return [a for a in script_args() if a.lower().endswith(".nc")]


def script_density_files():
    return [a for a in script_args() if not a.lower().endswith(".nc") and Path(a).exists()]


st.title("randi-gen — carte 3D de bruit ambiant sous-marin directif (RANDI)")
st.caption("Carte 3D NetCDF (lat, lon, azimut) — niveaux spectraux en dB re 1 µPa²/Hz, azimut 0 = Nord, sens horaire vu depuis le récepteur")

files = script_files()
density_paths = script_density_files()
uploaded = st.sidebar.file_uploader("Ou importer un fichier NetCDF (.nc)", type=["nc"], accept_multiple_files=True)
paths = list(files)
for u in uploaded or []:
    p = Path(tempfile.mkdtemp()) / u.name
    p.write_bytes(u.getbuffer())
    paths.append(str(p))

if density_paths:
    from randi_gen.density import _load_any

    with st.expander("Vessel density — données brutes (raster utilisé)", expanded=True):
        for dp in density_paths:
            try:
                raster = _load_any(dp)
            except Exception as e:
                st.warning(f"{dp} : lecture impossible ({e})")
                continue
            v = np.asarray(raster.values, float)
            v = np.where(np.isfinite(v) & (v > 0), np.log10(v), np.nan)
            dfig = go.Figure(go.Heatmap(x=raster.lons, y=raster.lats, z=v, colorscale="YlOrRd",
                                         colorbar=dict(title="log10 h/km²/mois")))
            dfig.update_layout(title=f"{Path(dp).name} — densité brute (h/km²/mois, log10), crs={raster.crs}",
                               xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)", height=520)
            st.plotly_chart(dfig, use_container_width=True)
            st.caption(f"source : {raster.source[:200]} | max = {float(np.nanmax(raster.values)):.1f} h/km²/mois")

if not paths:
    st.info("Aucun fichier .nc fourni. Lancez par exemple : `randi-gen serve output/randi_map_f1000Hz.nc` ou importez un fichier .nc.")
    st.stop()

fields = [read_field(p) for p in paths]
freqs = [f.frequency_hz for f in fields]

fi = st.sidebar.selectbox("Fréquence", range(len(freqs)), format_func=lambda i: f"{freqs[i]:g} Hz")
f = fields[fi]
az = st.sidebar.slider("Azimut (°)", 0.0, 355.0, 0.0, 5.0)
ai = int(np.argmin(np.abs(((f.azimuth - az + 180.0) % 360.0) - 180.0)))
cmap = st.sidebar.selectbox("Colormap", ["Viridis", "Plasma", "Inferno", "Cividis", "Turbo"])

z = f.noise_level[:, :, ai]
fig = go.Figure(go.Heatmap(x=f.lon, y=f.lat, z=z, colorscale=cmap, colorbar=dict(title="dB re 1 µPa²/Hz")))
fig.update_layout(
    title=f"Niveau directif — {freqs[fi]:g} Hz, azimut {f.azimuth[ai]:g}°",
    xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)", height=620,
)
st.plotly_chart(fig, use_container_width=True)

omni = f.omnidirectional
flat = np.argsort(omni.ravel())[::-1][:10]
opts = [(int(i // omni.shape[1]), int(i % omni.shape[1])) for i in flat]
sel = st.selectbox(
    "Point chaud (10 plus hauts niveaux omnidirectionnels)",
    range(len(opts)),
    format_func=lambda i: f"lat={f.lat[opts[i][0]]:.3f}°, lon={f.lon[opts[i][1]]:.3f}° — NLomni={omni[opts[i]]:.1f} dB",
)
hy, hx = opts[sel]

pol = go.Figure(go.Barpolar(r=f.noise_level[hy, hx, :], theta=f.azimuth, marker_color=z[hy, hx]))
pol.update_polars(angularaxis=dict(rotation=90, direction="clockwise"))
pol.update_layout(title=f"Diagramme polaire — lat={f.lat[hy]:.3f}°, lon={f.lon[hx]:.3f}°", height=460)

lat2d, lon2d = np.meshgrid(f.lat, f.lon, indexing="ij")
sub = max(1, max(lat2d.shape) // 40)
zz = z[::sub, ::sub]
sc = go.Figure(go.Scatter3d(
    x=lon2d[::sub, ::sub].ravel(), y=lat2d[::sub, ::sub].ravel(), z=zz.ravel(),
    mode="markers", marker=dict(size=2, color=zz.ravel(), colorscale=cmap),
))
sc.update_layout(
    title=f"Vue 3D sous-échantillonnée — azimut {f.azimuth[ai]:g}°",
    height=460, scene=dict(xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)", zaxis_title="dB"),
)

c1, c2 = st.columns(2)
c1.plotly_chart(pol, use_container_width=True)
c2.plotly_chart(sc, use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("NL max (azimut sélectionné)", f"{float(z.max()):.1f} dB")
m2.metric("Indice de directivité max", f"{float(f.directionality_index.max()):.1f} dB")
m3.metric("NL omnidirectionnel max", f"{float(omni.max()):.1f} dB")

import pandas as pd

rows = [
    {
        "lat": float(f.lat[i]),
        "lon": float(f.lon[j]),
        "NL_omni_dB": float(omni[i, j]),
        "DI_dB": float(f.directionality_index[i, j]),
        "azimut_max_deg": float(f.azimuth[int(np.argmax(f.noise_level[i, j, :]))]),
    }
    for i, j in opts
]
st.dataframe(pd.DataFrame(rows), use_container_width=True)
