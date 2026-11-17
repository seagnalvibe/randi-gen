"""Interface en ligne de commande randi-gen."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import DEFAULT_AZIMUTH_STEP_DEG, DEFAULT_BBOX, DEFAULT_FREQS, DEFAULT_RES_DEG


def build_parser():
    p = argparse.ArgumentParser(
        prog="randi-gen",
        description="Génère une carte 3D NetCDF (lat, lon, azimut) de prédiction de bruit ambiant "
                    "sous-marin directif (type RANDI), à partir de la fréquence et de statistiques AIS.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="calcule la carte et écrit le(s) fichier(s) NetCDF 3D")
    r.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                   default=list(DEFAULT_BBOX), help="boîte lat/lon en degrés (défaut: zone Toulon-Corse)")
    r.add_argument("--res", type=float, default=DEFAULT_RES_DEG, help="résolution de la grille (°)")
    r.add_argument("--freq", nargs="+", type=float, default=list(DEFAULT_FREQS), help="fréquence(s) en Hz")
    r.add_argument("--az-step", type=float, default=DEFAULT_AZIMUTH_STEP_DEG, help="pas angulaire d'azimut (°, doit diviser 360)")
    r.add_argument("--ais", default="synthetic", help="'synthetic' ou chemin d'un fichier CSV/GeoJSON de points AIS")
    r.add_argument("--ais-file", default=None, help="fichier CSV/GeoJSON de points AIS (colonnes lon,lat[,type,speed])")
    r.add_argument("--ais-url", default=None, help="URL http(s) de statistiques/points AIS (CSV ou GeoJSON), ex. export EMODnet ou MarineTraffic")
    r.add_argument("--seed", type=int, default=42, help="graine du générateur AIS synthétique")
    r.add_argument("--ships-scale", type=float, default=1.0, help="facteur d'échelle du nombre de navires (source synthétique)")
    r.add_argument("--max-range-km", type=float, default=300.0, help="portée max d'un navire (km)")
    r.add_argument("--min-range-m", type=float, default=500.0, help="distance min de validité de la TL (m)")
    r.add_argument("--spreading", type=float, default=1.5, help="exposant d'étalement : 1=cylindrique, 2=sphérique")
    r.add_argument("--out", default="output", help="répertoire de sortie")
    r.add_argument("--figures", action="store_true", help="génère aussi des figures PNG de démonstration")
    r.add_argument("--html", action="store_true", help="génère aussi un viewer HTML interactif")
    r.set_defaults(func=cmd_run)

    pl = sub.add_parser("plot", help="trace des figures PNG depuis un fichier NetCDF")
    pl.add_argument("file", help="fichier .nc produit par 'run'")
    pl.add_argument("--azimuth", type=float, default=None, help="azimut de la carte (défaut: azimut du max global)")
    pl.add_argument("--omni", action="store_true", help="carte du niveau omnidirectionnel")
    pl.add_argument("--di", action="store_true", help="carte de l'indice de directivité")
    pl.add_argument("--polar", nargs=2, type=float, metavar=("LAT", "LON"), default=None, help="ajoute un diagramme polaire en un point")
    pl.add_argument("--3d", action="store_true", help="ajoute une surface 3D")
    pl.add_argument("-o", "--out", default=None, help="fichier PNG de sortie")
    pl.set_defaults(func=cmd_plot)

    h = sub.add_parser("html", help="génère un viewer HTML interactif (curseur d'azimut, fréquence)")
    h.add_argument("files", nargs="+", help="fichier(s) .nc")
    h.add_argument("-o", "--out", default="viewer.html", help="fichier HTML de sortie")
    h.set_defaults(func=cmd_html)

    s = sub.add_parser("serve", help="lance l'interface Streamlit en local (127.0.0.1 uniquement)")
    s.add_argument("files", nargs="+", help="fichier(s) .nc")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(func=cmd_serve)
    return p


def cmd_run(args):
    from .ais import load_ais
    from .model import ModelParams, compute_directional_noise, make_grid
    from .netcdf_io import output_path, write_field

    bbox = tuple(float(v) for v in args.bbox)
    ships = load_ais(source=args.ais, file=args.ais_file, url=args.ais_url,
                     bbox=bbox, seed=args.seed, ships_scale=args.ships_scale)
    print(f"[randi-gen] AIS : {len(ships)} navires (source : {ships.source})")
    lats, lons = make_grid(bbox, args.res)
    print(f"[randi-gen] grille : {lats.size} x {lons.size} cellules (res={args.res}°), bbox={bbox}")
    params = ModelParams(azimuth_step_deg=args.az_step, max_range_km=args.max_range_km,
                         min_range_m=args.min_range_m, spreading=args.spreading)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fields, paths = [], []
    for freq in args.freq:
        field = compute_directional_noise(lats, lons, ships, freq, params)
        field.meta.update(ais_source=ships.source, bbox=list(bbox), resolution_deg=args.res)
        path = output_path(freq, outdir)
        write_field(field, path)
        fields.append(field)
        paths.append(path)
        print(f"[randi-gen] f={freq:g} Hz -> {path} | NL omni max={field.omnidirectional.max():.1f} dB, "
              f"DI max={field.directionality_index.max():.1f} dB")
    if args.figures:
        import matplotlib

        matplotlib.use("Agg")
        from .visualize import save_demo_figures

        for f, p in zip(fields, paths):
            fdir = outdir / f"{p.stem}_figures"
            save_demo_figures(f, fdir)
            print(f"[randi-gen] figures -> {fdir}")
    if args.html:
        from .visualize import build_html_viewer

        hp = outdir / "viewer.html"
        build_html_viewer(fields, hp)
        print(f"[randi-gen] viewer HTML -> {hp}")
    return 0


def cmd_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from . import visualize as viz
    from .netcdf_io import read_field

    field = read_field(args.file)
    out = Path(args.out) if args.out else Path(args.file).with_suffix(".png")
    if args.di:
        fig, _ = viz.plot_di(field)
    elif args.omni:
        fig, _ = viz.plot_omni(field)
    else:
        if args.azimuth is None:
            az = float(field.azimuth[int(np.unravel_index(int(np.argmax(field.noise_level)), field.noise_level.shape)[2])])
        else:
            az = float(args.azimuth)
        fig, _ = viz.plot_map_azimuth(field, az)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[randi-gen] figure -> {out}")
    if args.polar is not None:
        pfig = viz.plot_polar(field, args.polar[0], args.polar[1])
        p = out.with_name(out.stem + "_polar.png")
        pfig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(pfig)
        print(f"[randi-gen] figure -> {p}")
    if args.__dict__.get("3d"):
        if args.azimuth is None and not (args.di or args.omni):
            az = float(field.azimuth[int(np.unravel_index(int(np.argmax(field.noise_level)), field.noise_level.shape)[2])])
        else:
            az = float(args.azimuth or 0.0)
        sfig = viz.plot_surface3d(field, az)
        p = out.with_name(out.stem + "_3d.png")
        sfig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(sfig)
        print(f"[randi-gen] figure -> {p}")
    return 0


def cmd_html(args):
    from .netcdf_io import read_field
    from .visualize import build_html_viewer

    fields = [read_field(p) for p in args.files]
    out = build_html_viewer(fields, args.out)
    print(f"[randi-gen] viewer HTML -> {out}")
    return 0


def cmd_serve(args):
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("streamlit n'est pas installé : pip install 'randi-gen[ui]'", file=sys.stderr)
        return 2
    for f in args.files:
        if not Path(f).exists():
            print(f"fichier introuvable : {f}", file=sys.stderr)
            return 2
    app = Path(__file__).with_name("app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", "127.0.0.1",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--", *args.files,
    ]
    print(f"[randi-gen] interface locale : http://127.0.0.1:{args.port}  (Ctrl+C pour arrêter)")
    return subprocess.call(cmd)


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)
