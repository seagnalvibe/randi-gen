"""Interface en ligne de commande randi-gen."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import DEFAULT_BBOX, DEFAULT_FREQS, DEFAULT_N_AZIMUTH, DEFAULT_RES_DEG


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
    r.add_argument("--n-azimuth", type=int, default=None,
                   help=f"nombre de secteurs d'azimut sur 360° (défaut: {DEFAULT_N_AZIMUTH}) ; ignoré si --az-pattern est donné")
    r.add_argument("--az-aperture", type=float, default=None,
                   help="ouverture (FWHM, °) de la fenêtre gaussienne d'azimut centrée sur le cap de chaque navire "
                        "(défaut: 360/--n-azimuth) ; répartit l'énergie d'un navire sur les secteurs voisins au lieu d'un "
                        "tout-ou-rien, ce qui lisse les cartes (niveau omnidirectionnel inchangé)")
    r.add_argument("--az-pattern", default=None, metavar="FILE.mat",
                   help="fichier .mat tabulant pour chaque azimut la directivité à appliquer en fonction du cap sous "
                        "lequel le récepteur voit la source (variables azimuth, angle, directivity ; voir pattern.py) : "
                        "le nombre d'azimuts et l'échantillonnage angulaire sont extraits du fichier et la directivité "
                        "est interpolée (linéairement, périodique sur 360°), ce qui permet de faire varier la forme de "
                        "l'ouverture suivant l'angle")
    r.add_argument("--ais", default="synthetic", help="'synthetic' ou chemin d'un fichier CSV/GeoJSON de points AIS")
    r.add_argument("--ais-file", default=None, help="fichier CSV/GeoJSON de points AIS (colonnes lon,lat[,type,speed])")
    r.add_argument("--ais-url", default=None, help="URL http(s) de statistiques/points AIS (CSV ou GeoJSON), ex. export EMODnet ou MarineTraffic")
    r.add_argument("--ais-density", default=None, metavar="SPEC",
                   help="raster de densité de navires : 'emodnet' (WCS EMODnet, repli raster de test si indisponible), URL ou fichier local (.tif/.asc/.nc/.zip)")
    r.add_argument("--emodnet-layer", default="vesseldensity_allavg",
                   help="couche EMODnet vessel density (défaut: vesseldensity_allavg ; par type: vesseldensity_09avg cargo, _10 tanker, _08 passenger, _01 fishing)")
    r.add_argument("--ais-density-type", action="append", default=[], metavar="TYPE=SPEC",
                   help="raster de densité par type de navire, répétable (TYPE = nom de type EMODnet : cargo,tanker,passenger,fishing,other,service,dredging,sailing,pleasure,highspeed,tug,military,unknown ; SPEC='emodnet' -> couche vesseldensity_XXavg dédiée, ou chemin/URL). Chaque type contribue à la sommation avec son propre niveau de source SL(f,type,v)")
    r.add_argument("--density-hours-per-month", type=float, default=720.0,
                   help="heures par mois pour convertir la densité EMODnet (h/km²/mois) en navires (défaut 720)")
    r.add_argument("--density-scale", type=float, default=1.0, help="facteur d'échelle sur la densité échantillonnée")
    r.add_argument("--max-ships", type=int, default=20000, help="nombre max de navires échantillonnés depuis le raster")
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
    h.add_argument("--density", default=None, help="raster de vessel density (.tif/.asc/.nc) à afficher en données brutes")
    h.add_argument("-o", "--out", default="viewer.html", help="fichier HTML de sortie")
    h.set_defaults(func=cmd_html)

    d = sub.add_parser("density-plot", help="affiche un raster de vessel density brut en PNG")
    d.add_argument("file", help="raster .tif/.tiff/.asc/.grd/.nc/.zip ou 'emodnet'")
    d.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"), default=None)
    d.add_argument("--linear", action="store_true", help="échelle linéaire au lieu de log")
    d.add_argument("-o", "--out", default=None, help="fichier PNG de sortie")
    d.set_defaults(func=cmd_density_plot)

    s = sub.add_parser("serve", help="lance l'interface Streamlit en local (127.0.0.1 uniquement)")
    s.add_argument("files", nargs="+", help="fichier(s) .nc (et/ou raster de densité .tif/.asc à afficher en brut)")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(func=cmd_serve)
    return p


def _ships_from_density(args, bbox, outdir):
    import numpy as np

    from .ais import ShipCatalog
    from .density import (EMODNET_TYPE_LAYER, load_density_raster, sample_catalog_from_density,
                          write_ascii_grid)

    if args.ais_density_type:
        from .density import combine_rasters

        cats, rasters = [], []
        for item in args.ais_density_type:
            t, _, sp = item.partition("=")
            t = t.strip().lower()
            if t not in EMODNET_TYPE_LAYER or not sp:
                raise SystemExit(f"--ais-density-type attendu TYPE=SPEC avec TYPE in {sorted(EMODNET_TYPE_LAYER)} (reçu: {item!r})")
            if sp.strip().lower() == "emodnet":
                raster, note = load_density_raster("emodnet", bbox=bbox, outdir=outdir, seed=args.seed,
                                                   layer=EMODNET_TYPE_LAYER[t])
                print(f"[randi-gen] densité type={t} [{note}] : couche EMODnet {EMODNET_TYPE_LAYER[t]}, "
                      f"raster {raster.shape}, max={np.nanmax(raster.values):.1f} h/km²/mois")
            else:
                raster, note = load_density_raster(sp, bbox=bbox, outdir=outdir, seed=args.seed)
                print(f"[randi-gen] densité type={t} [{note}] : {sp}, raster {raster.shape}, "
                      f"max={np.nanmax(raster.values):.1f} h/km²/mois")
            rasters.append(raster)
            cats.append(sample_catalog_from_density(raster, bbox=bbox, seed=args.seed,
                                                    hours_per_month=args.density_hours_per_month,
                                                    scale=args.density_scale, max_ships=args.max_ships,
                                                    fixed_type=t))
        ships = ShipCatalog.concat(cats)
        ships.source = "vessel-density-per-type; " + ships.source
        combined = combine_rasters(rasters)
        if combined is not None:
            try:
                from .density import write_geotiff

                rp = write_geotiff(combined, Path(outdir) / "density_used.tif")
            except ImportError:
                rp = write_ascii_grid(combined, Path(outdir) / "density_used.asc")
            from .visualize import plot_density_raster

            plot_density_raster(combined, Path(outdir) / "density_used.png")
            print(f"[randi-gen] raster combiné (somme des types) {combined.shape}, "
                  f"max={np.nanmax(combined.values):.1f} h/km²/mois -> {rp} et density_used.png")
        return ships, combined
    raster, note = load_density_raster(args.ais_density, bbox=bbox, outdir=outdir, seed=args.seed,
                                       layer=args.emodnet_layer)
    try:
        from .density import write_geotiff

        rp = write_geotiff(raster, Path(outdir) / "density_used.tif")
    except ImportError:
        rp = write_ascii_grid(raster, Path(outdir) / "density_used.asc")
    print(f"[randi-gen] raster densité [{note}] {raster.shape}, crs={raster.crs}, "
          f"max={np.nanmax(raster.values):.1f}, moy={np.nanmean(raster.values):.2f} h/km²/mois -> {rp}")
    from .visualize import plot_density_raster

    plot_density_raster(raster, Path(outdir) / "density_used.png")
    print(f"[randi-gen] carte du raster brut -> {Path(outdir) / 'density_used.png'}")
    ships = sample_catalog_from_density(raster, bbox=bbox, seed=args.seed,
                                        hours_per_month=args.density_hours_per_month,
                                        scale=args.density_scale, max_ships=args.max_ships)
    return ships, raster


def cmd_run(args):
    import numpy as np

    from .ais import load_ais
    from .model import ModelParams, compute_directional_noise, make_grid
    from .netcdf_io import output_path, write_field

    bbox = tuple(float(v) for v in args.bbox)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    density_raster = None
    if args.ais_density or args.ais_density_type:
        ships, density_raster = _ships_from_density(args, bbox, outdir)
    else:
        ships = load_ais(source=args.ais, file=args.ais_file, url=args.ais_url,
                         bbox=bbox, seed=args.seed, ships_scale=args.ships_scale)
    print(f"[randi-gen] AIS : {len(ships)} navires (source : {ships.source})")
    lats, lons = make_grid(bbox, args.res)
    print(f"[randi-gen] grille : {lats.size} x {lons.size} cellules (res={args.res}°), bbox={bbox}")
    n_azimuth = args.n_azimuth if args.n_azimuth is not None else DEFAULT_N_AZIMUTH
    az_pattern = None
    if args.az_pattern:
        from .pattern import load_pattern

        az_pattern = load_pattern(args.az_pattern, n_azimuth_hint=n_azimuth)
        print(f"[randi-gen] directivité tabulée : {az_pattern.n_azimuth} azimuts, "
              f"{az_pattern.angle.size} échantillons d'angle depuis {args.az_pattern}")
        if args.n_azimuth is not None and az_pattern.azimuth.size != n_azimuth:
            print(f"[randi-gen] note : --n-azimuth {n_azimuth} ignoré, la grille vient du fichier "
                  f"({az_pattern.azimuth.size} azimuts)")
    params = ModelParams(n_azimuth=n_azimuth, azimuth_aperture_deg=args.az_aperture, azimuth_pattern=az_pattern,
                         max_range_km=args.max_range_km, min_range_m=args.min_range_m, spreading=args.spreading)
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
        build_html_viewer(fields, hp, density_raster=density_raster)
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


def cmd_density_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .density import load_density_raster
    from .visualize import plot_density_raster

    bbox = tuple(args.bbox) if args.bbox else None
    raster, note = load_density_raster(args.file, bbox=bbox, log=print)
    out = args.out or str(Path(args.file).with_suffix(".png"))
    fig, _ = plot_density_raster(raster, out, log_scale=not args.linear)
    plt.close(fig)
    print(f"[randi-gen] raster [{note}] {raster.shape} -> {out}")
    return 0


def cmd_html(args):
    from .netcdf_io import read_field
    from .visualize import build_html_viewer

    fields = [read_field(p) for p in args.files]
    density_raster = None
    if args.density:
        from .density import load_density_raster

        density_raster, _note = load_density_raster(args.density, log=print)
    out = build_html_viewer(fields, args.out, density_raster=density_raster)
    print(f"[randi-gen] viewer HTML -> {out}")
    return 0


def cmd_serve(args):
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("streamlit n'est pas installé : pip install 'randi-gen[ui]'", file=sys.stderr)
        return 2
    extra = []
    for f in args.files:
        if not Path(f).exists():
            print(f"fichier introuvable : {f}", file=sys.stderr)
            return 2
        if not f.lower().endswith(".nc"):
            extra.append(f)  # raster de densité à afficher en brut
        else:
            for cand in ("density_used.tif", "density_used.asc", "emodnet_vesseldensity_TEST.asc"):
                p = Path(f).parent / cand
                if p.exists() and str(p) not in extra:
                    extra.append(str(p))
    app = Path(__file__).with_name("app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", "127.0.0.1",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--", *args.files, *extra,
    ]
    print(f"[randi-gen] interface locale : http://127.0.0.1:{args.port}  (Ctrl+C pour arrêter)")
    return subprocess.call(cmd)


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)
