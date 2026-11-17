"""Cas test : prédiction RANDI entre Toulon et la Corse.

Lancer depuis la racine du dépôt :
    python examples/run_toulon_corsica.py

Produit dans examples/output/ :
- randi_map_f<freq>Hz.nc  : carte 3D NetCDF (lat, lon, azimut) par fréquence ;
- randi_map_f<freq>Hz_figures/*.png : figures de démonstration ;
- viewer.html             : viewer interactif (curseur d'azimut/fréquence).
"""

from pathlib import Path

from randi_gen.cli import main

OUT = Path(__file__).resolve().parent / "output"

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "run",
                "--bbox", "41.8", "5.5", "43.4", "9.6",  # entre Toulon et la Corse
                "--res", "0.05",
                "--freq", "100", "1000", "10000",
                "--az-step", "10",
                "--ais", "synthetic",
                "--seed", "42",
                "--out", str(OUT),
                "--figures",
                "--html",
            ]
        )
    )
