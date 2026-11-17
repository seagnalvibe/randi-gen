"""Cas test Toulon-Corse avec le mode « vessel density » EMODnet.

Lancer : python examples/run_toulon_corsica_emodnet.py

- essaie de télécharger la couche EMODnet vessel density (WCS) sur la bbox Toulon-Corse ;
- si le téléchargement échoue (hors-ligne, service en erreur), génère un raster de
  TEST synthétique au format EMODnet (sauvegardé dans le répertoire de sortie) ;
- échantillonne des navires proportionnellement à la densité et produit les NetCDF 3D.
"""

from pathlib import Path

from randi_gen.cli import main

OUT = Path(__file__).resolve().parent / "output" / "emodnet"

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "run",
                "--bbox", "41.8", "5.5", "43.4", "9.6",
                "--res", "0.05",
                "--freq", "100", "1000", "10000",
                "--n-azimuth", "36",
                "--ais-density", "emodnet",
                "--emodnet-layer", "vesseldensity_allavg",
                "--max-ships", "20000",
                "--out", str(OUT),
                "--figures",
                "--html",
            ]
        )
    )
