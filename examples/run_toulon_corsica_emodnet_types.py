"""Cas test Toulon-Corse en mode « vessel density par type de navire » (EMODnet).

Lancer : python examples/run_toulon_corsica_emodnet_types.py

Chaque type de navire est échantillonné depuis SA propre couche EMODnet
(vesseldensity_XXavg : cargo=09, tanker=10, passenger=08, fishing=01, other=00...),
puis contribue à la sommation de bruit avec son propre niveau de source SL(f, type, v).
Si le WCS EMODnet est indisponible, chaque couche retombe sur le raster de TEST
synthétique (sauvegardé dans le répertoire de sortie).
"""

from pathlib import Path

from randi_gen.cli import main

OUT = Path(__file__).resolve().parent / "output" / "emodnet_types"

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "run",
                "--bbox", "41.8", "5.5", "43.4", "9.6",
                "--res", "0.05",
                "--freq", "100", "1000", "10000",
                "--n-azimuth", "36",
                "--ais-density-type", "cargo=emodnet",
                "--ais-density-type", "tanker=emodnet",
                "--ais-density-type", "passenger=emodnet",
                "--ais-density-type", "fishing=emodnet",
                "--ais-density-type", "other=emodnet",
                "--max-ships", "20000",
                "--out", str(OUT),
                "--figures",
                "--html",
            ]
        )
    )
