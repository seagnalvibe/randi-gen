# randi-gen

Génération de **cartes 3D de prédiction de bruit ambiant sous-marin directif** (modèle de type **RANDI** : *Research Ambient Noise Directionality*), à partir de la **fréquence** et de **statistiques AIS** (positions de navires).

- Sortie principale : fichier **NetCDF 3D** — dimensions **`lat`, `lon`, `azimuth`** (azimut vu depuis le récepteur, 0 = Nord, sens horaire), variable `noise_level` en dB re 1 µPa²/Hz, plus `omnidirectional_level` et `directionality_index` (2D lat/lon). Un fichier par fréquence.
- **Interfaces de visualisation** : figures PNG (matplotlib), viewer HTML interactif autonome (plotly, curseur d'azimut/fréquence), et interface web **Streamlit** en local (`127.0.0.1`).
- **Cas test intégré** : zone **entre Toulon et la Corse** (`--bbox 41.8 5.5 43.4 9.6`, valeur par défaut).

## Installation

```bash
pip install -e .          # noyau : numpy, xarray, netCDF4, matplotlib, plotly, requests
pip install -e '.[ui]'    # + streamlit pour l'interface web
pip install -e '.[test]'  # + pytest
```

## Démarrage rapide (cas test Toulon–Corse)

```bash
# 1) carte 3D NetCDF à 3 fréquences + figures PNG + viewer HTML
randi-gen run --bbox 41.8 5.5 43.4 9.6 --res 0.05 --freq 100 1000 10000 --az-step 10 \
    --ais synthetic --out output --figures --html
# ou simplement :
python examples/run_toulon_corsica.py

# 2) visualisations
randi-gen plot output/randi_map_f1000Hz.nc --azimuth 90 --polar 43.0 6.5 --3d -o carte.png
randi-gen html output/randi_map_f*.nc -o output/viewer.html   # ouvrir dans un navigateur
randi-gen serve output/randi_map_f1000Hz.nc --port 8501       # http://127.0.0.1:8501
```

`python -m randi_gen ...` fonctionne aussi à la place de `randi-gen`.

## Statistiques AIS "disponibles sur Internet"

Par défaut (`--ais synthetic`), une **climatologie AIS synthétique réaliste** de la Méditerranée
nord-ouest est utilisée (lignes de ferries Toulon–Ajaccio/Bastia/Île Rousse, approches de
Marseille/Toulon, trafic côtier corse) — pratique hors-ligne et pour les tests.

Sur une machine connectée à Internet, on branche des statistiques AIS réelles :

```bash
# URL http(s) d'un CSV ou GeoJSON de points AIS (ex. export EMODnet Human Activities,
# API MarineTraffic, marinecadastre, etc.)
randi-gen run --ais-url "https://exemple.org/ais_points.csv" ...
# ou fichier local téléchargé au préalable
randi-gen run --ais-file ais_med.csv ...
```

Format CSV attendu : colonnes `lon,lat` (alias `longitude/lng`, `latitude`) + optionnel
`type`/`ship_type` (`tanker`, `cargo`, `passenger`/`ferry`, `fishing`, autre) et
`speed`/`sog` (nœuds). GeoJSON de `Point` accepté (propriétés `type`, `speed`/`sog`).

### Mode « vessel density » (raster de densité AIS, ex. EMODnet)

Alternative aux points AIS : un **raster de densité de navires** (heures/km²/mois, mailles ~1 km),
par ex. la couche **EMODnet Human Activities « Vessel Density Map »** (WMS/WCS :
`ows.emodnet-humanactivities.eu`, couches `vesseldensity_allavg`, `vesseldensity_09avg` cargo,
`_10` tanker, `_08` passenger, `_01` fishing, ...). Les navires sont échantillonnés
proportionnellement à la densité : `navires/cellule = densité × aire(km²) / heures_par_mois`.

```bash
# téléchargement direct via le WCS EMODnet sur la bbox ; si indisponible,
# un raster de TEST synthétique (format EMODnet, zone Toulon-Corse) est généré et sauvegardé
randi-gen run --bbox 41.8 5.5 43.4 9.6 --ais-density emodnet --emodnet-layer vesseldensity_allavg --out output_emodnet --figures --html
# ou raster téléchargé depuis le portail EMODnet (GeoTIFF zippé) ou local :
randi-gen run --ais-density VesselDensity_All_2023.tif ...
randi-gen run --ais-density https://.../vessel_density.zip ...
# ou par type de navire : chaque type a SA propre couche EMODnet dédiée
# (vesseldensity_XXavg, codes 00-12 : other=00, fishing=01, service=02, dredging=03,
#  sailing=04, pleasure=05, highspeed=06, tug=07, passenger=08, cargo=09, tanker=10,
#  military=11, unknown=12) et contribue à la sommation de bruit avec son propre
# niveau de source SL(f, type, v) (tanker 185, cargo 182, passager 178, pêche 168,
# autre 165 dB re 1 µPa @ 1 m à 100 Hz) :
randi-gen run --ais-density-type cargo=emodnet --ais-density-type tanker=emodnet \
    --ais-density-type passenger=emodnet --ais-density-type fishing=emodnet \
    --ais-density-type other=emodnet ...
# SPEC peut aussi être un fichier/URL local par type. En mode par type, le raster
# combiné (somme des couches par type) est sauvegardé comme density_used.tif/.asc
# et affiché (density_used.png), comme en mode --ais-density.
# Cas test : python examples/run_toulon_corsica_emodnet_types.py
```

Formats acceptés : `.tif/.tiff` (rasterio), `.asc/.grd` (ESRI ASCII, sans dépendance),
`.nc` (NetCDF 2D), `.zip` (premier raster trouvé). Options : `--density-hours-per-month` (720),
`--density-scale`, `--max-ships` (20000). Le raster utilisé est sauvé (`density_used.tif`,
`emodnet_vesseldensity_TEST.asc` en repli) dans le répertoire de sortie. Cas test :
`python examples/run_toulon_corsica_emodnet.py`.

**Affichage des données brutes vessel density** : `density_used.png` est produit à chaque
run en mode densité ; on peut aussi rendre n'importe quel raster avec
`randi-gen density-plot raster.tif -o densite.png`, l'ajouter au viewer HTML
(`randi-gen html map.nc --density density_used.tif ...`, carte en haut) et la retrouver
dans l'interface Streamlit (`randi-gen serve ...` détecte `density_used.tif/.asc` à côté
des `.nc`). La chaîne de conversion densité → navires → niveau directif est détaillée dans
[docs/MODEL.md](docs/MODEL.md#chaîne-complète--du-vessel-density-brut-au-niveau-de-bruit-directif-nllat-lon-az-f).

## Structure du NetCDF (CF-1.8)

| Élément | Dimensions | Description |
|---|---|---|
| `lat`, `lon` | `lat`, `lon` | coordonnées grille (degrees_north/east) |
| `azimuth` | `azimuth` | centres des secteurs (degree, 0=N horaire) |
| `noise_level` | **`lat, lon, azimuth`** | niveau spectral directif prédit (dB re 1 µPa²/Hz) |
| `omnidirectional_level` | `lat, lon` | niveau omnidirectionnel (dB re 1 µPa²/Hz) |
| `directionality_index` | `lat, lon` | DI = 10·log10(max/min sur azimut) (dB) |

Attributs globaux : `frequency_hz`, `ais_source`, `n_ships`, `bbox_south_west_north_east`,
`grid_resolution_deg`, `azimuth_step_deg`, `max_range_km`, `spreading_exponent`, `min_range_m`,
`history`, `references`.

## Modèle (résumé, voir docs/MODEL.md)

Pour chaque cellule réceptrice et chaque secteur d'azimut `az`, somme incohérente des navires
AIS situés dans ce secteur, atténués par la perte de transmission :

```
NL(lat, lon, az, f) = 10·log10( Σ_navires(az) 10^((SL(f,type,v) − TL(r,f))/10) + P_fond(f) )
TL(r, f) = 10·n·log10(r) + α_Thorp(f)·r/1000        (r en m, plafonnée à min_range_m)
SL(f)    = SL_100Hz(type) − 17·log10(f/100) + clip(5·log10(v/12), ±6)
```

avec `n` = exposant d'étalement (1,5 par défaut : intermédiaire cylindrique/sphérique).
Hypothèses et limites détaillées dans [docs/MODEL.md](docs/MODEL.md).

## Tests

```bash
python -m pytest            # inclut le cas d'intégration Toulon–Corse (NetCDF 3D + figures + HTML)
```

## Arborescence

```
src/randi_gen/
  cli.py         # commandes run / plot / html / serve
  config.py      # bbox par défaut (Toulon–Corse)
  geo.py         # haversine, cap, interpolation de routes
  ais.py         # catalogue AIS : synthétique Méditerranée NO, CSV/GeoJSON, URL
  acoustics.py   # niveaux de source (Ross/Wenz), Thorp, perte de transmission
  model.py       # modèle directif type RANDI
  netcdf_io.py   # écriture/lecture NetCDF 3D (lat, lon, azimuth)
  visualize.py   # PNG matplotlib + viewer HTML plotly
  app.py         # interface Streamlit (local 127.0.0.1)
examples/run_toulon_corsica.py
tests/
docs/MODEL.md
```
