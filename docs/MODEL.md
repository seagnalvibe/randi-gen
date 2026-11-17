# Modèle de bruit ambiant directif (type RANDI)

Approche de type **RANDI** (*Research Ambient Noise Directionality*) : la directivité du bruit
ambiant est prédite à partir de la **distribution spatiale des sources de trafic maritime**
(données AIS), intégrées par secteur d'azimut autour de chaque point récepteur.

## Grandeurs et unités

- Niveaux spectraux en **dB re 1 µPa²/Hz** (1 Hz de bande), source à **1 m** pour les SL.
- Azimut : **0 = Nord, sens horaire**, mesuré **depuis le récepteur vers la source**, secteurs de
  `azimuth_step_deg` (10° par défaut, doit diviser 360). Chaque navire n'est pas affecté à un seul
  secteur (tout-ou-rien) mais pondéré sur tous les secteurs par une **ouverture gaussienne** centrée
  sur son cap, de largeur à mi-hauteur (FWHM) `azimuth_aperture_deg` = `azimuth_step_deg` par défaut.
- Fréquence `f` en Hz, distance `r` en m, absorption en dB/km.

## Niveau de source d'un navire

Basé sur les courbes de Wenz (1962) / Ross (1976) :

```
SL(f, type, v) = SL_100Hz(type) − 17·log10(f / 100 Hz) + clip(5·log10(v / 12 kn), −6, +6)   [dB re 1 µPa @ 1 m]
SL_100Hz : tanker 185, cargo 182, passenger/ferry 178, fishing 168, other 165
```

## Perte de transmission

Étalement géométrique d'exposant `n` (1 = cylindrique, 2 = sphérique, **1,5** par défaut)
+ absorption de Thorp (1967) :

```
TL(r, f) = 10·n·log10(r) + α(f)·r/1000     avec r plafonnée à min_range_m (500 m)
α(f) [dB/km] = 0,11·f²/(1+f²) + 44·f²/(4100+f²) + 2,75e-4·f² + 0,003      (f en kHz)
```

## Niveau directif par cellule et par secteur d'azimut

Somme **incohérente** (énergétique) des navires AIS, chacun pondéré dans chaque secteur par une
**ouverture gaussienne** centrée sur son cap vu du récepteur, atténués par la TL, plus un fond
diffus omnidirectionnel `P_fond` :

```
NL(lat, lon, az, f) = 10·log10( Σ_{navires, r ≤ max_range} w(az, cap) · 10^((SL − TL(r,f))/10) + 10^(P_fond(f)/10) )
w(az, cap) = exp( −4·ln2 · (Δaz / ouverture)² ),   Δaz = écart angulaire signé (az − cap) ramené à [−180, 180[
ouverture (FWHM) = azimuth_aperture_deg  (= azimuth_step_deg par défaut, ex. 10° ; option --az-aperture)
P_fond(f) = 60 − 12·log10(f / 100 Hz)     [dB re 1 µPa²/Hz]
```

Les poids `w` sont **normalisés par navire** (Σ_az w = 1) : l'énergie totale reçue d'un navire est
conservée, donc le **niveau omnidirectionnel est inchangé** par rapport à un découpage tout-ou-rien ;
seules les cartes directives `NL(az)` sont **lissées** (un navire proche d'une frontière de secteur
alimente aussi le secteur voisin, au lieu de créer une marche d'escalier entre secteurs).

## Grandeurs dérivées

```
omnidirectionnel(lat, lon)  = 10·log10( Σ_navires 10^((SL − TL)/10) + 10^(P_fond/10) )
                            = 10·log10( Σ_az 10^(NL(az)/10) − (n_az − 1)·10^(P_fond/10) )
indice de directivité DI    = 10·log10( max_az NL / min_az NL )     [dB]
```

`P_fond` est un fond **omnidirectionnel** : il est compté **une seule fois** dans le niveau
omnidirectionnel (et une fois par secteur dans `NL(az)`, qui inclut le fond ambiant du secteur),
jamais `n_az` fois — sinon le plancher omnidirectionnel est artificiellement relevé de
`10·log10(n_az)` (~15,6 dB à pas de 10°) et masque le signal faible des rails lointains.

## Sources AIS

- `synthetic` : snapshot climatologique de la Méditerranée nord-ouest (lignes Toulon–Corse,
  approches portuaires), reproductible (seed) — défaut hors-ligne.
- `--ais-file` / `--ais-url` : points AIS réels en CSV (`lon,lat[,type,speed]`) ou GeoJSON,
  typiquement issus de statistiques publiques en ligne (EMODnet Human Activities, MarineTraffic,
  marinecadastre...). Le catalogue est filtré sur la bbox (marge 0,5°).
- `--ais-density SPEC` (**vessel density**, ex. EMODnet) : raster de densité AIS en
  h/km²/mois. `SPEC` = `emodnet` (téléchargement WCS `vesseldensity_allavg` sur la bbox ;
  **repli automatique sur un raster de test synthétique** sauvegardé dans le répertoire de
  sortie si le service est indisponible), une URL (`.tif/.asc/.nc/.zip`), ou un fichier
  local. Échantillonnage : `navires moyens/cellule = densité × aire_cellule(km²) / 720`
  (720 h/mois), puis tirage de Poisson par cellule, positions uniformes dans la cellule,
  type selon `--ais-density-type TYPE=SPEC` (couches EMODnet par type) ou mix par défaut.
  Le raster effectivement utilisé est sauvé (`density_used.tif`) et **affiché tel quel**
  (`density_used.png` à chaque run, `randi-gen density-plot raster.tif`, carte « données
  brutes » en haut du viewer HTML via `html --density raster.tif`, et dans l'interface
  Streamlit à côté des .nc).

## Chaîne complète : du vessel density brut au niveau de bruit directif NL(lat, lon, az, f)

1. **Raster brut EMODnet** (`vesseldensity_*`) : `D(x)` en **heures/km²/mois**, mailles ~1 km
   (EPSG:3857 natif, reprojeté en EPSG:4326 en mémoire). C'est la donnée affichée « brute ».
2. **Densité → navires** : la densité en heures par mois divisée par le nombre d'heures du
   mois donne le **nombre moyen de navires présents** par km² : `λ_cellule = D(x) · A_km² / H`
   avec `H = 720 h/mois` (option `--density-hours-per-month`) et `A` = aire de la cellule.
   Tirage de **Poisson(λ)** par cellule → `N` navires, positions **uniformes dans la
   cellule**, type de navire : couche EMODnet **dédiée par type** via `--ais-density-type
   TYPE=emodnet` (couche `vesseldensity_XXavg`, codes EMODnet 00–12, type normalisé vers les
   5 classes du modèle), sinon mix cargo 35 % / tanker 15 % / passagers 20 % / pêche 20 % /
   autre 10 % ; vitesse `~ N(μ_type, 1,5)` bornée [2, 25] nœuds. En mode par type, le raster
   combiné (somme des couches) est aussi sauvé comme `density_used.tif`.
   On obtient un **ShipCatalog (lat, lon, type, vitesse)** — strictement identique à celui
   du mode « points AIS », donc la suite du modèle est la même.
3. **Navire → source ponctuelle** : `SL(f, type, v)` (formule ci-dessus), en dB re 1 µPa @ 1 m —
   **chaque type a son propre `SL_100Hz`** (tanker 185, cargo 182, passager 178, pêche 168,
   autre 165 dB) : la sommation AIS tient donc compte du type de chaque navire.
4. **Sources → champ directif** : pour chaque cellule réceptrice (lat, lon) et chaque secteur
   d'azimut `az` (pas de 10°), somme incohérente des navires pondérés par l'ouverture gaussienne
   `w(az, cap)` (FWHM = `azimuth_aperture_deg` = pas par défaut, poids normalisés par navire) et
   atténués par la perte de transmission :
   `NL(lat, lon, az, f) = 10·log10( Σ w(az,cap) 10^((SL − TL(r,f))/10) + 10^(P_fond/10) )`,
   `r` = distance haversine navire→récepteur, `TL(r,f) = 15·log10(r_m) + α_Thorp(f)·r_km`.
5. **Sortie** : NetCDF 3D `noise_level(lat, lon, azimuth)` + `omnidirectional_level` +
   `directionality_index` (2D).

## Hypothèses et limites

- Propagation en **eau profonde à fond plat** : pas de bathymétrie, réfraction, réflexion fond/
  surface, ni ombre d'îles — la Corse n'occulte pas le champ (le modèle ne gère pas les masques
  terrestres ; on peut les imposer a posteriori en masquant les cellules à terre).
- Mer calme / état de mer fixe : le fond diffus `P_fond` est un paramètre, pas un modèle de vent.
- Snapshot AIS (instantané ou climatologique) : pas de variabilité temporelle modélisée.
- Somme incohérente en énergie ; pas d'interférences ni de cohérence de phase.
- Portée des navires limitée à `max_range_km` (300 km par défaut).
