import numpy as np
import pytest

from randi_gen.acoustics import background_level_db, source_level_db, transmission_loss_db
from randi_gen.ais import ShipCatalog
from randi_gen.geo import haversine_km
from randi_gen.model import ModelParams, compute_directional_noise, make_grid

BBOX = (41.8, 5.5, 43.4, 9.6)


def _ship(lat, lon, kind="cargo", speed=15.0):
    return ShipCatalog(lat=np.array([lat]), lon=np.array([lon]),
                       kind=np.array([kind]), speed_kn=np.array([speed]), source="test")


def test_make_grid():
    lats, lons = make_grid(BBOX, 0.1)
    assert len(lats) == 17  # 41.8 .. 43.4
    assert len(lons) == 42  # 5.5 .. 9.6
    assert lats[0] == pytest.approx(41.8)
    assert lats[-1] == pytest.approx(43.4)
    with pytest.raises(ValueError):
        make_grid((43.4, 9.6, 41.8, 5.5), 0.1)


def test_bad_n_azimuth():
    ships = _ship(43.0, 7.0)
    with pytest.raises(ValueError):
        compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(n_azimuth=0))


def test_single_ship_points_east():
    # récepteur en (43.0, 6.0), navire à l'est en (43.0, 7.0) -> azimut ~90°
    ships = _ship(43.0, 7.0, kind="cargo", speed=15.0)
    field = compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(n_azimuth=36))
    assert field.noise_level.shape == (1, 1, 36)
    az_bin = int(np.argmax(field.noise_level[0, 0]))
    assert field.azimuth[az_bin] == pytest.approx(85.0)  # cap ~89,7° -> pic sur le secteur centré sur 85°

    r_m = float(haversine_km(43.0, 6.0, 43.0, 7.0)) * 1000.0
    sl = float(source_level_db(1000.0, "cargo", 15.0))
    tl = float(transmission_loss_db(r_m, 1000.0, spreading=1.5, min_range_m=500.0))
    bg_db = float(background_level_db(1000.0))
    bg = 10.0 ** (bg_db / 10.0)
    expected_omni = 10.0 * np.log10(10.0 ** ((sl - tl) / 10.0) + bg)

    # ouverture gaussienne : l'énergie du navire est conservée (omni inchangé) mais étalée sur
    # les secteurs voisins -> le pic est inférieur au niveau "tout dans un secteur"
    assert field.omnidirectional[0, 0] == pytest.approx(expected_omni, abs=1e-6)
    assert expected_omni - 4.0 < field.noise_level[0, 0, az_bin] < expected_omni
    east_nb = (az_bin + 1) % 36
    assert field.noise_level[0, 0, east_nb] > bg_db + 10.0

    # le secteur opposé (ouest) ne contient que le fond
    west_bin = (az_bin + 18) % 36
    assert field.noise_level[0, 0, west_bin] == pytest.approx(bg_db, abs=1e-6)


def test_gaussian_aperture_profile():
    # navire plein nord du récepteur (cap = 0°) : profil gaussien symétrique de FWHM = 360/n_azimuth
    ships = _ship(44.0, 6.0)
    step = 10.0
    field = compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(n_azimuth=36))
    assert field.meta["n_azimuth"] == 36
    bg_db = float(background_level_db(1000.0))
    nl = field.noise_level[0, 0]
    i5 = int(np.argmin(np.abs(field.azimuth - 5.0)))
    i355 = int(np.argmin(np.abs(field.azimuth - 355.0)))
    i15 = int(np.argmin(np.abs(field.azimuth - 15.0)))
    assert int(np.argmax(nl)) in (i5, i355)
    assert nl[i5] == pytest.approx(nl[i355], abs=1e-6)  # symétrie autour du cap 0°
    sigma = step / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    w = lambda d: np.exp(-0.5 * (d / sigma) ** 2)
    ratio_db = 10.0 * np.log10(w(5.0) / w(15.0))
    e = lambda db: 10.0 ** (db / 10.0) - 10.0 ** (bg_db / 10.0)
    assert 10.0 * np.log10(e(nl[i5]) / e(nl[i15])) == pytest.approx(ratio_db, abs=0.05)
    assert field.meta["azimuth_aperture_deg"] == pytest.approx(step)
    assert field.meta["azimuth_sigma_deg"] == pytest.approx(sigma)


def test_aperture_override_and_validation():
    ships = _ship(44.0, 6.0)
    narrow = compute_directional_noise([43.0], [6.0], ships, 1000.0,
                                       ModelParams(n_azimuth=36, azimuth_aperture_deg=10.0))
    wide = compute_directional_noise([43.0], [6.0], ships, 1000.0,
                                     ModelParams(n_azimuth=36, azimuth_aperture_deg=40.0))
    i15 = int(np.argmin(np.abs(wide.azimuth - 15.0)))
    # ouverture plus large -> plus d'énergie dans les secteurs éloignés du cap, omni identique
    assert wide.noise_level[0, 0, i15] > narrow.noise_level[0, 0, i15] + 5.0
    assert wide.omnidirectional[0, 0] == pytest.approx(narrow.omnidirectional[0, 0], abs=1e-6)
    with pytest.raises(ValueError):
        compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(azimuth_aperture_deg=0.0))


def test_omni_and_di():
    ships = _ship(43.0, 7.0)
    field = compute_directional_noise([43.0], [6.0], ships, 1000.0)
    assert field.omnidirectional[0, 0] >= field.noise_level[0, 0].max() - 1e-9
    assert field.directionality_index[0, 0] > 5.0  # forte directivité avec un seul navire
    assert np.isfinite(field.noise_level).all()


def test_two_ships_opposite_sectors():
    ships = ShipCatalog(lat=np.array([43.0, 43.0]), lon=np.array([7.0, 5.0]),
                        kind=np.array(["cargo", "cargo"]), speed_kn=np.array([15.0, 15.0]), source="test")
    field = compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(n_azimuth=36))
    nl = field.noise_level[0, 0]
    east = int(np.argmax(np.where((field.azimuth > 60) & (field.azimuth < 120), nl, -np.inf)))
    west = int(np.argmax(np.where((field.azimuth > 240) & (field.azimuth < 300), nl, -np.inf)))
    assert nl[east] > 80.0 and nl[west] > 80.0
    north = int(np.argmax(np.where((field.azimuth > 330) | (field.azimuth < 30), nl, -np.inf)))
    assert nl[north] < nl[east] - 20.0


def test_noise_decreases_with_range():
    near = compute_directional_noise([43.0], [6.0], _ship(43.0, 6.5), 1000.0)
    far = compute_directional_noise([43.0], [6.0], _ship(43.0, 9.0), 1000.0)
    assert near.omnidirectional[0, 0] > far.omnidirectional[0, 0]


def test_frequency_effect():
    ships = _ship(43.0, 7.0)
    low = compute_directional_noise([43.0], [6.0], ships, 100.0)
    high = compute_directional_noise([43.0], [6.0], ships, 10000.0)
    # SL décroît avec f mais l'absorption croît : à ~80 km, 10 kHz < 100 Hz
    assert high.omnidirectional[0, 0] < low.omnidirectional[0, 0]


def test_source_level_depends_on_kind_in_summation():
    # même navire (position, vitesse), seul le type change : NL doit suivre SL(f, type, v)
    from randi_gen.acoustics import SL_100HZ_DB

    cargo = compute_directional_noise([43.0], [6.0], _ship(43.0, 7.0, kind="cargo", speed=15.0), 1000.0)
    fishing = compute_directional_noise([43.0], [6.0], _ship(43.0, 7.0, kind="fishing", speed=15.0), 1000.0)
    tanker = compute_directional_noise([43.0], [6.0], _ship(43.0, 7.0, kind="tanker", speed=15.0), 1000.0)
    d_cargo_fishing = cargo.omnidirectional[0, 0] - fishing.omnidirectional[0, 0]
    assert d_cargo_fishing == pytest.approx(SL_100HZ_DB["cargo"] - SL_100HZ_DB["fishing"], abs=0.5)
    assert tanker.omnidirectional[0, 0] > cargo.omnidirectional[0, 0]  # tanker plus bruyant que cargo


def test_max_range_cutoff():
    ships = _ship(41.0, 6.0)  # ~220 km au sud du récepteur
    f_in = compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(max_range_km=300.0))
    f_out = compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(max_range_km=50.0))
    assert f_in.omnidirectional[0, 0] > f_out.omnidirectional[0, 0]


# valeurs documentees (README.md, docs/MODEL.md) : SL a 100 Hz en dB re 1 uPa @ 1 m,
# du plus bruyant au moins bruyant
DOCUMENTED_SL_100HZ = {"tanker": 185.0, "cargo": 182.0, "passenger": 178.0, "fishing": 168.0, "other": 165.0}
ALL_KINDS = list(DOCUMENTED_SL_100HZ)


def _sl_doc(kind, speed=15.0, freq=1000.0):
    # SL du type recalcule a partir des valeurs documentees (independant de SL_100HZ_DB)
    return (DOCUMENTED_SL_100HZ[kind] - 17.0 * np.log10(freq / 100.0)
            + float(np.clip(5.0 * np.log10(speed / 12.0), -6.0, 6.0)))


def _expected_nl_1ship(kind, speed=15.0, freq=1000.0):
    r_m = float(haversine_km(43.0, 6.0, 43.0, 7.0)) * 1000.0
    sl = _sl_doc(kind, speed, freq)
    tl = float(transmission_loss_db(r_m, freq, spreading=1.5, min_range_m=500.0))
    bg = 10.0 ** (float(background_level_db(freq)) / 10.0)
    return 10.0 * np.log10(10.0 ** ((sl - tl) / 10.0) + bg)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_noise_level_matches_ship_type_source_level(kind):
    # un navire seul a l'est du recepteur : le NL de son secteur d'azimut doit etre
    # exactement 10*log10(10^((SL(f, type, v) - TL)/10) + fond), avec le SL documente du type
    from randi_gen.acoustics import SL_100HZ_DB

    assert SL_100HZ_DB[kind] == DOCUMENTED_SL_100HZ[kind]  # table du code = doc
    ships = _ship(43.0, 7.0, kind=kind, speed=15.0)
    field = compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(azimuth_step_deg=10.0))
    az_bin = int(np.argmax(field.noise_level[0, 0]))
    sl = float(source_level_db(1000.0, kind, 15.0))
    assert sl == pytest.approx(_sl_doc(kind))
    assert field.noise_level[0, 0, az_bin] == pytest.approx(_expected_nl_1ship(kind), abs=1e-9)


def test_noise_level_ordering_follows_ship_type():
    # meme navire (position, vitesse, frequence) : le NL croît avec le SL documente du type
    from randi_gen.acoustics import SL_100HZ_DB

    assert SL_100HZ_DB == DOCUMENTED_SL_100HZ
    nl = {}
    for kind in ALL_KINDS:
        field = compute_directional_noise([43.0], [6.0], _ship(43.0, 7.0, kind=kind, speed=15.0), 1000.0)
        nl[kind] = field.omnidirectional[0, 0]
    for louder, quieter in zip(ALL_KINDS, ALL_KINDS[1:]):
        assert DOCUMENTED_SL_100HZ[louder] > DOCUMENTED_SL_100HZ[quieter]
        assert nl[louder] > nl[quieter]
    # le navire domine le fond a ~80 km : l'ecart de NL = ecart de la table documentee
    assert nl["cargo"] - nl["fishing"] == pytest.approx(
        DOCUMENTED_SL_100HZ["cargo"] - DOCUMENTED_SL_100HZ["fishing"], abs=0.5)


def test_mixed_ship_types_sum_with_their_own_source_level():
    # deux navires superposes de types differents : sommation incoherente des deux SL documentees
    ships = ShipCatalog(lat=np.array([43.0, 43.0]), lon=np.array([7.0, 7.0]),
                        kind=np.array(["tanker", "fishing"]), speed_kn=np.array([15.0, 15.0]), source="test")
    field = compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(azimuth_step_deg=10.0))
    az_bin = int(np.argmax(field.noise_level[0, 0]))
    r_m = float(haversine_km(43.0, 6.0, 43.0, 7.0)) * 1000.0
    tl = float(transmission_loss_db(r_m, 1000.0, spreading=1.5, min_range_m=500.0))
    bg = 10.0 ** (float(background_level_db(1000.0)) / 10.0)
    energy = sum(10.0 ** ((_sl_doc(k) - tl) / 10.0) for k in ("tanker", "fishing"))
    expected = 10.0 * np.log10(energy + bg)
    assert field.noise_level[0, 0, az_bin] == pytest.approx(expected, abs=1e-9)


def test_ship_type_from_csv_drives_noise_level(tmp_path):
    # chaine complete : colonne ship_type du CSV AIS -> SL du type -> NL calcule
    from randi_gen.ais import load_ais

    csv = tmp_path / "ais.csv"
    csv.write_text("lon,lat,ship_type,sog\n7.0,43.0,tanker,15\n")
    cat = load_ais(file=str(csv))
    assert cat.kind[0] == "tanker"
    field = compute_directional_noise([43.0], [6.0], cat, 1000.0, ModelParams(azimuth_step_deg=10.0))
    az_bin = int(np.argmax(field.noise_level[0, 0]))
    assert field.noise_level[0, 0, az_bin] == pytest.approx(_expected_nl_1ship("tanker", 15.0), abs=1e-9)
