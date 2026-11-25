import numpy as np
import pytest
from scipy.io import savemat

from randi_gen.ais import ShipCatalog
from randi_gen.model import ModelParams, compute_directional_noise
from randi_gen.pattern import AzimuthPattern, load_pattern


def _ship(lat, lon, kind="cargo", speed=15.0):
    return ShipCatalog(lat=np.array([lat]), lon=np.array([lon]),
                       kind=np.array([kind]), speed_kn=np.array([speed]), source="test")


def _gauss(angle, center, sigma):
    d = (np.asarray(angle, float) - center + 180.0) % 360.0 - 180.0
    return np.exp(-0.5 * (d / sigma) ** 2)


def test_load_pattern_extracts_grid_and_sampling(tmp_path):
    azimuth = np.array([10.0, 40.0, 90.0, 200.0, 300.0])  # grille non uniforme
    angle = np.arange(0.0, 360.0, 5.0)
    values = np.stack([_gauss(angle, a, 20.0) for a in azimuth])
    path = tmp_path / "pattern.mat"
    savemat(path, {"azimuth": azimuth, "angle": angle, "directivity": values})
    pat = load_pattern(path)
    assert pat.n_azimuth == 5  # nombre d'azimuts extrait du fichier
    assert np.allclose(pat.azimuth, np.sort(azimuth))
    assert pat.angle.size == 72  # échantillonnage angulaire extrait du fichier
    assert pat.values.shape == (5, 72)
    assert pat.source == str(path)


def test_load_pattern_defaults_from_matrix_shape(tmp_path):
    values = np.ones((4, 36))
    path = tmp_path / "p.mat"
    savemat(path, {"directivity": values})
    pat = load_pattern(path)
    assert pat.n_azimuth == 4
    assert np.allclose(pat.azimuth, (np.arange(4) + 0.5) * 90.0)
    assert np.allclose(pat.angle, np.arange(36) * 10.0)


def test_load_pattern_1d_with_n_azimuth_hint(tmp_path):
    path = tmp_path / "p1d.mat"
    savemat(path, {"directivity": np.linspace(1.0, 0.0, 36)})
    pat = load_pattern(path, n_azimuth_hint=8)
    assert pat.values.shape == (8, 36)  # vecteur 1D diffusé sur les 8 azimuts
    with pytest.raises(ValueError):
        load_pattern(path)  # sans indice : nombre d'azimuts indéterminé


def test_load_pattern_missing_directivity(tmp_path):
    path = tmp_path / "bad.mat"
    savemat(path, {"azimuth": np.arange(4.0)})
    with pytest.raises(ValueError):
        load_pattern(path)
    with pytest.raises(FileNotFoundError):
        load_pattern(tmp_path / "absent.mat")


def test_pattern_periodic_interpolation():
    pat = AzimuthPattern(azimuth=[0.0, 180.0], angle=[0.0, 90.0, 180.0, 270.0],
                         values=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert pat.weight(0, 0.0) == pytest.approx(1.0)
    assert pat.weight(0, 45.0) == pytest.approx(0.5)   # interpolation linéaire entre 0° et 90°
    assert pat.weight(0, 90.0) == pytest.approx(0.0)
    assert pat.weight(0, 315.0) == pytest.approx(0.5)  # périodicité : entre 270° et 360°(=0°)
    assert pat.weight(0, np.array([0.0, 45.0])) == pytest.approx([1.0, 0.5])


def test_model_pattern_matches_tabulated_gaussian(tmp_path):
    # motif .mat = gaussienne tabulée sur la même grille que le mode analytique -> mêmes NL
    naz = 4
    azimuth = (np.arange(naz) + 0.5) * (360.0 / naz)
    angle = np.arange(0.0, 360.0, 2.0)  # échantillonnage fin -> interpolation quasi exacte
    sigma = 90.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    values = np.stack([_gauss(angle, a, sigma) for a in azimuth])
    path = tmp_path / "gauss.mat"
    savemat(path, {"azimuth": azimuth, "angle": angle, "directivity": values})
    ships = ShipCatalog(lat=np.array([43.0, 43.2, 42.9]), lon=np.array([7.0, 6.2, 6.8]),
                        kind=np.array(["cargo", "tanker", "fishing"]),
                        speed_kn=np.array([15.0, 12.0, 8.0]), source="test")
    f_ref = compute_directional_noise([43.0], [6.0], ships, 1000.0,
                                      ModelParams(n_azimuth=naz, azimuth_aperture_deg=90.0))
    f_pat = compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(azimuth_pattern=str(path)))
    assert np.allclose(f_pat.azimuth, azimuth)  # grille extraite du fichier
    assert f_pat.noise_level.shape == (1, 1, naz)
    assert np.allclose(f_pat.noise_level, f_ref.noise_level, atol=0.05)
    assert f_pat.omnidirectional[0, 0] == pytest.approx(f_ref.omnidirectional[0, 0], abs=1e-6)
    assert f_pat.meta["n_azimuth"] == naz
    assert f_pat.meta["azimuth_pattern"] == str(path)
    assert np.isnan(f_pat.meta["azimuth_aperture_deg"])


def test_model_pattern_shape_varies_with_azimuth(tmp_path):
    # secteur 0° étroit (σ=5°), secteur 180° large (σ=60°) : la forme de l'ouverture varie avec l'angle
    angle = np.arange(0.0, 360.0, 5.0)
    values = np.stack([_gauss(angle, 0.0, 5.0), _gauss(angle, 180.0, 60.0)])
    path = tmp_path / "asym.mat"
    savemat(path, {"azimuth": [0.0, 180.0], "angle": angle, "directivity": values})
    ships = _ship(44.0, 6.0)  # navire plein nord : cap = 0°
    f = compute_directional_noise([43.0], [6.0], ships, 1000.0, ModelParams(azimuth_pattern=path))
    assert f.azimuth.size == 2
    assert f.noise_level[0, 0, 0] > f.noise_level[0, 0, 1] + 10.0  # presque tout dans le secteur 0°


def test_model_pattern_interpolated_weight(tmp_path):
    # échantillonnage grossier [0, 90, 180, 270] avec pic en 0° : cap 45° -> poids interpolé 0.5
    angle = np.array([0.0, 90.0, 180.0, 270.0])
    values = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    path = tmp_path / "coarse.mat"
    savemat(path, {"azimuth": [0.0, 180.0], "angle": angle, "directivity": values})
    # navire au cap 45° depuis (43.0, 6.0)
    from randi_gen.acoustics import background_level_db
    from randi_gen.geo import destination_point

    lat, lon = destination_point(43.0, 6.0, 45.0, 50.0)
    f = compute_directional_noise([43.0], [6.0], _ship(lat, lon), 1000.0, ModelParams(azimuth_pattern=path))
    bg_db = float(background_level_db(1000.0))
    e = 10.0 ** (f.noise_level[0, 0] / 10.0) - 10.0 ** (bg_db / 10.0)
    assert 10.0 * np.log10(e[0] / e[1]) == pytest.approx(0.0, abs=1e-6)  # 0.5 / 0.5 après normalisation
    assert f.omnidirectional[0, 0] > f.noise_level[0, 0].max()  # énergie conservée, répartie sur 2 secteurs


def test_cli_az_pattern(tmp_path):
    from randi_gen.cli import main

    angle = np.arange(0.0, 360.0, 10.0)
    azimuth = np.arange(0.0, 360.0, 60.0)
    values = np.stack([_gauss(angle, a, 15.0) for a in azimuth])
    mat = tmp_path / "pat.mat"
    savemat(mat, {"azimuth": azimuth, "angle": angle, "directivity": values})
    out = tmp_path / "out"
    rc = main(["run", "--bbox", "42.8", "5.8", "43.2", "6.4", "--res", "0.2", "--freq", "1000",
               "--az-pattern", str(mat), "--ais", "synthetic", "--seed", "3", "--out", str(out)])
    assert rc == 0
    import xarray as xr

    with xr.open_dataset(out / "randi_map_f1000Hz.nc", engine="netcdf4") as ds:
        assert ds.sizes["azimuth"] == 6  # nombre d'azimuts extrait du fichier
        assert np.allclose(ds["azimuth"].values, azimuth)
        assert int(ds.attrs["n_azimuth"]) == 6
        assert ds.attrs["azimuth_pattern"] == str(mat)
        assert np.isfinite(ds["noise_level"].values).all()
