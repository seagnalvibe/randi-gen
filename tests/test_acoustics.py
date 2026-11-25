import numpy as np
import pytest

from randi_gen.acoustics import (
    SL_100HZ_DB,
    background_level_db,
    normalize_kind,
    source_level_db,
    thorp_absorption_db_km,
    transmission_loss_db,
)


def test_thorp_values():
    a1k = float(thorp_absorption_db_km(1000.0))
    assert 0.05 < a1k < 0.10  # ~0,07 dB/km à 1 kHz
    a10k = float(thorp_absorption_db_km(10000.0))
    assert 0.8 < a10k < 1.6  # ~1,2 dB/km à 10 kHz
    assert float(thorp_absorption_db_km(100.0)) < a1k < a10k


def test_source_level_spectrum():
    assert float(source_level_db(100.0, "cargo", 12.0)) == 182.0
    assert float(source_level_db(1000.0, "cargo", 12.0)) == 165.0
    assert source_level_db(10000.0, "cargo", 12.0) < source_level_db(100.0, "cargo", 12.0)
    # correction de vitesse bornée
    assert float(source_level_db(1000.0, "cargo", 24.0)) - float(source_level_db(1000.0, "cargo", 12.0)) == pytest.approx(5.0 * np.log10(2.0))
    assert float(source_level_db(1000.0, "cargo", 100.0)) - float(source_level_db(1000.0, "cargo", 12.0)) <= 6.0


def test_normalize_kind():
    assert normalize_kind("Ferry") == "passenger"
    assert normalize_kind("pleasure craft") == "other"
    assert normalize_kind("quelque chose") == "other"
    assert float(source_level_db(1000.0, "ferry", 12.0)) == float(source_level_db(1000.0, "passenger", 12.0))


def test_transmission_loss():
    tl_near = float(transmission_loss_db(500.0, 1000.0, spreading=1.5, min_range_m=500.0))
    assert abs(tl_near - (15.0 * np.log10(500.0) + float(thorp_absorption_db_km(1000.0)) * 0.5)) < 1e-9
    tl_far = float(transmission_loss_db(100_000.0, 1000.0, spreading=1.5, min_range_m=500.0))
    assert tl_far > tl_near
    # l'absorption pénalise les hautes fréquences à grande distance
    assert float(transmission_loss_db(100_000.0, 10000.0)) > float(transmission_loss_db(100_000.0, 100.0))
    # la distance est plafonnée à min_range_m
    assert float(transmission_loss_db(10.0, 1000.0, min_range_m=500.0)) == tl_near


def test_background():
    assert float(background_level_db(100.0)) == 60.0
    assert float(background_level_db(1000.0)) == 48.0
    assert float(background_level_db(10000.0)) < float(background_level_db(100.0))


# valeurs documentees (README.md, docs/MODEL.md) : SL a 100 Hz en dB re 1 uPa @ 1 m
DOCUMENTED_SL_100HZ = {"tanker": 185.0, "cargo": 182.0, "passenger": 178.0, "fishing": 168.0, "other": 165.0}
# du plus bruyant au moins bruyant, comme la table documentee
ALL_KINDS = list(DOCUMENTED_SL_100HZ)


def test_source_level_100hz_documented_values():
    assert SL_100HZ_DB == DOCUMENTED_SL_100HZ  # la table du code est bien la doc
    for kind, sl100 in DOCUMENTED_SL_100HZ.items():
        assert float(source_level_db(100.0, kind, 12.0)) == sl100


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_source_level_100hz_matches_type_table(kind):
    # SL(100 Hz, 12 kn) doit etre exactement la valeur de la table du type
    assert float(source_level_db(100.0, kind, 12.0)) == SL_100HZ_DB[kind]


def test_source_level_ordering_and_gaps_follow_type_table():
    # a vitesse identique, SL(f) ordonne les types comme SL_100Hz et les ecarts
    # entre types sont independants de la frequence (meme pente de 17 dB/decade)
    assert all(SL_100HZ_DB[a] > SL_100HZ_DB[b] for a, b in zip(ALL_KINDS, ALL_KINDS[1:]))
    for f in (100.0, 1000.0, 10000.0):
        sl = {k: float(source_level_db(f, k, 15.0)) for k in ALL_KINDS}
        for louder, quieter in zip(ALL_KINDS, ALL_KINDS[1:]):
            assert sl[louder] > sl[quieter]
        for a in ALL_KINDS:
            for b in ALL_KINDS:
                assert sl[a] - sl[b] == pytest.approx(SL_100HZ_DB[a] - SL_100HZ_DB[b], abs=1e-9)


def test_source_level_aliases_match_canonical_type():
    assert float(source_level_db(1000.0, "ferry", 12.0)) == float(source_level_db(1000.0, "passenger", 12.0))
    for alias in ("pleasure", "sailing", "military", "inconnu"):
        assert float(source_level_db(1000.0, alias, 12.0)) == float(source_level_db(1000.0, "other", 12.0))


def test_source_level_vectorized_over_kinds():
    kinds = np.array(ALL_KINDS)
    sl = source_level_db(1000.0, kinds, 12.0)
    assert sl.shape == kinds.shape
    for k, v in zip(kinds, sl):
        assert float(v) == pytest.approx(SL_100HZ_DB[k] - 17.0)  # 1 kHz = 100 Hz - 17 dB
