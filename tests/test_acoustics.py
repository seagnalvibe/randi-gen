import numpy as np
import pytest

from randi_gen.acoustics import (
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
