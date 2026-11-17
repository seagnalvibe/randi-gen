import numpy as np

from randi_gen.geo import bearing_deg, destination_point, haversine_km

TOULON = (43.1244, 5.9299)
AJACCIO = (41.9267, 8.7386)


def test_haversine_toulon_ajaccio():
    d = float(haversine_km(TOULON[0], TOULON[1], AJACCIO[0], AJACCIO[1]))
    assert 230.0 < d < 310.0  # ~266 km à vol d'oiseau


def test_haversine_zero():
    assert float(haversine_km(43.0, 6.0, 43.0, 6.0)) == 0.0


def test_bearing_cardinaux():
    assert abs(float(bearing_deg(43.0, 6.0, 44.0, 6.0))) < 1e-9      # Nord
    assert abs(float(bearing_deg(43.0, 6.0, 43.0, 7.0)) - 90.0) < 0.5  # Est
    assert abs(float(bearing_deg(43.0, 6.0, 42.0, 6.0)) - 180.0) < 1e-9  # Sud
    assert abs(float(bearing_deg(43.0, 6.0, 43.0, 5.0)) - 270.0) < 0.5  # Ouest


def test_bearing_toulon_ajaccio_sud_est():
    b = float(bearing_deg(TOULON[0], TOULON[1], AJACCIO[0], AJACCIO[1]))
    assert 100.0 < b < 140.0


def test_destination_roundtrip():
    la2, lo2 = destination_point(TOULON[0], TOULON[1], 120.0, 100.0)
    assert abs(float(haversine_km(TOULON[0], TOULON[1], la2, lo2)) - 100.0) < 1e-6
    back = float(bearing_deg(la2, lo2, TOULON[0], TOULON[1]))
    # sur un grand cercle, le cap de retour n'est pas exactement cap+180
    assert abs((back - 300.0 + 180.0) % 360.0 - 180.0) < 5.0
