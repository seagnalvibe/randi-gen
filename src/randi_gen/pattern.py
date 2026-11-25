"""Directivité tabulée par azimut, fournie sous forme de fichier matriciel .mat.

Format du fichier (variables MATLAB ; noms acceptés quelle que soit la casse) :

- `directivity` (n_az, n_ang) : poids **linéaires** (>= 0) de la directivité. Pour une
  directivité exprimée en dB, stocker `10**(dB/10)`. Un vecteur 1D (n_ang,) est diffusé
  identique sur tous les azimuts.
- `azimuth` (n_az,) : azimuts de la carte (degrés, 0 = Nord, sens horaire). Le **nombre
  d'azimuts** de la carte est extrait de ce vecteur ; si absent, grille uniforme de
  `n_az` secteurs (n_az = lignes de la matrice, ou option --n-azimuth pour un vecteur 1D).
- `angle` (n_ang,) : **échantillonnage angulaire** (degrés, 0 = Nord, horaire) = cap sous
  lequel le point récepteur voit la source. Si absent, uniforme sur [0, 360[ avec
  n_ang = colonnes de la matrice.

Pour un navire vu du récepteur sous le cap `brg`, le poids appliqué au secteur d'azimut
`az_k` est la directivité `directivity[k, :]` **interpolée linéairement** (périodiquement
sur 360°) au point `brg`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DIRECTIVITY_NAMES = ("directivity", "directivite", "pattern", "weights", "di", "d")
_AZIMUTH_NAMES = ("azimuth", "azimuths", "azimuth_deg", "az", "az_deg")
_ANGLE_NAMES = ("angle", "angles", "angle_deg", "bearing", "bearing_deg", "cap", "theta")


@dataclass
class AzimuthPattern:
    """Directivité tabulée : values[k, j] = poids du secteur azimuth[k] pour un cap = angle[j]."""

    azimuth: np.ndarray  # (n_az,) degrés, trié dans [0, 360)
    angle: np.ndarray    # (n_ang,) degrés, trié dans [0, 360)
    values: np.ndarray   # (n_az, n_ang) poids linéaires
    source: str = ""

    def __post_init__(self):
        self.azimuth = np.sort(np.mod(np.asarray(self.azimuth, float), 360.0))
        self.angle = np.unique(np.mod(np.asarray(self.angle, float), 360.0))
        self.values = np.asarray(self.values, float)
        if self.values.ndim == 1:
            self.values = np.broadcast_to(self.values, (self.azimuth.size, self.angle.size))
        if self.values.shape != (self.azimuth.size, self.angle.size):
            raise ValueError(
                f"directivité de forme {self.values.shape} incompatible avec "
                f"azimuth ({self.azimuth.size},) et angle ({self.angle.size},)"
            )
        self._angle_ext = np.concatenate([self.angle, self.angle[:1] + 360.0])
        self._values_ext = np.concatenate([self.values, self.values[:, :1]], axis=1)

    @property
    def n_azimuth(self):
        return self.azimuth.size

    def weight(self, k, brg):
        """Poids du secteur k pour un cap `brg` (array) : interpolation linéaire périodique sur `angle`."""
        brg = np.asarray(brg, float)
        dm = np.mod(brg - self.angle[0], 360.0) + self.angle[0]
        return np.interp(dm, self._angle_ext, self._values_ext[k])


def _load_mat_vars(path):
    """Variables numériques d'un fichier .mat (scipy, repli h5py pour le format v7.3/HDF5)."""
    path = str(path)
    try:
        from scipy.io import loadmat

        try:
            return {k: v for k, v in loadmat(path, squeeze_me=True).items() if not k.startswith("__")}
        except NotImplementedError:
            pass  # .mat v7.3 = HDF5, non lu par scipy
    except ImportError:
        pass
    try:
        import h5py
    except ImportError:
        raise RuntimeError(
            f"lecture de {path} : installer scipy (pip install scipy) pour les .mat v5/v7, "
            "ou h5py pour les .mat v7.3 (HDF5)"
        )
    out = {}
    with h5py.File(path, "r") as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name.split("/")[-1]] = np.asarray(obj)

        f.visititems(_visit)
    return out


def _pick(vars, names):
    lowered = {k.lower(): v for k, v in vars.items()}
    for n in names:
        if n in lowered:
            return np.asarray(lowered[n], float)
    return None


def load_pattern(path, n_azimuth_hint=None) -> AzimuthPattern:
    """Charge un motif de directivité depuis un fichier .mat.

    Le nombre d'azimuts et l'échantillonnage angulaire sont extraits du fichier
    (variables `azimuth` et `angle`) ; à défaut, grilles uniformes déduites de la forme
    de la matrice `directivity` (n_azimuth_hint requis si `directivity` est 1D sans `azimuth`).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fichier de directivité introuvable : {path}")
    variables = _load_mat_vars(path)
    values = _pick(variables, _DIRECTIVITY_NAMES)
    if values is None:
        raise ValueError(
            f"{path} : aucune variable de directivité trouvée "
            f"(noms acceptés : {', '.join(_DIRECTIVITY_NAMES)})"
        )
    azimuth = _pick(variables, _AZIMUTH_NAMES)
    angle = _pick(variables, _ANGLE_NAMES)
    if values.ndim == 2 and angle is not None and values.shape[0] == angle.size and values.shape[1] != angle.size:
        values = values.T  # .mat stocké en colonne (h5py) : l'axe angle est en ligne
    if values.ndim == 1:
        values = values.ravel()
        n_ang = values.size
        if angle is not None and angle.size != n_ang:
            raise ValueError(f"{path} : angle ({angle.size}) incompatible avec directivity ({n_ang},)")
        if angle is None:
            angle = np.arange(n_ang) * (360.0 / n_ang)
        if azimuth is not None:
            n_az = azimuth.size
        elif n_azimuth_hint is not None:
            n_az = int(n_azimuth_hint)
        else:
            raise ValueError(f"{path} : directivity 1D sans variable azimuth -> préciser n_azimuth")
        azimuth = azimuth if azimuth is not None else (np.arange(n_az) + 0.5) * (360.0 / n_az)
    else:
        n_az, n_ang = values.shape
        if angle is not None and angle.size != n_ang:
            raise ValueError(f"{path} : angle ({angle.size}) incompatible avec directivity {values.shape}")
        if angle is None:
            angle = np.arange(n_ang) * (360.0 / n_ang)
        if azimuth is not None and azimuth.size != n_az:
            raise ValueError(f"{path} : azimuth ({azimuth.size}) incompatible avec directivity {values.shape}")
        if azimuth is None:
            azimuth = (np.arange(n_az) + 0.5) * (360.0 / n_az)
    return AzimuthPattern(azimuth=azimuth.ravel(), angle=angle.ravel(), values=values, source=str(path))
