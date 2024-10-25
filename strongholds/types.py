from typing import Iterable

import numpy as np
import nptyping as npt

__all__ = ["Generator", "Scalar", "NVector", "ScalarLike",
           "RectPoint", "PolarPoint", "RectCoordinates", "PolarCoordinates"]

Generator = np.random.Generator

Scalar = float | npt.Float
NVector = npt.NDArray[npt.Shape["* n"], npt.Float]
ScalarLike = Scalar | NVector

RectPoint = npt.NDArray[npt.Shape["[x, z]"], npt.Float]
PolarPoint = npt.NDArray[npt.Shape["[r, phi]"], npt.Float]

RectCoordinates = RectPoint | npt.NDArray[npt.Shape["*, [x, z]"], npt.Float]
PolarCoordinates = PolarPoint | npt.NDArray[npt.Shape["*, [r, phi]"], npt.Float]
