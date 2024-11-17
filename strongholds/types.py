from typing import Callable, Iterable, Self

import numpy as np
import nptyping as npt

__all__ = ["Generator", "Scalar", "NSequence", "ScalarLike", "Point", "Points"]

Generator = np.random.Generator

Scalar = int | float | npt.Int | npt.Float
NSequence = npt.NDArray[npt.Shape["* n"], npt.Float]
ScalarLike = Scalar | NSequence

Point = complex | npt.Complex
Points = npt.NDArray[npt.Shape["*"], npt.Complex]

CoordinateSets = npt.NDArray[npt.Shape["*, *"], npt.Complex]

PointTuple = npt.NDArray[npt.Shape["[x, z]"], npt.Float]
CoordinateTuples = PointTuple | npt.NDArray[npt.Shape["*, [x, z]"], npt.Float]
