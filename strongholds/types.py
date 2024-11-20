from typing import Callable, Iterable, Self

import numpy as np
import nptyping as npt

__all__ = ["Generator", "Scalar", "NSequence", "ScalarLike", "Point", "Points"]

Generator = np.random.Generator

Scalar = int | float | npt.Int32 | npt.Float64
NSequence = npt.NDArray[npt.Shape["* n"], npt.Float64]
ScalarLike = Scalar | NSequence

Point = complex | npt.Complex128
Points = npt.NDArray[npt.Shape["*"], npt.Complex128]
PointLike = Point | Points

CoordinateSets = npt.NDArray[npt.Shape["*, *"], npt.Complex128]

PointTuple = npt.NDArray[npt.Shape["[x, z]"], npt.Float64]
CoordinateTuples = PointTuple | npt.NDArray[npt.Shape["*, [x, z]"], npt.Float64]
