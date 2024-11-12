from typing import Callable, Iterable, SupportsFloat as Scalar

import numpy as np
import nptyping as npt

__all__ = ["Generator", "Scalar", "NSequence", "ScalarLike", "Point", "Coordinates"]

Generator = np.random.Generator

NSequence = npt.NDArray[npt.Shape["* n"], npt.Float]
ScalarLike = Scalar | NSequence

Point = complex | npt.Complex
PointProbs = dict[Point, Scalar]

Coordinates = Point | npt.NDArray[npt.Shape["*"], npt.Complex]
CoordinateSets = npt.NDArray[npt.Shape["*, *"], npt.Complex]

PointTuple = npt.NDArray[npt.Shape["[x, z]"], npt.Float]
CoordinateTuples = PointTuple | npt.NDArray[npt.Shape["*, [x, z]"], npt.Float]
