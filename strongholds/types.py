from typing import Iterable

import numpy as np
import nptyping as npt

__all__ = ["Generator", "Scalar", "NSequence", "ScalarLike", "Point", "Coordinates"]

Generator = np.random.Generator

Scalar = float | npt.Float
NSequence = npt.NDArray[npt.Shape["* n"], npt.Float]
ScalarLike = Scalar | NSequence

Point = npt.NDArray[npt.Shape["*"], npt.Complex]
Coordinates = Point | npt.NDArray[npt.Shape["*, *"], npt.Complex]
