from typing import Callable, Iterable, Self

import numpy as np
import nptyping as npt

__all__ = ["Callable", "Iterable", "Self", "Generator", "Scalar", "NSequence", "ScalarLike", "Point", "Points"]

Generator = np.random.Generator

Scalar = int | float | npt.Int32 | npt.Float64
NSequence = npt.NDArray[npt.Shape["* n"], npt.Float64]
ScalarLike = Scalar | NSequence

Point = complex | npt.Complex128
Points = npt.NDArray[npt.Shape["*"], npt.Complex128]
PointLike = Point | Points
