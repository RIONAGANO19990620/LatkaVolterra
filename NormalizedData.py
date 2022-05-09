import dataclasses

import numpy as np


@dataclasses.dataclass
class NormalizedData:
    array: np.ndarray

    @property
    def max(self) -> float:
        return self.array.max()

    @property
    def min(self) -> float:
        return self.array.min()

    @property
    def data(self) -> np.ndarray:
        return (self.array - self.min) / (self.max - self.min)


@dataclasses.dataclass
class NormalizedDataList:
    u_array: np.ndarray
    v_array: np.ndarray
    t_array: np.ndarray

    @property
    def u_normalized(self) -> NormalizedData:
        return NormalizedData(self.u_array)

    @property
    def v_normalized(self) -> NormalizedData:
        return NormalizedData(self.v_array)

    @property
    def t_normalized(self) -> NormalizedData:
        return NormalizedData(self.t_array)
