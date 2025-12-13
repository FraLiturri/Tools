from __future__ import annotations
from typing import Callable
import numpy as np


def f(x):
    return x


class Autocorrelation:
    def __init__(self, data: list | np.array, *, max_lag: int, function: Callable = f):
        self.data = function(data)
        self.n = len(data)
        self.max_lag = max_lag
        self.function = function

    def correlation_t(self, data: list | np.array, t: int) -> float:
        if t >= len(data):
            return 0

        mean = np.mean(data)
        variance = np.var(data)
        numerator = np.sum((data[:-t] - mean) * (data[t:] - mean))
        C_t = numerator / (len(data) * variance)

        return C_t

    def compute(self) -> float:
        tau_int = 0.5
        for t in range(1, len(self.data)):
            c_t = self.correlation_t(self.data, t)
            tau_int += c_t
            if t > self.max_lag * tau_int:
                break
        return tau_int

    def __call__(self) -> float:
        return self.compute()
