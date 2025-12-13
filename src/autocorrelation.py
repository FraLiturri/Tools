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

    def correlation_t(self, data: list | np.array, t: int) -> np.array:
        res = 0
        self_corr = 0
        if t < len(data):
            for i in range(0, len(data) - t):
                res += (data[i] - np.mean(data)) * (data[i + t] - np.mean(data))
                self_corr = (data[i] - np.mean(data)) ** 2 / len(data)
            return res / ((self_corr) * (len(data) - t))
        else: 
            return 0

    def compute(self) -> float:
        tau_int = 1
        for i in range(0, len(self.data)):
            c_t = self.correlation_t(self.data, i)
            tau_int += 2 * c_t
            if i > 110:
                break
        return tau_int

    def __call__(self) -> float:
        return self.compute()
