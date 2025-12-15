import numpy as np

class Autocorrelation:
    def __init__(self, data: np.ndarray, *, max_lag: int = 5, therm : int = 0, function: callable = lambda x: x):
        self.data = data[therm:]
        self.data = function(self.data)
        self.n = len(data)
        self.max_lag = max_lag
        self.function = function

    def correlation_t(self, data: np.array, t: int) -> float:
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
