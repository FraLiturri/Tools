import os
import numpy as np
from src.bootstrap import Bootstrap
from src.autocorrelation import Autocorrelation

filepath = "C:/Users/franc/Downloads/Q6.txt"
data = np.loadtxt(filepath)

bootstrap = Bootstrap(data, blocks=100, boot_samples=10**3, function=lambda x: x**2)
mean_f, stddev = bootstrap()

autocorr = Autocorrelation(data, max_lag=6, function=lambda x: x)
tau_int = autocorr()

print(f"\nEstimated Mean: {mean_f} \nEstimated StdDev: {stddev}")
print(f"tau_int: {tau_int}\n")
