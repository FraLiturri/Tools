import os
import numpy as np
from src.bootstrap import Bootstrap
from src.autocorrelation import Autocorrelation

filepath = "C:/Users/franc/Downloads/Q6.txt"
#filepath = r"C:\Users\franc\OneDrive\Desktop\Github Projects\Tools\data\test_data.txt"
#filepath = r"C:\Users\franc\OneDrive\Desktop\iCloudDrive\Master Thesis\Code\Q_tests\NumericalMethods\Q1_hmc.txt"
data = np.loadtxt(filepath, max_rows= 1000)

bootstrap = Bootstrap(data, blocks=100, boot_samples=10**3, function=lambda x: x**2)
mean_f, stddev = bootstrap()

autocorr = Autocorrelation(data, max_lag=5, function=lambda x: x)
tau_int = autocorr()

print(f"\nEstimated Mean: {mean_f} \nEstimated StdDev: {stddev} \n")
print(f"Integrated Autocorrelation Time: {tau_int}\n")
