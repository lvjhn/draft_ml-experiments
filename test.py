import numpy as np
from scipy import stats
import timeit

# Example vector for profiling
vector = np.random.rand(10000, 300)  # Adjust size as needed

# Profiling each function
i = 0  # Index for column
print("Profiling results:")
print("Min:", timeit.timeit(lambda: np.min(vector[:, i]), number=100))
print("Max:", timeit.timeit(lambda: np.max(vector[:, i]), number=100))
print("Median:", timeit.timeit(lambda: np.median(vector[:, i]), number=100))
print("Std Dev:", timeit.timeit(lambda: np.std(vector[:, i]), number=100))
print("Variance:", timeit.timeit(lambda: np.var(vector[:, i]), number=100))
print("CV:", timeit.timeit(lambda: stats.variation(vector[:, i]), number=100))
print("25th Percentile:", timeit.timeit(lambda: np.percentile(vector[:, i], 25), number=100))
print("50th Percentile:", timeit.timeit(lambda: np.percentile(vector[:, i], 50), number=100))
print("Kurtosis:", timeit.timeit(lambda: stats.kurtosis(vector[:, i]), number=100))
print("Skewness:", timeit.timeit(lambda: stats.skew(vector[:, i]), number=100))
