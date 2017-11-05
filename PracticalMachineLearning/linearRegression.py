from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(data_point_count, variance, step=2, correlation=False):
    val = 1
    y_local = []
    for i in range(data_point_count):
        y = val + random.randrange(-variance, variance)
        y_local.append(y)
        if correlation:
            if correlation == 'pos':
                val += step
            elif correlation == 'neg':
                val -= step
    x_local = [i for i in range(len(y_local))]
    return np.array(x_local, dtype=np.float64), np.array(y_local, dtype=np.float64)


def best_fit_slot_and_intercept(x_local, y_local):
    m_local = (((mean(x_local) * mean(y_local)) - mean(x_local * y_local)) /
               ((mean(x_local) ** 2) - (mean(x_local ** 2))))
    b_local = mean(y_local) - m_local * mean(x_local)
    return m_local, b_local


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for _ in ys_orig]
    squared_error_regression_line = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regression_line / squared_error_y_mean)


xs, ys = create_dataset(100, 50, 2, correlation='pos')
print(xs)
print(ys)
m, b = best_fit_slot_and_intercept(xs, ys)
print(m, b)
regression_line = [(m * x) + b for x in xs]
print(regression_line)
print(coefficient_of_determination(ys, regression_line))

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
