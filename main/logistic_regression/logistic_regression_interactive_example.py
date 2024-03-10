import matplotlib.pyplot as plt
import numpy as np

from common.plt_quad_logistic import plt_quad_logistic

plt.style.use('./common/deeplearning.mplstyle')

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_range = np.array([-1, 7])
b_range = np.array([1, -14])

# To see a plot, add a breakpoint and run debug
quad = plt_quad_logistic(x_train, y_train, w_range, b_range)
