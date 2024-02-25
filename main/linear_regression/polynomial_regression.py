# Lab: Feature Engineering and Polynomial Regression
import matplotlib.pyplot as plt
import numpy as np

from common.lab_utils_multi import run_gradient_descent_feng, zscore_normalize_features

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

#
# 1 Linear regression without feature engineering
# This first example shows the limitations of linear regression without feature engineering
#

# create target data
x = np.arange(0, 20, 1)
y = 1 + x ** 2
X = x.reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("no feature engineering")
plt.plot(x, np.dot(X, model_w) + model_b, label="Predicted Value")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

#
# 2 Polynomial regression with feature engineered
#
# To fit our not linear data we can add engineered features (artificial x’s).
# In example below we assume that x^2 suits well and choose it.
#

# create target data
x = np.arange(0, 20, 1)
y = 1 + x ** 2
X = x ** 2  # <-- added engineered feature
X = X.reshape(-1, 1)  # X should be a 2-D Matrix

model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("Added x**2 feature")
plt.plot(x, np.dot(X, model_w) + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#
# 3 Polynomial regression with feature engineered
#
# To fit our not linear data we can add engineered features (artificial x’s).
# We don’t know in advance which artificial x’s to add. So we can add different ones. The gradient descent will find
# a very small coefficient w for those x that are not suitable, so they will have little influence on the final model.
# And vice versa, for x's that are well suited, it will find big coefficient w.
#
# NOTE: adding artificial x’s like x^2, x^3 we make the spread between the input data very large, and in order for the
# gradient descent to work effectively, we first need to scale this data.
#

# create target data
x = np.arange(0, 20, 1)
y = x ** (1 / 2)
X = np.c_[x, x ** (1 / 2), x ** 2, x ** 3]  # <-- added engineered feature
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X, axis=0)}")

# add mean_normalization
X = zscore_normalize_features(X)
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X, axis=0)}")

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=2e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("Normalized x, x**(1/2), x**2, x**3 feature")
plt.plot(x, np.dot(X, model_w) + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#
# 4 Polynomial regression with feature engineered
#
# With feature engineering, even quite complex functions can be modeled
#

x = np.arange(0, 20, 1)
y = np.cos(x / 2)

X = np.c_[x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9, x ** 10, x ** 11, x ** 12, x ** 13]
X = zscore_normalize_features(X)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("Normalized x x**2, x**3 ... x**13 feature")
plt.plot(x, X @ model_w + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
