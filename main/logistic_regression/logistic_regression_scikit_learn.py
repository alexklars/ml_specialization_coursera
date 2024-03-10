# Lab: Logistic Regression using Scikit-Learn
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
#
# X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5], [10, 20.5], [11, 32.5]])
# y = np.array([0, 0, 0, 1, 1, 1, 2, 2])

# Create and fit the logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X, y)
print(f"number of iterations completed: {lr_model.n_iter_}")

# View parameters
# Note, the parameters are associated with the normalized input data.
b_norm = lr_model.intercept_
w_norm = lr_model.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

# Making predictions
y_pred = lr_model.predict(X)
print("Prediction on training set:", y_pred)
