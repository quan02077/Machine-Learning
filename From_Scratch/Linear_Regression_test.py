from Linear_Regression import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regressor = LinearRegression(lr=0.01, n_iters=1000)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_value = mse(y_test, predicted)
print(mse_value)

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, predicted, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()