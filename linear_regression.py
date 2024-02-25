import numpy as np
from scipy.sparse import data
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split


# linear regression class
class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # function for fiting
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(x, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # function for prediction
    def predict(self, x):
        y_predicted = np.dot(x, self.weights) + self.bias
        return y_predicted


x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)


model = LinearRegression(lr=0.001)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)