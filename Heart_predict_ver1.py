# đây là bản chính thức
import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle


# IMPORT DATA FROM THE DATA SHEET
data = pd.read_csv('framingham.csv')
# data[4238, 16]
data = data.dropna()

y = data.pop('TenYearCHD').to_numpy().reshape(-1, 1)
x = data.to_numpy()
sm = SMOTE(sampling_strategy = 0.25, random_state=42)
X, Y = sm.fit_resample(x, y)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y ,  random_state=10, test_size=0.25, shuffle=True)
r, c = X_train.shape
# X_train[2742, 15]  X_test[914, 15]
def sigmoid(x) :
    return (1 / (1 + np.exp(-x)))
class Logistic:
    def __init__(self, r, c, base):
        self.row = r
        self.col = c
        self.base = base
    def train_data(self, x, y):
        self.x_train = np.append(x, np.ones((self.row, 1)), axis = 1)
        self.y_train = np.array(y).reshape(-1, 1)
        self.w = np.zeros((self.col + 1, 1))
    def predict(self):
        return sigmoid(self.x_train @ self.w)
    def loss(self):
        l = np.log(self.predict()).T @ self.y_train + (np.ones(self.row, 1) - self.y_train).T @ np.log(np.ones(self.row, 1) - self.predict())
    def gradient_descent(self, lr, iteration):
        for iter in range(10):
          #print("Epoch: ", iter)
          for _ in range(round(iteration /10)) :
            self.w -= lr * np.dot(self.x_train.T, (self.predict() - self.y_train)) / self.row
          #print(self.accuracy())
        return self.w
    def classify(self):
        pred = np.zeros((self.row, 1))
        for _ in range(self.row):
            if self.predict()[_] >= self.base:
                pred[_] = 1
        return pred
    def accuracy(self):
        Acc = metrics.accuracy_score(self.y_train, self.classify())
        return Acc
    def confusion_matrix(self):
        confusion_matrix = metrics.confusion_matrix(self.y_train, self.classify())
        print(confusion_matrix)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
        cm_display.plot()
        plt.show()
    def graph(self):
        plt.scatter(self.x_train[:,0], self.y_train)


def decision(x, t):
  if x < t:
    return 0
  return 1

threshold = 0.3
logistic = Logistic(r, c, threshold)
logistic.train_data(X_train, Y_train)
weight = logistic.gradient_descent(0.000005, 10000)
print(logistic.accuracy())
logistic.confusion_matrix()
padding = ' ' * 25
for _ in range(c - 1):
    print('{:.20s} {}'.format(data.keys()[_] + padding, weight[_]))

input_data = X_test[1] # ma trận 15 phần tử
input_data = np.append(input_data, 1)
input = input.reshape(1, -1)
print(decision(sigmoid(input_data @ weight), threshold))

filename = 'HEART_DISEASE.sav'
pickle.dump(logistic,open(filename,'wb'))