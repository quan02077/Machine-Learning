import matplotlib.pyplot as plt

x = [1, 2]
y = [2, 4]

iterations = 100
w = 0
b = 0
m = len(x)
learning_rate = 0.1

for i in range(iterations):
  #Dự đoán
  y_hat = []

  for xi in x:
    prediction = w * xi + b
    y_hat.append(prediction)

  #gradient
  dw = 0
  db = 0

  for i in range(m):
    error = y_hat[i] - y[i]

    dw += error * x[i]
    db += error

  dw = dw/m
  db = db/m

  #Cập nhật w, b

  new_w = w - learning_rate * dw
  new_b = b - learning_rate * db

print("w =", new_w)
print("b =", new_b)

# from sklearn.linear_model import LinearRegression

# x = [[1], [2]]
# y = [2, 4]

# model = LinearRegression()

# X = model.fit(x,y)

# print(X.predict([[3]]))