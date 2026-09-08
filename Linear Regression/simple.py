x = [2, 4, 6]
y = [3, 7, 8]

x_tb = 0
y_tb = 0
for i in range(len(x)):
    x_tb += x[i]
    y_tb += y[i]

x_tb /= len(x)
y_tb /= len(y)

var = 0
for i in range(len(x)):
    var += (x[i] - x_tb) ** 2

var /= len(x) - 1

cov = 0
for i in range(len(x)):
    cov += (x[i] - x_tb) * (y[i] - y_tb)

cov /= len(x) - 1

B = cov / var
a = y_tb - B * x_tb

print ("x_tb:", x_tb)
print ("y_tb:", y_tb)

print ("var:", var)
print ("cov:", cov)

print ("a:", a)
print ("B:", B)

print("Phuong trinh: y = ", a, "+", B,"x")