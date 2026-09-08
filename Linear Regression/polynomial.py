import numpy as np

D = np.array([
    [1, 1, 1],
    [1, 2, 4],
    [1, 3, 9]
])

Y = np.array([2, 6, 12])

Dt = D.T 

a = np.linalg.inv(Dt @ D) @ Dt @ Y

rounded_a = np.round(a, 4)
print("Bo trong so a:", rounded_a)