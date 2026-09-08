import numpy as np

X = np.array([
    [1, 1, 1],
    [1, 2, 0],
    [1, 0, 2]
], dtype=float)

Y = np.array([5, 6, 6], dtype=float)

# Cách 1: Dùng Pseudo-inverse (Ma trận giả nghịch đảo) - Khuyên dùng
Xt = X.T
B = np.linalg.pinv(Xt @ X) @ Xt @ Y
    
print("Bo trong so B (Pinv):", B)

# Cách 2: Dùng trực tiếp lstsq (Chuẩn production)
B_lstsq, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

print("Bo trong so B (Lstsq):", B_lstsq)