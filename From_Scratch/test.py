from Linear_Regression import LinearRegression
from Logistic_Regression import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Đọc file CSV
df = pd.read_csv('wine_quality_merged.csv')

# LỰA CHỌN 1: LOGISTIC REGRESSION (Phân loại Rượu ngon / Bình thường)

# Chọn các cột chỉ số hóa học làm đặc trưng X
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'pH', 'sulphates', 'alcohol']].values

# Tạo nhãn y: 1 nếu chất lượng quality >= 6 (Ngon), 0 nếu < 6 (Bình thường)
y = (df['quality'] >= 6).astype(int).values

# Chia tập Train / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# CHUẨN HÓA DỮ LIỆU (Cực kỳ quan trọng với dữ liệu thực tế)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hàm tính Accuracy
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Chạy mô hình Logistic Regression
regressor = LogisticRegression(lr=0.1, n_iters=1000) # Khuyên dùng lr = 0.1 cho dữ liệu đã scale
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

print("Wine Quality Classification Accuracy:", accuracy(y_test, predicted))


# =========================================================
# LỰA CHỌN 2: LINEAR REGRESSION (Nếu muốn dự đoán nồng độ cồn 'alcohol')
# =========================================================
"""
# 1. Chọn đặc trưng X và nhãn y (số thực)
X_lin = df[['fixed acidity', 'volatile acidity', 'citric acid', 'pH', 'sulphates']].values
y_lin = df['alcohol'].values

# 2. Chia tập Train / Test
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lin, y_lin, test_size=0.2, random_state=1234)

# 3. Chuẩn hóa X
scaler_l = StandardScaler()
X_train_l = scaler_l.fit_transform(X_train_l)
X_test_l = scaler_l.transform(X_test_l)

# 4. Chạy mô hình Linear Regression
lin_reg = LinearRegression(lr=0.01, n_iters=1000)
lin_reg.fit(X_train_l, y_train_l)
pred_lin = lin_reg.predict(X_test_l)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print("Linear Regression MSE:", mse(y_test_l, pred_lin))
"""