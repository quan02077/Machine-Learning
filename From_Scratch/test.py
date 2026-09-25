import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import 3 mô hình From-Scratch
from Linear_Regression import LinearRegression
from Logistic_Regression import LogisticRegression
from Naive_Bayes import NaiveBayes


# =========================================================
# 1. CÁC HÀM ĐÁNH GIÁ (METRICS) DÙNG CHUNG
# =========================================================
def accuracy(y_true, y_pred):
    """Tính độ chính xác cho bài toán Phân loại (Classification)"""
    return np.sum(y_true == y_pred) / len(y_true)

def mse(y_true, y_pred):
    """Tính sai số bình phương trung bình cho bài toán Hồi quy (Regression)"""
    return np.mean((y_true - y_pred) ** 2)


# =========================================================
# 2. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU CHUNG
# =========================================================
df = pd.read_csv('wine_quality_merged.csv')

# Danh sách các cột đặc trưng hóa học dùng làm X
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'pH', 'sulphates']


# =========================================================
# 3. MÔ HÌNH 1: LINEAR REGRESSION (Dự đoán nồng độ cồn 'alcohol')
# =========================================================
print("--- 1. LINEAR REGRESSION ---")
X_lin = df[features].values
y_lin = df['alcohol'].values  # Nhãn y là số thực liên tục

# Chia tập train/test và scale data
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_lin, y_lin, test_size=0.2, random_state=1234)

scaler_l = StandardScaler()
X_train_l = scaler_l.fit_transform(X_train_l)
X_test_l = scaler_l.transform(X_test_l)

# Huấn luyện & Dự đoán
lin_reg = LinearRegression(lr=0.01, n_iters=1000)
lin_reg.fit(X_train_l, y_train_l)
pred_lin = lin_reg.predict(X_test_l)

print(f"Linear Regression MSE: {mse(y_test_l, pred_lin):.4f}\n")


# =========================================================
# 4. CHUẨN BỊ DỮ LIỆU PHÂN LOẠI (Cho Logistic Regression & Naïve Bayes)
# =========================================================
# Đặc trưng X bao gồm cả cột alcohol
features_cls = features + ['alcohol']
X_cls = df[features_cls].values

# Nhãn y nhị phân: 1 nếu quality >= 6 (Rượu ngon), 0 nếu < 6 (Bình thường)
y_cls = (df['quality'] >= 6).astype(int).values

# Chia tập train/test và scale data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=1234)

scaler_c = StandardScaler()
X_train_c = scaler_c.fit_transform(X_train_c)
X_test_c = scaler_c.transform(X_test_c)


# =========================================================
# 5. MÔ HÌNH 2: LOGISTIC REGRESSION (Phân loại Rượu ngon / Bình thường)
# =========================================================
print("--- 2. LOGISTIC REGRESSION ---")
log_reg = LogisticRegression(lr=0.1, n_iters=1000)
log_reg.fit(X_train_c, y_train_c)
pred_log = log_reg.predict(X_test_c)

print(f"Logistic Regression Accuracy: {accuracy(y_test_c, pred_log) * 100:.2f}%\n")


# =========================================================
# 6. MÔ HÌNH 3: GAUSSIAN NAÏVE BAYES (Phân loại Rượu ngon / Bình thường)
# =========================================================
print("--- 3. GAUSSIAN NAÏVE BAYES ---")
nb = NaiveBayes()
nb.fit(X_train_c, y_train_c)
pred_nb = nb.predict(X_test_c)

print(f"Naive Bayes Accuracy: {accuracy(y_test_c, pred_nb) * 100:.2f}%\n")