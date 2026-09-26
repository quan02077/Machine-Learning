import numpy as np


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape #Lấy số lượng mẫu và số lượng target
        self._classes = np.unique(y) #Lấy ra các lớp duy nhất trong nhãn y (btw chưa hiểu duy nhất là như nào)
        n_classes = len(self._classes) #Sau khi lấy ra các lớp duy nhất thì đếm số lượng (số lượng ờ đây là số lớp duy nhất đúng ko?)

        # tính mean, var, và prior cho mỗi lớp
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)#Tính trung bình muy (btw ko hiểu tại sao lại lấy n_classes, n_features mà ko phải n_samples, n_features)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)#Tính phương sai delta (thắc mắc như trên)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):#Dùng vòng lặp để lặp qua từng lớp duy nhất
            X_c = X[y == c] #Lấy ra mẫu có target y giống lớp c (btw chưa hiểu giống là giống như thế nào, ntn là giống)
            self._mean[idx, :] = X_c.mean(axis=0) #Tính trung bình của lớp đó đúng ko?
            self._var[idx, :] = X_c.var(axis=0)#Cùng câu hỏi
            self._priors[idx] = X_c.shape[0] / float(n_samples)#Và cùng câu hỏi nữa, tại sao lại lấy shape[0] mà ko phải shape[1] hay 2,3,4,5... (btw chưa hiểu shape là gì)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]  #Dự đoán tại sao dùng vòng lặp?
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        #này là tính theo công thức Bayes đúng ko mà có argmax đúng ko?
        for idx, c in enumerate(self._classes):#Tính cho mỗi lớp đúng ko?
            prior = np.log(self._priors[idx])#prior là xác suất kiểu ví dụ P(Yes) mà tổng cả Yes = 9 và No = 5, tổng = 14 đi thì cái này là P(Yes) là 9/14 đúng ko? (btw chưa hiểu tại sao lại log)
            posterior = np.sum(np.log(self._pdf(idx, x)))#là sao?????
            posterior = prior + posterior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    # tính mật độ xác suất  
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx] #Lại tính mean nữa à
        var = self._var[class_idx] #Cùng thắc mắc trên
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator