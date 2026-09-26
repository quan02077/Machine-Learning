import numpy as np


# 1. HÀM TÍNH ENTROPY
def entropy(y):
  p = np.mean(y == 1)
  if p == 0 or p == 1:
    return 0
  return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# 2. HÀM TÌM CÂU HỎI TỐT NHẤT
def find_best_split(X, y):
  best_gain = -1
  best_col, best_thresh = None, None

  for col in range(X.shape[1]):
    thresholds = np.unique(X[:, col])

    for thresh in thresholds:
      left_mask = X[:, col] <= thresh
      right_mask = ~left_mask

      if sum(left_mask) == 0 or sum(right_mask) == 0:
        continue

      gain = entropy(y) - (
          (sum(left_mask) / len(y)) * entropy(y[left_mask])
          + (sum(right_mask) / len(y)) * entropy(y[right_mask])
      )

      if gain > best_gain:
        best_gain = gain
        best_col = col
        best_thresh = thresh

  return best_col, best_thresh


# 3. HÀM TRỒNG CÂY (BUILD TREE)
def build_tree(X, y, depth=0, max_depth=3):
  if len(np.unique(y)) == 1 or depth == max_depth:
    return int(np.mean(y) >= 0.5)

  col, thresh = find_best_split(X, y)
  if col is None:
    return int(np.mean(y) >= 0.5)

  left_mask = X[:, col] <= thresh
  return {
      'col': col,
      'thresh': thresh,
      'left': build_tree(X[left_mask], y[left_mask], depth + 1, max_depth),
      'right': build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth),
  }


# 4. DỰ ĐOÁN CHO 1 MẪU
def predict_one(tree, x):
  if not isinstance(tree, dict):
    return tree

  if x[tree['col']] <= tree['thresh']:
    return predict_one(tree['left'], x)
  else:
    return predict_one(tree['right'], x)


# 5. DỰ ĐOÁN CHO TẬP DỮ LIỆU
def predict_tree(tree, X):
  return np.array([predict_one(tree, x) for x in X])