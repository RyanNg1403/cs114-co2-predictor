from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import ace_tools_open as ace_tools
from collections import namedtuple
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
import pandas as pd
#%pip install mlcroissant
import mlcroissant as mlc





# Lasso Via Coordinate Descent
class LassoCD:  
    """
    Lasso regression via cyclic coordinate descent.
    """

    def __init__(self, alpha=1.0, tol=1e-4, max_iter=1000):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.loss_history = []  # Track loss values during training

    @staticmethod
    def _soft_threshold(rho, alpha):
        if rho > alpha:
            return rho - alpha
        elif rho < -alpha:
            return rho + alpha
        else:
            return 0.0

    def _compute_loss(self, X, y, w):
        """Compute the loss (MSE + L1 penalty)"""
        y_pred = X.dot(w)
        mse = np.mean((y - y_pred) ** 2)
        l1_penalty = self.alpha * np.sum(np.abs(w))
        return mse + l1_penalty

    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Store means and stds
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0, ddof=1)
        self.y_mean = y.mean()

        # Standardize X and center y
        Xs = (X - self.X_mean) / np.where(self.X_std == 0, 1.0, self.X_std)
        ys = y - self.y_mean

        n_samples, n_features = Xs.shape
        w = np.zeros(n_features)

        # Precompute column norms
        col_norms = np.sum(Xs ** 2, axis=0)

        # Track loss every 500 iterations
        track_interval = 500
        self.loss_history = []

        for itr in range(self.max_iter):
            w_old = w.copy()

            for j in range(n_features):
                # Partial residual excluding feature j
                residual = ys - Xs.dot(w) + Xs[:, j] * w[j]
                rho = np.dot(Xs[:, j], residual) / n_samples
                w[j] = self._soft_threshold(rho, self.alpha) / (col_norms[j] / n_samples)

            # Track loss
            if (itr + 1) % track_interval == 0:
                loss = self._compute_loss(Xs, ys, w)
                self.loss_history.append((itr + 1, loss))

            # Check convergence
            if np.max(np.abs(w - w_old)) < self.tol:
                break

        # Store coefficients in original scale
        self.coef_ = w / np.where(self.X_std == 0, 1.0, self.X_std)
        self.intercept_ = self.y_mean - self.X_mean.dot(self.coef_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_

# Lasso via Gradient Descent
class LassoGD:
    def __init__(self, alpha=1.0, lr=0.01, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        # Standardize X and center y
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0, ddof=1)
        self.y_mean = y.mean()

        Xs = (X - self.X_mean) / self.X_std
        ys = y - self.y_mean

        w = np.zeros(n_features)

        for _ in range(self.max_iter):
            y_pred = Xs @ w
            error = y_pred - ys

            grad = (Xs.T @ error) / n_samples  # gradient of MSE part

            # Subgradient for L1: sign(w)
            subgrad = self.alpha * np.sign(w)

            # Full gradient with subgradient
            update = grad + subgrad

            w_new = w - self.lr * update

            # Convergence check
            if np.max(np.abs(w_new - w)) < self.tol:
                break

            w = w_new

        self.coef_ = w / self.X_std
        self.intercept_ = self.y_mean - np.dot(self.X_mean, self.coef_)

        return self
    
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_
    
# Lasso via Proximal Gradient Descent

class LassoPGD:
    """
    Lasso regression using Proximal Gradient Descent (PGD).
    """
    def __init__(self, alpha=1.0, lr=0.01, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def _soft_threshold(self, x, thresh):
        """
        Soft-thresholding operator for proximal step.
        """
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

    def fit(self, X, y):
        # Convert inputs to arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        # Standardize X and center y
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0, ddof=1)
        self.y_mean = y.mean()
        Xs = (X - self.X_mean) / self.X_std
        ys = y - self.y_mean

        # Initialize weights
        w = np.zeros(n_features)

        for iteration in range(self.max_iter):
            # Gradient of the smooth (MSE) part
            y_pred = Xs @ w
            error = y_pred - ys
            grad = (Xs.T @ error) / n_samples

            # Gradient descent step
            w_temp = w - self.lr * grad

            # Proximal (soft-thresholding) step
            w_new = self._soft_threshold(w_temp, self.lr * self.alpha)

            # Check convergence
            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                break

            w = w_new

        # Rescale coefficients to original feature scale
        self.coef_ = w / self.X_std
        self.intercept_ = self.y_mean - np.dot(self.X_mean, self.coef_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_




# Decision Tree
Node = namedtuple("Node", ["feature_index", "threshold", "left", "right", "value", "n_samples", "impurity"])

class ScratchDecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.n_features_ = None
        self._feature_importances = None  # To store raw importance scores
        self.total_samples_ = None  # To store total number of samples

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.total_samples_ = len(y)
        self._feature_importances = np.zeros(self.n_features_)  # Initialize importance array
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - y.mean()) ** 2)

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_idx, best_thr = None, None
        parent_mse = self._mse(y)

        for idx in range(self.n_features_):
            sorted_idx = np.argsort(X[:, idx])
            X_sorted = X[sorted_idx, idx]
            y_sorted = y[sorted_idx]

            for i in range(1, len(X_sorted)):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                thr = (X_sorted[i] + X_sorted[i - 1]) / 2

                left_mask = X[:, idx] <= thr
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue

                mse_left = self._mse(y[left_mask])
                mse_right = self._mse(y[right_mask])
                mse_total = (left_mask.sum() * mse_left + right_mask.sum() * mse_right) / len(y)

                impurity_decrease = parent_mse - mse_total
                if impurity_decrease >= self.min_impurity_decrease and mse_total < best_mse:
                    best_mse, best_idx, best_thr = mse_total, idx, thr

        return best_idx, best_thr

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        impurity = self._mse(y)

        if n_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return Node(None, None, None, None, y.mean(), n_samples, impurity)

        idx, thr = self._best_split(X, y)
        if idx is None:
            return Node(None, None, None, None, y.mean(), n_samples, impurity)

        left_mask = X[:, idx] <= thr
        right_mask = ~left_mask

        # Compute impurity reduction for this split
        mse_parent = self._mse(y)
        mse_left = self._mse(y[left_mask])
        mse_right = self._mse(y[right_mask])
        mse_total = (left_mask.sum() * mse_left + right_mask.sum() * mse_right) / n_samples
        impurity_reduction = mse_parent - mse_total

        # Update feature importance
        self._feature_importances[idx] += impurity_reduction * n_samples

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(idx, thr, left, right, None, n_samples, impurity)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    @property
    def feature_importances_(self):
        """Compute normalized feature importances (sum to 1)."""
        if self._feature_importances is None:
            raise ValueError("Fit the model before accessing feature importances.")
        # Normalize by total importance and total samples
        total_importance = self._feature_importances.sum()
        if total_importance == 0:
            return np.zeros(self.n_features_)
        return self._feature_importances / total_importance

def compare_feature_importance(models, feature_names):
    importances = {}
    for name, model in models.items():
        try:
            # Access feature importances
            importances[name] = model.feature_importances_
        except AttributeError:
            raise ValueError(f"Model {name} does not have a feature_importances_ attribute.")
    
    # Create DataFrame with feature names as index
    df = pd.DataFrame(importances, index=feature_names)
    df.index.name = 'Features'
    return df