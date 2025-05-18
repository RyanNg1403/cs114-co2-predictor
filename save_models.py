import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from main import LassoCD, LassoGD, LassoPGD, ScratchDecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

# Load and preprocess data
data = pd.read_csv('Agrofood_co2_emission.csv')

# Handle numeric columns and fill NaN values
numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Handle Area feature with LabelEncoder
lea = LabelEncoder()
data.Area = lea.fit_transform(data.Area)

# Extract X and y
X = data.drop("total_emission", axis=1)
y = data["total_emission"]

# Remove outliers
lower_bound = y.quantile(0.05)
upper_bound = y.quantile(0.95)
mask = (y >= lower_bound) & (y <= upper_bound)
y_cleaned = y[mask]
X_cleaned = X[mask]

# Save label encoder
joblib.dump(lea, 'models/area_encoder.joblib')

# Save feature means for default values
feature_means = pd.Series(X_cleaned.mean(), index=X_cleaned.columns)
joblib.dump(feature_means, 'models/feature_means.joblib')

# Train and save Lasso models
alpha = 20
tol = 1e-6
max_iter = 10000
lr = 0.05

# Lasso CD
lasso_cd = LassoCD(alpha=alpha, tol=tol, max_iter=max_iter)
lasso_cd.fit(X_cleaned, y_cleaned)
joblib.dump(lasso_cd, 'models/lasso_cd.joblib')

# Lasso GD
lasso_gd = LassoGD(alpha=alpha, lr=lr, max_iter=max_iter, tol=tol)
lasso_gd.fit(X_cleaned, y_cleaned)
joblib.dump({'coef_': lasso_gd.coef_, 'intercept_': lasso_gd.intercept_}, 'models/lasso_gd_params.joblib')

# Lasso PGD
lasso_pgd = LassoPGD(alpha=alpha, lr=lr, max_iter=max_iter, tol=tol)
lasso_pgd.fit(X_cleaned, y_cleaned)
joblib.dump({'coef_': lasso_pgd.coef_, 'intercept_': lasso_pgd.intercept_}, 'models/lasso_pgd_params.joblib')

# Sklearn Lasso
sklearn_lasso = Lasso(alpha=alpha, tol=tol, max_iter=max_iter)
sklearn_lasso.fit(X_cleaned, y_cleaned)
joblib.dump(sklearn_lasso, 'models/sklearn_lasso.joblib')

# Train and save Decision Tree models
max_depth = 7
min_samples_split = 10

# Scratch Decision Tree
scratch_dt = ScratchDecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
scratch_dt.fit(X_cleaned, y_cleaned)
joblib.dump(scratch_dt, 'models/scratch_dt.joblib')

# Sklearn Decision Tree
sklearn_dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
sklearn_dt.fit(X_cleaned, y_cleaned)
joblib.dump(sklearn_dt, 'models/sklearn_dt.joblib')

print("All models have been trained and saved successfully!") 