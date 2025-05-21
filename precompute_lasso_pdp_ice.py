import numpy as np
import pandas as pd
import joblib
import os
from main import LassoCD
from sklearn.preprocessing import LabelEncoder

# Settings
DATA_PATH = 'Agrofood_co2_emission.csv'
MODEL_DIR = 'models'
PDP_ICE_PATH = os.path.join(MODEL_DIR, 'lasso_pdp_ice.joblib')
ALPHAS = np.arange(0, 40.5, 0.5)
N_ICE_SAMPLES = 5  # Precompute 5 ICE samples only
N_GRID = 30  # Number of grid points for each feature

# Load and preprocess data
print('Loading data...')
data = pd.read_csv(DATA_PATH)
numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Encode Area
area_encoder = joblib.load(os.path.join(MODEL_DIR, 'area_encoder.joblib'))
data['Area'] = area_encoder.transform(data['Area'])

# Extract X and y
X = data.drop('total_emission', axis=1)
y = data['total_emission']

# Filter to 90% y range
lower = y.quantile(0.05)
upper = y.quantile(0.95)
mask = (y >= lower) & (y <= upper)
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

feature_names = X.columns.tolist()

# For each feature, get grid of values (quantile-based for numeric, all for categorical)
feature_grids = {}
for feat in feature_names:
    if feat == 'Area':
        # All unique encoded values
        feature_grids[feat] = np.unique(X[feat])
    else:
        # Use quantiles for grid
        vals = np.linspace(X[feat].min(), X[feat].max(), N_GRID)
        feature_grids[feat] = vals

# Precompute PDP/ICE for each alpha, each feature
data_dict = {}
np.random.seed(42)
all_ice_indices = np.random.choice(len(X), size=N_ICE_SAMPLES, replace=False)
X_ice = X.iloc[all_ice_indices].copy()
for alpha in ALPHAS:
    alpha_key = f'{alpha:.1f}'
    model_path = os.path.join(MODEL_DIR, f'cd_lasso_at_{alpha_key}.joblib')
    if not os.path.exists(model_path):
        continue
    print(f'Processing alpha={alpha_key}')
    model = joblib.load(model_path)
    data_dict[alpha_key] = {}
    for feat in feature_names:
        grid = feature_grids[feat]
        pdp = []
        ice = [[] for _ in range(N_ICE_SAMPLES)]  # 5 lists, one per sample
        for val in grid:
            X_temp = X.copy()
            X_temp[feat] = val
            preds = model.predict(X_temp)
            pdp.append(np.mean(preds))
            # ICE for all 5 samples
            for i in range(N_ICE_SAMPLES):
                x_row = X_ice.iloc[i].copy()
                x_row[feat] = val
                pred = model.predict([x_row.values])[0]
                ice[i].append(pred)
        data_dict[alpha_key][feat] = {
            'x': grid.tolist(),
            'pdp': pdp,
            'ice': ice,  # 5 curves, each is a list of len(grid)
            'ice_indices': all_ice_indices.tolist(),
        }

print(f'Saving to {PDP_ICE_PATH}')
joblib.dump({'data': data_dict, 'feature_names': feature_names, 'alphas': [f"{a:.1f}" for a in ALPHAS]}, PDP_ICE_PATH)
print('Done.') 