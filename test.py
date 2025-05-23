from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import ace_tools_open as ace_tools
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
import pandas as pd
from main import *


# Load the CSV file (make sure the path is correct)
data = pd.read_csv('Agrofood_co2_emission.csv')

numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Convert categorical variables to numerical
lea = LabelEncoder()
data.Area = lea.fit_transform(data.Area)

X = data.drop("total_emission", axis=1)
y = data["total_emission"]

# Define the percentile bounds (10th and 90th percentiles)
lower_bound = y.quantile(0.05)  # 10th percentile
upper_bound = y.quantile(0.95)  # 90th percentile

# Identify outliers (values outside the 10th to 90th percentile range)
outliers = y[(y < lower_bound) | (y > upper_bound)]
print(f"Number of outliers: {len(outliers)}")
print(f"Outlier values:\n{outliers}")

# Filter X and y to keep only data within the 10th to 90th percentile range
mask = (y >= lower_bound) & (y <= upper_bound)
y_cleaned = y[mask]
X_cleaned = X[mask]

# Use the cleaned variables
y = y_cleaned.copy()


X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.2, random_state=42
)


def run_lasso_scratch_cd(X_train, y_train, alpha, tol, max_iter):
    lasso_cd = LassoCD(alpha=alpha, tol=tol, max_iter=max_iter)
    lasso_cd.fit(X_train, y_train)
    return lasso_cd.coef_, lasso_cd.intercept_

def run_lasso_scratch_pgd(X_train, y_train, alpha, lr, max_iter, tol):
    lasso_gd = LassoPGD(alpha, lr, max_iter, tol)
    lasso_gd.fit(X_train, y_train)
    return lasso_gd.coef_, lasso_gd.intercept_
def run_lasso_sklearn(X_train, y_train, alpha, tol, max_iter):

    lasso = Lasso(alpha, tol=tol, max_iter=max_iter)
    lasso.fit(X_train, y_train)
    return lasso.coef_, lasso.intercept_
def compare_models(X_train, y_train, X_test, y_test, alpha, tol, max_iter, lr):
    methods = {
        'Lasso_sklearn'    : run_lasso_sklearn,
        'Lasso_scratch_CD' : run_lasso_scratch_cd,
        'Lasso_scratch_PGD': run_lasso_scratch_pgd,
    }

    coef_data    = {}
    metrics_data = {}

    for name, func in methods.items():
        # fit & get coef/intercept
        if name == 'Lasso_scratch_PGD':
            coef, intercept = func(X_train, y_train, alpha, lr, max_iter, tol)
        else:
            coef, intercept = func(X_train, y_train, alpha, tol, max_iter)

        # flatten in case they come back 2D
        coef      = np.ravel(coef)
        intercept = float(intercept.item())

        # store
        coef_data[name] = np.hstack(([intercept], coef))

        # predictions
        y_train_pred = X_train.values.dot(coef) + intercept
        y_test_pred  = X_test .values.dot(coef) + intercept

        metrics_data[name] = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2' : r2_score(y_test,  y_test_pred),
            'train_rmse': root_mean_squared_error(y_train, y_train_pred),
            'test_rmse' : root_mean_squared_error(y_test,  y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae' : mean_absolute_error(y_test,  y_test_pred)
        }

    # Build index: intercept + actual feature names
    feature_names = ['intercept'] + list(X_train.columns)
    coef_df = pd.DataFrame(coef_data, index=feature_names)

    metrics_df = pd.DataFrame(metrics_data).T

    return coef_df, metrics_df

# Model parameters
alpha = 20 # Not the best param, because our target is strongly correlated with all features, if alpha is small, no coeficient is shinked to 0, lasso can not select features
tol = 1e-6
max_iter = 50000
lr = 0.05

# Compare models using our function
coef_comparison, metrics_comparison = compare_models(
    X_train, y_train,
    X_test, y_test,
    alpha=alpha, tol=tol,
    max_iter=max_iter, lr=lr
)

# Display results
ace_tools.display_dataframe_to_user("Model Metrics", metrics_comparison)
ace_tools.display_dataframe_to_user("Model Coefficents", coef_comparison)

# Save metrics to CSV for plotting script
metrics_comparison.to_csv("plots/metrics_comparison.csv")

def compare_trees(X_train, y_train, X_test, y_test,
                  max_depth=None, min_samples_split=2, random_state=42):
    methods = {
        'Tree_scratch': ScratchDecisionTreeRegressor,
        'Tree_sklearn': SKDecisionTreeRegressor
    }
    metrics = []
    models = {}

    for name, func in methods.items():
        if name == 'Tree_sklearn':
            model = func(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
        else:
            model = func(max_depth=max_depth, min_samples_split=min_samples_split)
    
        model.fit(X_train, y_train)
        models[name] = model
        
        # Collect metrics
        for split, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            y_pred = model.predict(X)
            metrics.append({
                'model': name,
                'split': split,
                'r2': r2_score(y, y_pred),
                'rmse': root_mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred)
            })

    # Create initial DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Pivot to align with Lasso format (single-level columns)
    metrics_df = metrics_df.pivot(index='model', columns='split', values=['r2', 'rmse', 'mae'])

    # Flatten the MultiIndex columns
    metrics_df.columns = [f"{split}_{metric}" for metric in ['r2', 'rmse', 'mae'] for split in ['train', 'test']]
    metrics_df = metrics_df.reset_index().set_index('model')

    return metrics_df, models


tree_metrics, models = compare_trees(X_train, y_train, X_test, y_test, max_depth=7, min_samples_split=10, random_state=42)
importance_df = compare_feature_importance(models, X_train.columns)
ace_tools.display_dataframe_to_user("Decision tree metrics", tree_metrics)
ace_tools.display_dataframe_to_user("Feature Importance", importance_df)

# Save tree metrics to CSV for plotting script
tree_metrics.to_csv("plots/tree_metrics.csv")

# --- Linear Regression (Scratch and Sklearn) ---
lin_model_scratch = ScratchLinearRegression().fit(X_train, y_train)
lin_model_sklearn = LinearRegression().fit(X_train, y_train)

# --- KNN Regression (Scratch) ---
# Remove grid search, use k=5, p=2 for both
knn_model_scratch = ScratchKNNRegressor(n_neighbors=5, p=2).fit(X_train, y_train)
knn_model_sklearn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

# --- Collect metrics for all models ---
def get_metrics(model, X_train, y_train, X_test, y_test, is_sklearn=False):
    if is_sklearn:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    return {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': root_mean_squared_error(y_train, y_train_pred),
        'test_rmse': root_mean_squared_error(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred)
    }

# Existing metrics
all_metrics = {}
all_metrics['Lasso_scratch_CD'] = metrics_comparison.loc['Lasso_scratch_CD'].to_dict()
all_metrics['Lasso_sklearn'] = metrics_comparison.loc['Lasso_sklearn'].to_dict()
all_metrics['Lasso_scratch_PGD'] = metrics_comparison.loc['Lasso_scratch_PGD'].to_dict()
all_metrics['Tree_scratch'] = tree_metrics.loc['Tree_scratch'].to_dict()
all_metrics['Tree_sklearn'] = tree_metrics.loc['Tree_sklearn'].to_dict()

# Add Linear Regression
all_metrics['Linear_scratch'] = get_metrics(lin_model_scratch, X_train, y_train, X_test, y_test)
all_metrics['Linear_sklearn'] = get_metrics(lin_model_sklearn, X_train, y_train, X_test, y_test, is_sklearn=True)

# Add KNN Regression
all_metrics['KNN_scratch'] = get_metrics(knn_model_scratch, X_train, y_train, X_test, y_test)
all_metrics['KNN_sklearn'] = get_metrics(knn_model_sklearn, X_train, y_train, X_test, y_test, is_sklearn=True)

# Save all metrics to CSV for plotting
all_metrics_df = pd.DataFrame(all_metrics).T
all_metrics_df.to_csv('plots/all_metrics_comparison.csv')
