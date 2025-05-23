import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set a professional style and color palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

# Ensure the 'plots' directory exists
os.makedirs('plots', exist_ok=True)

# Load metrics from CSV files saved by test.py
metrics_df = pd.read_csv('plots/all_metrics_comparison.csv', index_col=0)

# Define a consistent color palette
palette = sns.color_palette('colorblind', 8)
bar_kwargs = dict(edgecolor='black', linewidth=1.2)

# Helper to add value labels
def add_value_labels(ax, values, fmt="{:.3f}", spacing=10):
    for i, v in enumerate(values):
        ax.annotate(fmt.format(v),
                    xy=(i, v),
                    xytext=(0, spacing),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='#333')

# Model name lists
all_model_names = [
    'Lasso_scratch_CD', 'Lasso_sklearn',
    'Tree_scratch', 'Tree_sklearn',
    'Linear_scratch', 'Linear_sklearn',
    'KNN_scratch', 'KNN_sklearn'
]
# Only keep those present in the DataFrame
all_model_names = [m for m in all_model_names if m in metrics_df.index]

# 1. Compare r2 scores of all models
r2_vals = metrics_df.loc[all_model_names, 'test_r2']
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(all_model_names, r2_vals, color=palette[:len(all_model_names)], **bar_kwargs)
ax.set_ylabel('Test $R^2$ Score', fontsize=13, fontweight='bold')
ax.set_title('Test $R^2$ Score Comparison (All Models)', fontsize=15, fontweight='bold', pad=30)
ax.set_ylim(0, max(1, r2_vals.max() + 0.12))
add_value_labels(ax, r2_vals, fmt="{:.3f}", spacing=10)
plt.xticks(fontsize=12, fontweight='bold', rotation=20)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('plots/r2_comparison_all.png', dpi=120)
plt.close()

# 2. Compare MSE losses of all models
mse_vals = metrics_df.loc[all_model_names, 'test_rmse']**2
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(all_model_names, mse_vals, color=palette[:len(all_model_names)], **bar_kwargs)
ax.set_ylabel('Test MSE Loss', fontsize=13, fontweight='bold')
ax.set_title('Test MSE Loss Comparison (All Models)', fontsize=15, fontweight='bold', pad=30)
ax.set_ylim(0, mse_vals.max() * 1.18)
add_value_labels(ax, mse_vals, fmt="{:.0f}", spacing=10)
plt.xticks(fontsize=12, fontweight='bold', rotation=20)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('plots/mse_comparison_all.png', dpi=120)
plt.close()

# 3. Compare MAE losses of all models
mae_vals = metrics_df.loc[all_model_names, 'test_mae']
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(all_model_names, mae_vals, color=palette[:len(all_model_names)], **bar_kwargs)
ax.set_ylabel('Test MAE Loss', fontsize=13, fontweight='bold')
ax.set_title('Test MAE Loss Comparison (All Models)', fontsize=15, fontweight='bold', pad=30)
ax.set_ylim(0, mae_vals.max() * 1.18)
add_value_labels(ax, mae_vals, fmt="{:.2f}", spacing=10)
plt.xticks(fontsize=12, fontweight='bold', rotation=20)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('plots/mae_comparison_all.png', dpi=120)
plt.close()

# 4. Compare r2 scores of all linear models
linear_models = [
    'Lasso_scratch_CD', 'Lasso_scratch_GD', 'Lasso_scratch_PGD', 'Lasso_sklearn',
    'Linear_scratch', 'Linear_sklearn'
]
linear_models = [m for m in linear_models if m in metrics_df.index]
linear_r2 = metrics_df.loc[linear_models, 'test_r2']
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(linear_models, linear_r2, color=palette[:len(linear_models)], **bar_kwargs)
ax.set_ylabel('Test $R^2$ Score', fontsize=13, fontweight='bold')
ax.set_title('Test $R^2$ Score Comparison (Linear Models)', fontsize=15, fontweight='bold', pad=30)
ax.set_ylim(0, max(1, linear_r2.max() + 0.12))
add_value_labels(ax, linear_r2, fmt="{:.3f}", spacing=10)
plt.xticks(fontsize=12, fontweight='bold', rotation=20)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('plots/r2_comparison_linear.png', dpi=120)
plt.close()

# 5. Compare r2 scores of all non-linear models
nonlinear_models = [
    'KNN_scratch', 'KNN_sklearn', 'Tree_scratch', 'Tree_sklearn'
]
nonlinear_models = [m for m in nonlinear_models if m in metrics_df.index]
nonlinear_r2 = metrics_df.loc[nonlinear_models, 'test_r2']
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(nonlinear_models, nonlinear_r2, color=palette[:len(nonlinear_models)], **bar_kwargs)
ax.set_ylabel('Test $R^2$ Score', fontsize=13, fontweight='bold')
ax.set_title('Test $R^2$ Score Comparison (Non-linear Models)', fontsize=15, fontweight='bold', pad=30)
ax.set_ylim(0, max(1, nonlinear_r2.max() + 0.12))
add_value_labels(ax, nonlinear_r2, fmt="{:.3f}", spacing=10)
plt.xticks(fontsize=12, fontweight='bold', rotation=20)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('plots/r2_comparison_nonlinear.png', dpi=120)
plt.close()

# 6. Compare r2 scores of all scratch models
scratch_models = [
    'Lasso_scratch_CD', 'Lasso_scratch_GD', 'Lasso_scratch_PGD',
    'Tree_scratch', 'Linear_scratch', 'KNN_scratch'
]
scratch_models = [m for m in scratch_models if m in metrics_df.index]
scratch_r2 = metrics_df.loc[scratch_models, 'test_r2']
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(scratch_models, scratch_r2, color=palette[:len(scratch_models)], **bar_kwargs)
ax.set_ylabel('Test $R^2$ Score', fontsize=13, fontweight='bold')
ax.set_title('Test $R^2$ Score Comparison (Scratch Models)', fontsize=15, fontweight='bold', pad=30)
ax.set_ylim(0, max(1, scratch_r2.max() + 0.12))
add_value_labels(ax, scratch_r2, fmt="{:.3f}", spacing=10)
plt.xticks(fontsize=12, fontweight='bold', rotation=20)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('plots/r2_comparison_scratch.png', dpi=120)
plt.close()

# 7. Compare r2 scores of all sklearn models
sklearn_models = [
    'Lasso_sklearn', 'Tree_sklearn', 'Linear_sklearn', 'KNN_sklearn'
]
sklearn_models = [m for m in sklearn_models if m in metrics_df.index]
sklearn_r2 = metrics_df.loc[sklearn_models, 'test_r2']
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(sklearn_models, sklearn_r2, color=palette[:len(sklearn_models)], **bar_kwargs)
ax.set_ylabel('Test $R^2$ Score', fontsize=13, fontweight='bold')
ax.set_title('Test $R^2$ Score Comparison (Sklearn Models)', fontsize=15, fontweight='bold', pad=30)
ax.set_ylim(0, max(1, sklearn_r2.max() + 0.12))
add_value_labels(ax, sklearn_r2, fmt="{:.3f}", spacing=10)
plt.xticks(fontsize=12, fontweight='bold', rotation=20)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('plots/r2_comparison_sklearn.png', dpi=120)
plt.close()

print('All plots saved in the "plots" directory.') 