from flask import Flask, render_template, request, jsonify, Response
import joblib
import numpy as np
import pandas as pd
import json
from sklearn.tree import _tree
from supertree import SuperTree
import tempfile
import os
from sklearn.preprocessing import LabelEncoder
from ipywidgets.embed import embed_minimal_html

app = Flask(__name__)

# Load models and feature means
feature_means = joblib.load('models/feature_means.joblib')
area_encoder = joblib.load('models/area_encoder.joblib')

# Load original data
data = pd.read_csv('Agrofood_co2_emission.csv')
# Handle numeric columns and fill NaN values
numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
# Handle Area feature with LabelEncoder
data.Area = area_encoder.transform(data.Area)
# Extract X and y
X_train = data.drop("total_emission", axis=1)
y_train = data["total_emission"]

# Filter data to 90% range
lower_bound = y_train.quantile(0.05)  # 5th percentile
upper_bound = y_train.quantile(0.95)  # 95th percentile
mask = (y_train >= lower_bound) & (y_train <= upper_bound)
y_train = y_train[mask]
X_train = X_train[mask]

# Compute feature means from filtered data
feature_means = X_train.mean()

# Load Lasso models
lasso_cd = joblib.load('models/lasso_cd.joblib')
lasso_gd_params = joblib.load('models/lasso_gd_params.joblib')
lasso_pgd_params = joblib.load('models/lasso_pgd_params.joblib')
sklearn_lasso = joblib.load('models/sklearn_lasso.joblib')

# Load Decision Tree models
scratch_dt = joblib.load('models/scratch_dt.joblib')
sklearn_dt = joblib.load('models/sklearn_dt.joblib')

# Get feature names from feature_means
feature_names = feature_means.index.tolist()

@app.route('/')
def home():
    # Get unique values for Area feature
    area_values = area_encoder.classes_.tolist()
    return render_template('index.html', 
                           feature_means=feature_means.to_dict(),
                           area_values=area_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        algorithm = data['algorithm']
        lasso_type = data.get('lasso_type', 'CD')
        use_sklearn = data.get('use_sklearn', False)
        
        # Create feature vector with defaults
        feature_dict = {}
        for feature in feature_means.index:
            if feature in features and features[feature] != '':
                if feature == 'Area':
                    value = area_encoder.transform([features[feature]])[0]
                else:
                    value = float(features[feature])
                feature_dict[feature] = value
            else:
                feature_dict[feature] = feature_means[feature]
        feature_df = pd.DataFrame([feature_dict])
        predictions = {}

        if algorithm in ['Lasso', 'Both']:
            if lasso_type == 'All':
                predictions['Lasso CD (Scratch)'] = float(lasso_cd.predict(feature_df)[0])
                predictions['Lasso GD (Scratch)'] = float((np.dot(feature_df, lasso_gd_params['coef_']) + lasso_gd_params['intercept_'])[0])
                predictions['Lasso PGD (Scratch)'] = float((np.dot(feature_df, lasso_pgd_params['coef_']) + lasso_pgd_params['intercept_'])[0])
            else:
                if lasso_type == 'CD':
                    predictions['Lasso CD (Scratch)'] = float(lasso_cd.predict(feature_df)[0])
                elif lasso_type == 'GD':
                    predictions['Lasso GD (Scratch)'] = float((np.dot(feature_df, lasso_gd_params['coef_']) + lasso_gd_params['intercept_'])[0])
                elif lasso_type == 'PGD':
                    predictions['Lasso PGD (Scratch)'] = float((np.dot(feature_df, lasso_pgd_params['coef_']) + lasso_pgd_params['intercept_'])[0])
            if use_sklearn:
                predictions['Lasso (Sklearn)'] = float(sklearn_lasso.predict(feature_df)[0])

        if algorithm in ['Decision Tree', 'Both']:
            predictions['Decision Tree (Scratch)'] = float(scratch_dt.predict(feature_df)[0])
            if use_sklearn:
                predictions['Decision Tree (Sklearn)'] = float(sklearn_dt.predict(feature_df)[0])

        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/supertree', methods=['POST'])
def generate_supertree():
    try:
        data = request.get_json()
        features = data['features']
        
        # Create feature vector with defaults for the prediction sample
        feature_dict = {}
        for feature in feature_means.index:
            if feature in features and features[feature] != '':
                if feature == 'Area':
                    value = area_encoder.transform([features[feature]])[0]
                else:
                    value = float(features[feature])
                feature_dict[feature] = value
            else:
                feature_dict[feature] = feature_means[feature]
        prediction_sample = pd.DataFrame([feature_dict])
        
        # Create a temporary file to store the HTML
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
            # Create SuperTree instance with sklearn model using the actual training data
            st = SuperTree(sklearn_dt, 
                          feature_data=X_train[:500],  # The actual training features
                          target_data=y_train[:500],   # The actual training targets
                          feature_names=feature_names,
                          target_names=['CO2 Emission'])
            
            # Save the HTML to the temporary file with the prediction sample
            st.save_html(temp_file.name, show_sample=prediction_sample.to_numpy()[0].tolist())
            
            # Read the HTML content
            with open(temp_file.name, 'r') as f:
                html_content = f.read()
            
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            return html_content
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/lasso_coefficients', methods=['GET'])
def get_lasso_coefficients():
    try:
        # Load all pre-trained Lasso models
        alpha_values = np.arange(0, 40.5, 0.5)
        coefficients = {}
        non_zero_counts = {}
        loss_histories = joblib.load('models/lasso_loss_histories.joblib')
        for alpha in alpha_values:
            # Always use one decimal place for alpha in the filename
            model_path = f'models/cd_lasso_at_{alpha:.1f}.joblib'
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                alpha_key = f'{alpha:.1f}'  # Use string key with one decimal place
                coefficients[alpha_key] = model.coef_.tolist()
                non_zero_counts[alpha_key] = int(np.sum(np.abs(model.coef_) > 1e-10))
        return jsonify({
            'coefficients': coefficients,
            'non_zero_counts': non_zero_counts,
            'feature_names': feature_names,
            'loss_histories': loss_histories
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
