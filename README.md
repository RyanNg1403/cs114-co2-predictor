# CO2 Emission Predictor

A web application for predicting and analyzing CO2 emissions from agricultural and food production data. The app provides interactive model predictions, decision tree visualizations, Lasso regression analysis, and AI-powered recommendations for emission reduction.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Application](#web-application)
  - [Model Training Scripts](#model-training-scripts)
- [Model Details](#model-details)
- [Data](#data)
- [Environment Variables](#environment-variables)
- [Acknowledgements](#acknowledgements)

---

## Features

- **CO2 Emission Prediction**: Predicts total CO2 emissions using Lasso regression (multiple variants) and Decision Tree models.
- **Interactive Visualizations**:
  - Decision Tree (DT) visualization, including an interactive "SuperTree" view.
  - Lasso coefficient and loss history analysis across a range of regularization strengths.
- **AI Recommendations**: Provides actionable, context-specific recommendations for emission reduction using LLMs (Gemini or Groq).
- **Custom and Scikit-learn Implementations**: Compare scratch implementations of Lasso and Decision Tree with scikit-learn versions.
- **Outlier Handling**: Models are trained on data filtered to the 5th–95th percentile range for robust predictions.

---

## Project Structure

```
.
├── app.py                      # Main Flask web application
├── requirements.txt            # Python dependencies
├── Agrofood_co2_emission.csv   # Main dataset
├── main.py                     # Custom model implementations (Lasso, Decision Tree)
├── save_models.py              # Script to train and save main models
├── train_lasso_models.py       # Script to train Lasso models for a range of alphas
├── llm_recommendations.py      # LLM-based recommendation logic
├── models/                     # Saved model files and encoders
├── templates/
│   └── index.html              # Main HTML template for the web app
└── ...
```

---

## Installation

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies** (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare environment variables**:

   - Create a `.env` file in the root directory.
   - Set the following variable to choose your LLM provider:
     ```
     MODEL_PROVIDER=GOOGLE
     ```
     or
     ```
     MODEL_PROVIDER=GROQ
     ```
   - (You may need to set up API keys for Gemini or Groq as required by `pydantic-ai`.)

4. **Prepare the data**:

   - Ensure `Agrofood_co2_emission.csv` is present in the root directory.

5. **Train and save models** (if not already present):

   - To train all main models (Lasso, Decision Tree, encoders, etc.):
     ```bash
     python save_models.py
     ```
   - To train Lasso models for a range of alpha values (for visualization):
     ```bash
     python train_lasso_models.py
     ```

---

## Usage

### Web Application

Start the Flask app:

```bash
python app.py
```

- Open your browser and go to `http://127.0.0.1:5000/`
- Use the tabs to:
  - Make predictions with different models and feature inputs.
  - Visualize decision trees and Lasso regression coefficients.
  - Get AI-powered recommendations for emission reduction.

### Model Training Scripts

- `save_models.py`: Trains and saves the main models (Lasso variants, Decision Trees, encoders, feature means).
- `train_lasso_models.py`: Trains and saves Lasso models for a range of alpha values (0 to 40, step 0.5) and their loss histories for visualization.

---

## Model Details

- **Lasso Regression**:
  - Custom implementations: Coordinate Descent (CD), Gradient Descent (GD), Proximal Gradient Descent (PGD).
  - Scikit-learn Lasso for comparison.
  - Coefficient and loss history visualization for different regularization strengths (alpha).
- **Decision Tree Regression**:
  - Custom scratch implementation.
  - Scikit-learn DecisionTreeRegressor for comparison.
  - Interactive tree visualization using SuperTree.
- **LLM Recommendations**:
  - Uses Gemini or Groq via `pydantic-ai` to generate actionable recommendations based on model predictions and input features.

---

## Data

- **Source**: `Agrofood_co2_emission.csv`
- **Preprocessing**:
  - Numeric columns: NaN values filled with column means.
  - Categorical "Area" column: Encoded using `LabelEncoder`.
  - Outlier removal: Only data within the 5th–95th percentile of the target variable is used for training.

---

## Environment Variables

- `.env` file required for LLM provider selection:
  - `MODEL_PROVIDER=GOOGLE` or `MODEL_PROVIDER=GROQ`
- Additional environment variables may be required for LLM API keys, depending on your provider and `pydantic-ai` setup.

---

## Acknowledgements

- Data and domain inspiration from agricultural and food production CO2 emission studies.
- Uses [SuperTree](https://github.com/mljar/supertree) for interactive tree visualization.
- LLM recommendations powered by [pydantic-ai](https://github.com/pydantic/pydantic-ai).
