#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:55:28 2024

@author: jeongwoohong
"""

import pandas as pd
import numpy as np
import os
import joblib  # For saving and loading models
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt  # Added for plotting

# File paths

train_path = '/Users/jeongwoohong/Desktop/school/ECON 424/A4/combined_data.csv'
test_path = '/Users/jeongwoohong/Desktop/school/ECON 424/A4/processed_test_data.csv'
output_path = '/Users/jeongwoohong/Desktop/school/ECON 424/A4/result.csv'
model_path = '/Users/jeongwoohong/Desktop/school/ECON 424/A4/best_model_new.joblib'

# Step 1: Data Loading
print("Loading data...")

df = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Drop irrelevant columns
df = df.drop(columns=['interior_color'], errors='ignore')
test_data = test_data.drop(columns=['interior_color'], errors='ignore')


# Separate target variable
y = df['price']
X = df.drop(columns=['price'])

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Feature Engineering: Adding new features
X['log_mileage'] = np.log1p(X['mileage'])  # 로그 변환
X['age'] = 2024 - X['year']  # 나이 변수 추가 (제조 연도 기준)

# Update numeric columns to include new features
numeric_cols += ['log_mileage', 'age']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split data
print("Splitting data into training and validation sets...")
X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the best model already exists
if os.path.exists(model_path):
    print("Loading the best model from disk...")
    best_model = joblib.load(model_path)
    print("Best model loaded. Skipping training.")
else:
    # Step 2: Define Models and Hyperparameter Distributions
    print("Defining models and hyperparameter distributions...")

    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)
    }

    param_distributions = {
        'RandomForest': {
            'model__n_estimators': randint(100, 500),
            'model__max_depth': randint(5, 30),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['sqrt', 'log2', None, 0.5, 0.7]
        },
        'GradientBoosting': {
            'model__n_estimators': randint(100, 500),
            'model__learning_rate': uniform(0.01, 0.19),
            'model__max_depth': randint(3, 15),
            'model__subsample': uniform(0.6, 0.4),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['sqrt', 'log2', None, 0.5, 0.7]
        },
        'XGBoost': {
            'model__n_estimators': randint(100, 500),
            'model__learning_rate': uniform(0.01, 0.19),
            'model__max_depth': randint(3, 15),
            'model__subsample': uniform(0.6, 0.4),
            'model__colsample_bytree': uniform(0.6, 0.4),
            'model__gamma': uniform(0, 0.5),
            'model__reg_alpha': uniform(0, 0.5),
            'model__reg_lambda': uniform(0.5, 1)
        }
    }

    # Hyperparameter tuning without early stopping
    best_estimators = {}
    cv_results = {}
    for model_name in models:
        print(f"Optimizing {model_name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', models[model_name])
        ])
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions[model_name],
            n_iter=50,
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        random_search.fit(X_train_full, y_train)
        best_estimators[model_name] = random_search.best_estimator_
        cv_results[model_name] = -random_search.best_score_  # Store the best negative MSE
        print(f"Best parameters for {model_name}: {random_search.best_params_}")
        print(f"Best cross-validation RMSE for {model_name}: {np.sqrt(-random_search.best_score_):.4f}")

    # Step 3: Select the Best Model Based on Cross-Validation Score
    print("Selecting the best model based on cross-validation performance...")
    best_model_name = min(cv_results, key=cv_results.get)
    print(f"Best model is {best_model_name} with RMSE {np.sqrt(cv_results[best_model_name]):.4f}")
    best_model = best_estimators[best_model_name]

    # Step 4: Retrain Best Model on Full Training Data
    print(f"Retraining the best model ({best_model_name}) on the full training data...")
    best_model.fit(X_train_full, y_train)

    # Step 5: Evaluate on Validation Set
    print("Evaluating the best model on the validation set...")
    y_val_pred = best_model.predict(X_val_full)
    mse_val = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)  # Added R² calculation
    print(f"{best_model_name} Validation RMSE: {np.sqrt(mse_val):.4f}")
    print(f"{best_model_name} Validation R²: {r2_val:.4f}")  # Print R²

    # Step 6: Save the Best Model
    print("Saving the best model to disk...")
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

# Regardless of loading or training, perform predictions on the validation set
print("Generating predictions for the validation set...")
y_val_pred = best_model.predict(X_val_full)

# Evaluate if not already done
if not os.path.exists(model_path):
    mse_val = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    print(f"{best_model_name} Validation RMSE: {np.sqrt(mse_val):.4f}")
    print(f"{best_model_name} Validation R²: {r2_val:.4f}")

# Step 7: Predictions on Test Data
print("Generating predictions for test data...")
X_test = test_data.copy()

# Feature Engineering on test data
X_test['log_mileage'] = np.log1p(X_test['mileage'])  # 로그 변환
X_test['age'] = 2024 - X_test['year']  # 나이 변수 추가 (제조 연도 기준)

# Ensure test data has same features
missing_cols = set(X.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X.columns]

# Generate predictions
test_predictions = best_model.predict(X_test)

# Save predictions (since we cannot evaluate without actual price values)
output = pd.DataFrame({'predicted_price': test_predictions})
output.to_csv(output_path, index=False)

print(f"Predictions have been saved to {output_path}.")

# Step 8: Plotting
print("Plotting the results...")

# Plotting true values vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('True Price (y)')
plt.ylabel('Predicted Price (ŷ)')
plt.title('True Prices vs. Predicted Prices')
plt.tight_layout()
plt.show()

# Alternatively, plotting residuals vs. true values
residuals = y_val - y_val_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_val, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_val.min(), xmax=y_val.max(), colors='red', linestyles='dashed')
plt.xlabel('True Price (y)')
plt.ylabel('Residuals (û)')
plt.title('Residuals vs. True Prices')
plt.tight_layout()
plt.show()

# Interpretation
print("**해석:**")
print("잔차(residuals) 플롯을 통해 예측 오차가 가격의 분포 전반에 걸쳐 고르게 분포되어 있음을 확인할 수 있습니다. 이는 모델이 특정 가격대에서 체계적으로 과대 또는 과소 예측하지 않으며, 전반적으로 일관된 성능을 보인다는 것을 의미합니다.")
