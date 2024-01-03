import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV

# Load the training data from the saved CSV file
train_df = pd.read_csv("/Users/danieldominguez/Documents/Code/ATS_RandomForests/data/train_v1.csv")

# Load the testing dataset with features
test_df = pd.read_csv("/Users/danieldominguez/Documents/Code/ATS_RandomForests/data/test_v1.csv")

# Define the columns to one-hot encode
categorical_columns = ['ecoregion', 'type', 'season','WC']

# Remove 'mag' from both DataFrames
train_df = train_df.drop(['mag', 'uniqueID'], axis=1)
test_df = test_df.drop(['mag', 'uniqueID'], axis=1)


# Create a OneHotEncoder instance
encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')  # Drop the first category to avoid multicollinearity

# Fit and transform the encoder on the training data
train_encoded = encoder.fit_transform(train_df[categorical_columns])

# Transform the testing data using the same encoder
test_encoded = encoder.transform(test_df[categorical_columns])

# Get the feature names for the one-hot encoded columns
feature_names = encoder.get_feature_names_out(categorical_columns)

# Create DataFrames from the one-hot encoded arrays
train_encoded_df = pd.DataFrame(data=train_encoded, columns=feature_names, index=train_df.index)
test_encoded_df = pd.DataFrame(data=test_encoded, columns=feature_names, index=test_df.index)

# Concatenate the one-hot encoded DataFrames with the original DataFrames
train_df = pd.concat([train_df, train_encoded_df], axis=1)
test_df = pd.concat([test_df, test_encoded_df], axis=1)

# Drop the original categorical columns
train_df = train_df.drop(categorical_columns, axis=1)
test_df = test_df.drop(categorical_columns, axis=1)

# Define the feature columns and target column
feature_columns = train_df.columns.drop('value')
target_column = 'value'

# Split the data into training and testing sets
train_features = train_df[feature_columns]
train_target = train_df[target_column]
test_features = test_df[feature_columns]
test_target = test_df[target_column]

# Apply MinMax scaling to numerical features
scaler = MinMaxScaler(feature_range=(-1, 1))
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# Create an XGBoost Regressor
xgb_regressor = xgb.XGBRegressor(random_state=22)

# Define the hyperparameter grid for grid search
param_grid = {
    'n_estimators': [50,100, 200,500],
    'max_depth': [10, 25, 50, 100],
    'learning_rate': [0.001, 0.01, 0.1],
}

# Create the grid search object
grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the grid search to the training data
grid_search.fit(train_features_scaled, train_target)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model from grid search for predictions
best_xgb_regressor = grid_search.best_estimator_
predictions = best_xgb_regressor.predict(test_features_scaled)

# Calculate and print the performance metrics
mae_errors = abs(predictions - test_target)
print('Error (MAE): ', round(np.mean(mae_errors), 2))
mse_errors = np.sqrt(np.mean((predictions - test_target) ** 2))
print('Error (MSE): ', round(mse_errors, 2))

# Plotting the actual vs predicted values
plt.scatter(test_target, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()