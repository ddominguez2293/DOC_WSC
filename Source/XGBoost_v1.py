import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load the training data from the saved CSV file
train_df = pd.read_csv("/Users/danieldominguez/Documents/Code/DOC_WSC/Data/DOC_train_v1.csv")

# Load the testing dataset with features
test_df = pd.read_csv("/Users/danieldominguez/Documents/Code/DOC_WSC/Data/DOC_test_v1.csv")

# Remove 'mag' from both DataFrames
train_df = train_df.drop(['mag', 'uniqueID'], axis=1)
test_df = test_df.drop(['mag', 'uniqueID'], axis=1)

# Define the columns to one-hot encode
categorical_columns = ['ecoregion', 'type', 'season', 'WC']

# Identify numerical columns in the train_df and test_df
numerical_columns_train = train_df.select_dtypes(include=np.number).columns
numerical_columns_test = test_df.select_dtypes(include=np.number).columns

# Create a new DataFrame with only the numerical columns from both train_df and test_df
train_numerical_df = train_df[numerical_columns_train]
test_numerical_df = test_df[numerical_columns_test]

# Apply MinMax scaling to numerical features
scaler = MinMaxScaler(feature_range=(-1, 1))
train_numerical_scaled = scaler.fit_transform(train_numerical_df)
test_numerical_scaled = scaler.transform(test_numerical_df)

# Convert train_numerical_scaled to a DataFrame
train_numerical_scaled_df = pd.DataFrame(data=train_numerical_scaled, columns=numerical_columns_train, index=train_df.index)

# Convert test_numerical_scaled to a DataFrame
test_numerical_scaled_df = pd.DataFrame(data=test_numerical_scaled, columns=numerical_columns_test, index=test_df.index)


# Transform categorical columns using one-hot encoding
encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
train_encoded = encoder.fit_transform(train_df[categorical_columns])
test_encoded = encoder.transform(test_df[categorical_columns])

# Get the feature names for the one-hot encoded columns
feature_names = encoder.get_feature_names_out(categorical_columns)

# Create DataFrames from the one-hot encoded arrays
train_encoded_df = pd.DataFrame(data=train_encoded, columns=feature_names, index=train_df.index)
test_encoded_df = pd.DataFrame(data=test_encoded, columns=feature_names, index=test_df.index)

# Concatenate the numerical and one-hot encoded DataFrames with the original DataFrame
train_final_df = pd.concat([train_numerical_scaled_df, train_encoded_df], axis=1)
test_final_df = pd.concat([test_numerical_scaled_df, test_encoded_df], axis=1)

# Define the feature columns and target column
feature_columns = train_final_df.columns.drop(['value'])
# Define the target column
target_column = 'value'

# Split the data into training and testing sets
train_features = train_final_df[feature_columns]
train_target = train_df[target_column]
test_features = test_final_df[feature_columns]
test_target = test_df[target_column]


print(numerical_columns_test)
# Define the custom evaluation metric that computes the mean quadrupled error
def custom_metric(preds, dtrain):
    labels = dtrain.get_label()
    errors = preds - labels
    mean_quadrupled_error = np.mean(errors ** 6)
    return 'mean_quadrupled_error', mean_quadrupled_error

# Define the hauber metric that uses a threshold, mae by defa
def hauber_metric(preds, dtrain):
    labels = dtrain.get_label()
    errors = preds - labels
    delta = 0.5  # You can adjust the delta parameter as needed
    quadratic_loss = np.where(np.abs(errors) < delta, 0.5 * errors**2, delta * (np.abs(errors) - 0.5 * delta))
    mean_quadratic_loss = np.mean(quadratic_loss)
    return 'mean_huber_loss', mean_quadratic_loss

def mae_metric(preds, dtrain):
    labels = dtrain.get_label()
    errors = preds - labels
    mean_absolute_error = np.mean(np.abs(errors))
    return 'mean_absolute_error', mean_absolute_error

xgb_regressor = xgb.XGBRegressor(
    n_estimators=2000,# How many boosting rounds, default 200 based on grid search
    max_depth=20, #How many features should a branch be able to use, default 10 based on grid search
    learning_rate=0.01,# How fast should the model learn, default 0.1 based on grid search
    random_state=22, # random seed for reproducibility
)

# Specify the evaluation set to track the learning curve
eval_set = [(train_features, train_target), (test_features, test_target)]

# Train the XGBoost Regressor on training data and track the learning curve
xgb_regressor.fit(
    train_features,
    train_target,
    eval_metric=hauber_metric,
    verbose=True,
    eval_set=eval_set,
)

# Retrieve the learning curve
results = xgb_regressor.evals_result()
train_logloss = results['validation_0']['mean_huber_loss']
test_logloss = results['validation_1']['mean_huber_loss']
n_rounds = len(train_logloss)

# Use the XGBoost Regressor's predict method on the test data
predictions = xgb_regressor.predict(test_features)

# Use the testing set to validate the performance
# Print out the mean absolute error (MAE)
mae_errors = abs(predictions - test_target)
print('Error (MAE): ', round(np.mean(mae_errors), 2))

# See its performance (mean squared errors)
mse_errors = np.sqrt(np.mean((predictions - test_target) ** 2))
print('Error (MSE): ', round(mse_errors, 2))

# Create a subplot for the learning curve
plt.figure(figsize=(12, 6))

# Scatter plot for predictions
plt.subplot(1, 2, 1)
plt.scatter(test_target, predictions, alpha=0.5, marker='o', color='b', label='Predictions')
plt.plot([min(test_target), max(test_target)], [min(test_target), max(test_target)], linestyle='--', color='r', label='1-1 Line')
plt.xlabel('DOC Actual (mg/L)')
plt.ylabel('DOC Predictions (mg/L)')
plt.title('DOC Predictions vs. Actual')
plt.legend(loc='upper left')
plt.xscale('log')
plt.yscale('log')
plt.grid(False)  # Remove grid lines

# Learning curve
plt.subplot(1, 2, 2)
plt.plot(range(1, n_rounds + 1), train_logloss, label='Train Log Loss')
plt.plot(range(1, n_rounds + 1), test_logloss, label='Test Log Loss')
plt.xlabel('Number of Rounds')
plt.ylabel('Mean Quadrupled Error')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Retrieve the feature importance
feature_importance = xgb_regressor.feature_importances_

# Display feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top 20 features
top_20_features = feature_importance_df.head(20)
print("Top 20 Features by Importance:")
print(top_20_features)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='blue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()