
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# set local path
local_path = "/Users/danieldominguez/Documents/Code/DOC_WSC/Data/"

# Load the training data from the CSV file
train_data = pd.read_csv(local_path+ "DOC_train_v1.csv")

# Load the test data from the CSV file
test_data = pd.read_csv(local_path+ "DOC_test_v1.csv")

# Concatenate the training and test data
data = pd.concat([train_data, test_data], ignore_index=True)

# Separate the target variable (value) and drop the 'mag' column and any other identifying column
y = data["value"]
X = data.drop(["value", "mag","uniqueID"], axis=1)

# Perform one-hot encoding for categorical variables
categorical_cols = ["ecoregion", "type", "season",'WC']
X_categorical = X[categorical_cols]
X_numerical = X.drop(categorical_cols, axis=1)

encoder = OneHotEncoder(sparse=False)
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Perform min-max scaling from -1 to 1 on numerical features
scaler = MinMaxScaler(feature_range=(-1, 1))
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Combine encoded categorical and scaled numerical features
X_processed = np.hstack((X_categorical_encoded, X_numerical_scaled))

# Split the data into training and test sets based on the shape of the original datasets
X_train = X_processed[:len(train_data)]
X_test = X_processed[len(train_data):]

# Split the target variable (value) into training and test sets
y_train = y[:len(train_data)]
y_test = y[len(train_data):]



def dynamic_penalty_loss(y_true, y_pred):
    penalty=y_true
    absolute_errors = tf.abs(y_true - y_pred)
    penalty_mask = tf.cast(absolute_errors >= y_true*0.7, dtype=tf.float32) #absolute_errors <= 10.0, experiment with reinforcement learning
    penalty = penalty_mask * penalty  # Apply the penalty to values greater than x
    loss = tf.reduce_mean(absolute_errors + penalty)
    return loss

def weighted_dynamic_penalty_loss(y_true, y_pred):
    penalty = tf.where(y_true < 10, y_true * 3, y_true)
    absolute_errors = tf.abs(y_true - y_pred)
    penalty_mask = tf.cast(absolute_errors <= 10, dtype=tf.float32) #absolute_errors <= 10.0
    penalty = penalty_mask * penalty  # Apply the penalty to values greater than 10 times the actual value
    loss = tf.reduce_mean(absolute_errors + penalty)
    return loss


# Define the neural network settings for regression
model_settings = {
    "hiddens": [1024,1024,1024,1024],  # Specify the number of hidden layers and units
    "activations": [ "leaky_relu", "leaky_relu", "leaky_relu", "leaky_relu"],  # Specify the activation functions for hidden layers
    "dropout_rate": 0.2,  # Specify the dropout rate
    "learning_rate": 0.001,  # Specify the learning rate #0.00001
    "random_seed": 42  # Specify a random seed for reproducibility
}

# Build the model for regression
def build_model(x_train, settings):
    # create input layer
    input_layer = tf.keras.layers.Input(shape=x_train.shape[1:])
    # create hidden layers with dropout
    layers = input_layer
    for hidden, activation in zip(settings["hiddens"], settings["activations"]):
        layers = tf.keras.layers.Dense(
            units=hidden,
            activation=activation,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
        )(layers)
        # Add dropout layer
        layers = tf.keras.layers.Dropout(rate=settings["dropout_rate"])(layers)

    # create output layer for regression (1 unit with linear activation)
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation="linear",  # Linear activation for regression
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 1),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 2),
    )(layers)

    # construct the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model

# Compile the model for regression
def compile_model(model, settings):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
        loss=dynamic_penalty_loss,  # Custom regression loss
        metrics=["mean_absolute_error"],  # Use MAE as a metric
    )
    return model

# Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',  # Monitor validation loss
    patience=50,  # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True  # Restore the model weights from the epoch with the best validation loss
)

# Build and compile the regression model
regression_model = build_model(X_train, model_settings)
regression_model = compile_model(regression_model, model_settings)

# Calculate sample weights based on the target values
threshold = 40
sample_weights = np.where(y_train >= threshold, 1.0, 1.0)  # Assign higher weight to values >= 40

# Train the regression model with early stopping
regression_history = regression_model.fit(
    X_train,
    y_train,
    epochs=20000,  # Set a high number of epochs
    batch_size=256,  # Specify the batch size
    callbacks=[early_stopping],  # Use early stopping callback,
    sample_weight=sample_weights,  # Pass the sample weights
    shuffle=True,
)

# Make predictions on the test data
y_pred = regression_model.predict(X_test)

# Evaluate the regression model on the test data
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test Mean Absolute Error: {test_mae}")

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot loss vs. epoch
ax1.plot(regression_history.history['loss'], label='Training Loss')
ax1.plot(regression_history.history['mean_absolute_error'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs. Epoch')
ax1.legend()

# Plot prediction vs. actual on a log scale
ax2.scatter(y_test, y_pred, alpha=0.5)
ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Predicted Values (log scale)")
ax2.set_title("Prediction vs. Actual (log scale)")

# Add a one-to-one line
min_value = np.min(np.concatenate([np.ravel(y_test), np.ravel(y_pred)]))
max_value = np.max(np.concatenate([np.ravel(y_test), np.ravel(y_pred)]))
ax2.plot([min_value, max_value], [min_value, max_value], 'k--', lw=2)


# Set y-axis to log scale
ax2.set_yscale('log')
ax2.set_xscale('log')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# Test Mean Absolute ErrorL 1.99