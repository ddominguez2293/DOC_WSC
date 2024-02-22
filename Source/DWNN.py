import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt

# this code applies the Deep-Wide Neural Network architecure
# The Deep part is connecting the remote sensing parameters through a series of hidden layers with decreaseing amount of nodes in subsequent layers
# The Wide part connects the categorical columns to the last layer in the nodes 
# In theory this helps the neural network pick up on established trends represented in the data such as seasons, ecoregion, etc without weighting down the value fo the remote sensing parameters

# set local path
local_path = "/Users/danieldominguez/Documents/Code/DOC_WSC/Data/"

# Define a custom loss function (quadratic error)
def custom_loss(y_true, y_pred):
    # Calculate the cubed error (mean cubed error)
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    return loss

# Mean average error loss
def mae_loss(y_true, y_pred):
    # Calculate the Mean Absolute Error (MAE)
    loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return loss

#huber_loss normally (<=) uses mse below delta and mae above, reverse(>=) if  I want higher values weighted more
def huber_loss(y_true, y_pred, delta=1.0):#10
    error = y_true - (y_pred)
    is_large_error = tf.abs(error) <= delta  # Changed from <= to >=
    loss = tf.where(is_large_error, 0.5 * tf.square(error), delta * (tf.abs(error) - 0.5 * delta))
    return tf.reduce_mean(loss)

#The Pinball Loss is a quantile regression loss
def pinball_loss(y_true, y_pred, quantile=0.8):
    residual = y_true - y_pred
    pinball_loss = tf.reduce_mean(tf.maximum(quantile * residual, (quantile - 1) * residual), axis=-1)
    return pinball_loss

# huber but also adds weight for non-gaussian distribution
def asymmetric_huber_loss(y_true, y_pred, alpha=10.0, beta=0.8):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic_loss = tf.where(abs_error <= alpha, 0.5 * tf.square(error), alpha * (abs_error - 0.5 * alpha))
    asymmetric_loss = tf.where(error >= 0, beta * quadratic_loss, (1 - beta) * quadratic_loss)
    return tf.reduce_mean(asymmetric_loss, axis=-1)

def mae_loss_with_penalty(y_true, y_pred, penalty=100.0):
    absolute_errors = tf.abs(y_true - y_pred)
    penalty_mask = tf.cast(absolute_errors >= 50, dtype=tf.float32) #absolute_errors <= 10.0
    penalty = penalty_mask * penalty  # Apply the penalty to values greater than 10 times the actual value
    loss = tf.reduce_mean(absolute_errors + penalty)
    return loss

def dynamic_penalty_loss(y_true, y_pred):
    penalty=y_true # error penalty will increase with the true value so higher values are weighted more
    absolute_errors = tf.abs(y_true - y_pred)
    penalty_mask = tf.cast(absolute_errors >= y_true*0.7, dtype=tf.float32) #absolute_errors >= 10.0
    penalty = penalty_mask * penalty  # Apply the penalty to values greater than 10 times the actual value
    loss = tf.reduce_mean(absolute_errors + penalty)
    return loss

def weighted_dynamic_penalty_loss(y_true, y_pred):
    absolute_errors = tf.abs(y_true - y_pred)
    penalty = tf.where(y_true < 10, 0.01, y_true)
    penalty_mask = tf.cast(absolute_errors >= 10, dtype=tf.float32) #absolute_errors >= 10.0 applies to only severely overpredicted low values or if is not predicting high values 
    penalty = penalty_mask * penalty  # Apply the penalty to values greater than 10 times the actual value
    loss = tf.reduce_mean(absolute_errors + penalty)
    return loss

def scaled_absolute_loss(y_true, y_pred):
    # Calculate the absolute errors
    absolute_errors = tf.abs(y_true - y_pred)
    
    # Calculate the scaling factor (magnitude of y_true)
    magnitude = tf.abs(y_true) + 1e-10  # Add a small constant to avoid division by zero
    
    # Scale the absolute errors by the magnitude
    scaled_errors = absolute_errors / magnitude
    
    # Calculate the loss as the mean of the scaled errors
    loss = tf.reduce_mean(scaled_errors)
    
    return loss

def scaled_absolute_loss_with_penalty(y_true, y_pred, lower_bound=0.1, upper_bound=10.0):
    # Calculate the absolute errors
    absolute_errors = tf.abs(y_true - y_pred)
    
    # Calculate the scaling factor (magnitude of y_true)
    magnitude = tf.abs(y_true) + 1e-10  # Add a small constant to avoid division by zero
    
    # Scale the absolute errors by the magnitude
    scaled_errors = absolute_errors / magnitude
    
    # Apply a penalty for values outside the specified range
    penalty = tf.where((scaled_errors < lower_bound) | (scaled_errors > upper_bound), scaled_errors, 10.0)
    
    # Calculate the loss as the mean of the scaled errors and the penalty
    loss = tf.reduce_mean(scaled_errors + penalty)
    
    return loss


def weighted_dynamic_reward_loss(y_true, y_pred):
    reward = tf.where(y_true < 10, 0.01, y_true/2) # weight so that the amount of small values does not override the high values can make loss infinitely negative is set to high
    absolute_errors = tf.abs(y_true - y_pred)
    reward_mask = tf.cast(absolute_errors <= 10, dtype=tf.float32) # Apply to small errors
    reward = reward_mask * reward  # Apply the reward to values within a certain error threshold
    loss = tf.reduce_mean(absolute_errors+10- reward)  # Invert the penalty
    return loss


def dynamic_reward_loss(y_true, y_pred):
    reward = y_pred
    absolute_errors = tf.abs(y_true - y_pred)
    reward_mask = tf.cast(absolute_errors <= 1, dtype=tf.float32)
    reward = reward_mask * reward  # Apply the reward for predictions within the specified threshold
    loss = tf.reduce_mean(absolute_errors - reward)  # Change penalty to reward
    return loss

# Read in the data file
data = pd.read_csv(local_path + "aquasat_full_secchi.csv")

# Select the feature to predict as the label
labels = np.array(data['value'])

# Drop the 'uniqueID' column and the label column
features = data.drop(['value', 'uniqueID', 'mag',"SiteID"], axis=1)

# One-hot encode categorical features
categorical_columns = ['ecoregion', 'type', 'season', 'WC']
cat_encoder = OneHotEncoder(sparse=False)
cat_encoded = cat_encoder.fit_transform(features[categorical_columns])

# Get the numerical features
numerical_features = features.drop(categorical_columns, axis=1)

# Scale numerical features to the range [0, 1] using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
numerical_features_scaled = scaler.fit_transform(numerical_features)

# Define the wide input layer
wide_input_layer = tf.keras.layers.Input(shape=(cat_encoded.shape[1],), name='wide_input')  # Use cat_encoded.shape[1] for wide input

# Experimented with doing a smaller version of the deep network for the categorical columns if you dont want to use this make sure to call the input layer below
# In theory this should change the weights to be more represenatitive of the data but not over train it
wide_layer_1 = tf.keras.layers.Dense(32, activation='relu')(wide_input_layer)
wide_layer_1 = tf.keras.layers.Dropout(0.2)(wide_layer_1)

wide_layer_2 = tf.keras.layers.Dense(16, activation='relu')(wide_layer_1)
wide_layer_2 = tf.keras.layers.Dropout(0.0)(wide_layer_2)

wide_layer_3 = tf.keras.layers.Dense(8, activation='relu')(wide_layer_2)
wide_layer_3 = tf.keras.layers.Dropout(0.0)(wide_layer_3)

wide_layer_4 = tf.keras.layers.Dense(4, activation='relu')(wide_layer_3)
wide_layer_4 = tf.keras.layers.Dropout(0.0)(wide_layer_4)

wide_layer_5 = tf.keras.layers.Dense(2, activation='relu')(wide_layer_4)
wide_layer_5 = tf.keras.layers.Dropout(0.0)(wide_layer_5)

wide_layer_6 = tf.keras.layers.Dense(1, activation='relu')(wide_layer_5)
wide_layer_6 = tf.keras.layers.Dropout(0.0)(wide_layer_6)

# Define the deep input layer
deep_input_layer = tf.keras.layers.Input(shape=(numerical_features_scaled.shape[1],), name='deep_input')

dr=0.2 # drop out rate for all layers

# Create the deep part of the network
deep_layer_1 = tf.keras.layers.Dense(2048, activation='leaky_relu')(deep_input_layer) # call the amount of nodes in each hidden layer, the activation function, and the previous layer
deep_layer_1 = tf.keras.layers.Dropout(dr)(deep_layer_1)  # Add a dropout layer with a dropout rate 
deep_layer_2 = tf.keras.layers.Dense(1024, activation='leaky_relu')(deep_layer_1)
deep_layer_2 = tf.keras.layers.Dropout(dr)(deep_layer_2)  # Add a dropout layer with a dropout rate 
deep_layer_3 = tf.keras.layers.Dense(512, activation='leaky_relu')(deep_layer_2)
deep_layer_3 = tf.keras.layers.Dropout(dr)(deep_layer_3)  # Add a dropout layer with a dropout rate 
deep_layer_4 = tf.keras.layers.Dense(256, activation='leaky_relu')(deep_layer_3)
deep_layer_4 = tf.keras.layers.Dropout(dr)(deep_layer_4)  # Add a dropout layer with a dropout rate 
deep_layer_5 = tf.keras.layers.Dense(128, activation='leaky_relu')(deep_layer_4)
deep_layer_5 = tf.keras.layers.Dropout(dr)(deep_layer_5)  # Add a dropout layer with a dropout rate 
deep_layer_6 = tf.keras.layers.Dense(64, activation='leaky_relu')(deep_layer_5)
deep_layer_6 = tf.keras.layers.Dropout(dr)(deep_layer_6)  # Add a dropout layer with a dropout rate 
deep_layer_7 = tf.keras.layers.Dense(32, activation='leaky_relu')(deep_layer_6)
deep_layer_7 = tf.keras.layers.Dropout(dr)(deep_layer_7)  # Add a dropout layer with a dropout rate 
deep_layer_8 = tf.keras.layers.Dense(16, activation='leaky_relu')(deep_layer_7)
deep_layer_8 = tf.keras.layers.Dropout(dr)(deep_layer_8)  
deep_layer_9 = tf.keras.layers.Dense(8, activation='leaky_relu')(deep_layer_8)
deep_layer_9 = tf.keras.layers.Dropout(dr)(deep_layer_9) 
deep_layer_10 = tf.keras.layers.Dense(4, activation='leaky_relu')(deep_layer_9)
deep_layer_10 = tf.keras.layers.Dropout(dr)(deep_layer_10) 
deep_layer_11 = tf.keras.layers.Dense(2, activation='leaky_relu')(deep_layer_10)


# Concatenate the wide and deep parts of the network (last layers of each)
concatenated = tf.keras.layers.Concatenate()([wide_input_layer, deep_layer_11]) # if you want to use the wide hidden layers make sure to call the last layer here else leave as be

# Create the output layer
output_layer = tf.keras.layers.Dense(1)(concatenated)


# Create the combined model
model = tf.keras.Model(inputs=[wide_input_layer, deep_input_layer], outputs=output_layer)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) 

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=mae_loss, metrics=['mae', 'mse'],weighted_metrics=[])

# Split the data into training and testing sets
split_size = 0.2  # Tunable parameter: fraction of the dataset used for testing

# Split the data into training and testing sets
train_wide, test_wide, train_deep, test_deep, train_labels, test_labels = train_test_split(
    cat_encoded, numerical_features_scaled, labels, test_size=split_size, random_state=22
)

# Define the Early Stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',  # Monitor  loss
    patience=50,  # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True  # Restore the model weights from the epoch with the best validation loss
)

#threshold = 10
#sample_weights = np.where(train_labels >= threshold, 10.0, 1.0)  # Assign higher weight to values >= 40

threshold_below = 10 #100
threshold_above = 40 #1000

conditions = [
    train_labels < threshold_below,
    (train_labels >= threshold_below) & (train_labels <= threshold_above),
    train_labels > threshold_above
]

weights = [
    1.0,  # Weight for values below 10
    1.0,  # Weight for values between 10 and 40
    1.0  # Weight for values above 40
]

sample_weights = np.select(conditions, weights, default=1.0)

# Train the model
history = model.fit([train_wide, train_deep], 
                    train_labels, 
                    epochs=100000, 
                    batch_size=1024, #256
                    validation_split=0.2,
                    sample_weight=sample_weights, 
                    callbacks=[early_stopping], 
                    shuffle=True)

# Evaluate the model on the test set
test_loss, test_mae,test_mse = model.evaluate([test_wide, test_deep], test_labels)

# Print test loss and mean absolute error
print("Test Loss:", test_loss)
print("Test MAE:", test_mae)

# Predictions on the test set
predictions = model.predict([test_wide, test_deep])

# Create a 1x2 grid for subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# First subplot: Predictions vs. Actual Values with log scales
axes[0].scatter(test_labels, predictions, alpha=0.5)
axes[0].plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], linestyle='--', color='red', linewidth=2, label='1.0x Actual Values')  # One-to-one line
axes[0].set_xlabel('Actual Values (log scale)')
axes[0].set_ylabel('Predicted Values (log scale)')
axes[0].set_title('Predictions vs. Actual Values')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].grid(True)
axes[0].legend()

# Second subplot: Training and Validation Loss
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

# Display the figure
plt.tight_layout()
plt.show()

