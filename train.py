# INPUTS THE CSV FILE AND TRAINS

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
csv_file = "/content/output.csv"
df = pd.read_csv(csv_file)

# Separate features and labels
X = df[["Shear Length", "Crack Length", "Number of Cracks", "Color Variation"]].values
y = df["Label"].values  # 1 = Welded, 0 = Non-Welded

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training & testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
model.save("welding_classifier_from_csv.h5")
print("Model saved as welding_classifier_from_csv.h5")
