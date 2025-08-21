# Install TensorFlow if not already installed
!pip install tensorflow

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np

# Image Preprocessing
image_size = (150, 150)  # Image size (150x150 for input images)
batch_size = 32  # Batch size for training and validation

# Train Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values to [0, 1]
    rotation_range=20,         # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,     # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2,    # Randomly shift images vertically by up to 20%
    shear_range=0.2,           # Shear transformation
    zoom_range=0.2,            # Random zoom
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest'        # Fill missing pixels after transformation
)

# Validation Data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling, no augmentation

# Train Data Generator with Augmentation
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/minor4/dataset/train',   # Path to training data
    target_size=image_size,                         # Resize all images to 150x150
    batch_size=batch_size,                          # Batch size
    class_mode='binary'                             # Binary classification (0 or 1)
)

# Validation Data Generator (no augmentation, only rescaling)
validation_generator = validation_datagen.flow_from_directory(
    '/content/drive/MyDrive/minor4/dataset/validation',   # Path to validation data
    target_size=image_size,                            # Resize all images to 150x150
    batch_size=batch_size,                             # Batch size
    class_mode='binary'                                # Binary classification
)

# Example: Additional numerical features (shear length, crack length, number of cracks, color variation)
num_features = 4  # Modify based on actual number of parameters

# --- Image Input (CNN Model) ---
image_input = Input(shape=(150, 150, 3), name="image_input")

x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

# --- Numerical Input ---
numerical_input = Input(shape=(num_features,), name="numerical_input")

y = Dense(64, activation='relu')(numerical_input)
y = Dense(32, activation='relu')(y)

# --- Combine Both Inputs ---
combined = concatenate([x, y])
output = Dense(1, activation='sigmoid', name="output")(combined)  # Binary classification

# Create Model
model = Model(inputs=[image_input, numerical_input], outputs=output)

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print Model Summary
model.summary()

# Dummy Numerical Data (Replace with actual feature values)
num_samples = len(train_generator.filenames)
train_numerical_data = np.random.rand(num_samples, num_features)
validation_numerical_data = np.random.rand(len(validation_generator.filenames), num_features)

# Extract Image Data
def data_generator(image_gen, numerical_data):
    while True:
        image_batch, labels = next(image_gen)
        batch_size = image_batch.shape[0]
        yield ((image_batch, numerical_data[:batch_size]), labels)

# Specify output signature for tf.data.Dataset
output_signature = (
    (
        tf.TensorSpec(shape=(None, 150, 150, 3), dtype=tf.float32),  # Image input
        tf.TensorSpec(shape=(None, num_features), dtype=tf.float32) # Numerical input
    ),
    tf.TensorSpec(shape=(None,), dtype=tf.float32)  # Labels
)

# Create tf.data.Dataset objects
# Pass the data_generator function using lambda to make it callable
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_generator, train_numerical_data),
    output_signature=output_signature
)
val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(validation_generator, validation_numerical_data),
    output_signature=output_signature
)

# Train the Model
model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_data=val_dataset,
    validation_steps=len(validation_generator)
)

# Save the model
model.save('welding_classifier_multi_input.h5')
