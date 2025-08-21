# CONVERTS IMAGE FEATURES IN CSV FILE


import cv2
import numpy as np
import os
import pandas as pd
from skimage.measure import label

# Define dataset paths
dataset_path = "/content/drive/MyDrive/minor4/dataset/train"  # Change to validation if needed
output_csv = "/content/output.csv"

# Function to extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 1. Shear Length & Crack Length (Edge Detection) ---
    edges = cv2.Canny(image_gray, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shear_length = 0
    crack_length = 0
    for contour in contours:
        length = cv2.arcLength(contour, closed=False)
        if length > 50:  # Adjust threshold as needed
            shear_length += length
        else:
            crack_length += length

    # --- 2. Count Number of Cracks (Connected Components) ---
    labeled_cracks = label(edges > 0)  # Identify crack regions
    num_cracks = np.max(labeled_cracks)  # Count distinct cracks

    # --- 3. Color Variation (Standard Deviation) ---
    color_variation = np.std(image)  # Measure intensity variation

    return shear_length, crack_length, num_cracks, color_variation

# Initialize Data Storage
data = []

# Process each image in the dataset
# Verify if the categories 'welded' and 'non_welded' exist in your dataset_path
# If not, adjust these names accordingly
for category in ["welded", "non_welded"]:
    category_path = os.path.join(dataset_path, category)

    # Check if the category path exists before processing images in it
    if os.path.exists(category_path):
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            shear_length, crack_length, num_cracks, color_variation = extract_features(image_path)
            # Changed 'label' to 'image_label' to avoid conflict with skimage.measure.label
            image_label = 1 if category == "welded" else 0  # Assign binary labels

            # Append to dataset
            data.append([image_name, shear_length, crack_length, num_cracks, color_variation, image_label])
    else:
        print(f"Warning: Category folder '{category_path}' not found. Skipping...")

# Save as CSV
df = pd.DataFrame(data, columns=["Image Name", "Shear Length", "Crack Length", "Number of Cracks", "Color Variation", "Label"])
df.to_csv(output_csv, index=False)

print(f"Feature extraction completed! CSV saved at: {output_csv}")