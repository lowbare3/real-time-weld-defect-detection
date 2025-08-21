# GRAPH GENERATION

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file with extracted features
csv_file = "/content/output.csv"  # Adjust the path if needed
df = pd.read_csv(csv_file)

# Display first few rows of the dataset
print(df.head())

# --- 1. Distribution of Features by Category (Welded vs Non-Welded) ---

# Set Seaborn style
sns.set(style="whitegrid")

# Plot: Shear Length
plt.figure(figsize=(8, 6))
sns.boxplot(x='Label', y='Shear Length', data=df, palette="Set2")
plt.title('Shear Length Distribution (Welded vs Non-Welded)')
plt.xlabel('Category (0: Non-Welded, 1: Welded)')
plt.ylabel('Shear Length')
plt.xticks([0, 1], ['Non-Welded', 'Welded'])
plt.show()

# Plot: Crack Length
plt.figure(figsize=(8, 6))
sns.boxplot(x='Label', y='Crack Length', data=df, palette="Set2")
plt.title('Crack Length Distribution (Welded vs Non-Welded)')
plt.xlabel('Category (0: Non-Welded, 1: Welded)')
plt.ylabel('Crack Length')
plt.xticks([0, 1], ['Non-Welded', 'Welded'])
plt.show()

# Plot: Number of Cracks
plt.figure(figsize=(8, 6))
sns.boxplot(x='Label', y='Number of Cracks', data=df, palette="Set2")
plt.title('Number of Cracks Distribution (Welded vs Non-Welded)')
plt.xlabel('Category (0: Non-Welded, 1: Welded)')
plt.ylabel('Number of Cracks')
plt.xticks([0, 1], ['Non-Welded', 'Welded'])
plt.show()

# Plot: Color Variation
plt.figure(figsize=(8, 6))
sns.boxplot(x='Label', y='Color Variation', data=df, palette="Set2")
plt.title('Color Variation Distribution (Welded vs Non-Welded)')
plt.xlabel('Category (0: Non-Welded, 1: Welded)')
plt.ylabel('Color Variation')
plt.xticks([0, 1], ['Non-Welded', 'Welded'])
plt.show()

# --- 2. Pairplot to Compare All Features Together ---

# We can plot a pairplot for all the numerical features and categorize by 'Label' (Welded vs Non-Welded)
sns.pairplot(df, hue='Label', vars=["Shear Length", "Crack Length", "Number of Cracks", "Color Variation"], palette="Set2")
plt.suptitle('Pairplot of Features by Category (Welded vs Non-Welded)', y=1.02)
plt.show()

# --- 3. Correlation Heatmap to Analyze Relationships Between Features ---

# Calculate correlation matrix
correlation_matrix = df[["Shear Length", "Crack Length", "Number of Cracks", "Color Variation"]].corr()

# Plot a heatmap of the correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()

# --- 4. Feature Comparison (Optional) ---

# You can also compare specific features like shear length or crack length between welded and non-welded samples by calculating their means or plotting histograms.

# Compare average shear length between welded and non-welded
avg_shear_length = df.groupby('Label')['Shear Length'].mean()
print(f"Average Shear Length: \n{avg_shear_length}")

# Compare average number of cracks between welded and non-welded
avg_num_cracks = df.groupby('Label')['Number of Cracks'].mean()
print(f"Average Number of Cracks: \n{avg_num_cracks}")

