import matplotlib.pyplot as plt
import numpy as np

# Sample data: different methods and their respective accuracy values
methods = ['Stable Diffusion', 'Raw Data', 'Gaining', 'Data Augmentation']
accuracies = [0.30, 0.58, 0.71, 0.85]

# Create a bar plot
plt.bar(methods, accuracies, color='skyblue')

# Add accuracy labels above each bar
for i in range(len(methods)):
    plt.text(i, accuracies[i] + 0.01, f'{accuracies[i]:.2f}', ha='center', va='bottom')

plt.ylim(0, 1)

# Label the chart
plt.xlabel('Data Refining Methods')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Data Refining Methods')

# Show the plot
plt.show()
