# AI Model Selection using Iris Dataset

import pandas as pd

# Step 1: Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Step 2: Define column names
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Step 3: Read dataset
data = pd.read_csv(url, names=columns)

# Step 4: Display first few rows (optional)
print("Dataset Preview:")
print(data.head())

# Step 5: Check if label column exists
if 'species' in data.columns:
    print("\nSupervised Learning Detected")
    print("Selected Model: Classification Model")
else:
    print("\nUnsupervised Learning Detected")
    print("Selected Model: Clustering Model")

# Step 6: End
print("\nProgram Executed Successfully")
