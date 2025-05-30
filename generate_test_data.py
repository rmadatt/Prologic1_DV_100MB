import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate dataset
n_rows = 15000  # Adjust to target ~2 MB
data = {
    'ID': range(1, n_rows + 1),
    'Age': np.random.randint(18, 80, n_rows),
    'Income': np.random.normal(50000, 15000, n_rows).round(2),
    'Score': np.random.uniform(0, 100, n_rows).round(2),
    'Hours_Worked': np.random.normal(40, 10, n_rows).round(2),
    'Department': np.random.choice(['Sales', 'Tech', 'HR', 'Marketing'], n_rows),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], n_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Introduce some anomalies for testing
df.loc[np.random.choice(n_rows, 100), 'Income'] = np.random.uniform(100000, 150000, 100)  # High income outliers
df.loc[np.random.choice(n_rows, 50), 'Hours_Worked'] = np.random.uniform(80, 100, 50)  # Extreme hours

# Save to CSV
output_file = 'test_data.csv'
df.to_csv(output_file, index=False)

# Check file size
file_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
print(f"Generated CSV file: {output_file}, Size: {file_size:.2f} MB")
