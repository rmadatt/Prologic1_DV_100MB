import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate dataset
n_rows = 35000  # Adjusted to target ~5 MB
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
df.loc[np.random.choice(n_rows, 200), 'Income'] = np.random.uniform(100000, 150000, 200)  # High income outliers
df.loc[np.random.choice(n_rows, 100), 'Hours_Worked'] = np.random.uniform(80, 100, 100)  # Extreme hours

# Save to CSV
output_file = "C:/Users/nuraa/Documents/test_data_5mb.csv"
df.to_csv(output_file, index=False)
print(f"CSV file saved at: {output_file}")
