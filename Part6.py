import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from a CSV file
df = pd.read_csv(r'single_family_home_values.csv')  # Replace with your file path

# Set display options for Pandas
desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

# Select only the numeric columns for correlation. This list must contain only numeric columns.
numeric_columns = ['bedrooms', 'bathrooms', 'rooms', 'squareFootage', 'lotSize', 'yearBuilt', 'priorSaleAmount']
df_numeric = df[numeric_columns]

# Convert all columns to numeric, coercing errors into NaN, then fill NaN with zeros
df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)

# Calculate the correlation matrix
correlation_matrix = df_numeric.corr()

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True)
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Print the correlation matrix
print(correlation_matrix)

# Print the covariance matrix (optional)
covariance_matrix = df_numeric.cov()
print(covariance_matrix)
