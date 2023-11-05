import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore

# Load your dataset
df = pd.read_csv('train.csv')

# Visualizing the range of features with no normalization
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title('No Normalization')
plt.show()

# Min-Max Normalization
min_max_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_minmax)
plt.title('Min-Max Normalization')
plt.show()

# Log Normalization - assuming all values are positive
# If you have zero or negative values, you need to shift the dataset before applying log
min_value = df.min().min()
# Calculate the shift needed to ensure all values are positive
shift = abs(min_value) + 1 if min_value < 0 else 0
# Shift the dataset
df_shifted = df + shift
# Apply log normalization
df_log = np.log(df_shifted)
# Now you can visualize it
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_log)
plt.title('Log Normalization with Shift')
plt.show()

df_log = np.log1p(df.clip(0)) # clip(0) ensures no value is less than 0
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_log)
plt.title('Log Normalization')
plt.show()

# Z-Score Normalization
df_zscore = df.apply(zscore)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_zscore)
plt.title('Z-Score Normalization')
plt.show()

