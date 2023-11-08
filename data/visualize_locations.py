import pandas as pd
import plotly.express as px

# Read the data
data = pd.read_csv('train.csv')  # Replace with your CSV file path

# Create a simple scatter plot on a map with Plotly Express
fig = px.scatter_geo(data,
                     lat='lat',
                     lon='lon',
                     projection="natural earth")

# Show the figure
fig.show()

