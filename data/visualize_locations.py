import pandas as pd
import plotly.express as px

data = pd.read_csv('train.csv') 
fig = px.scatter_geo(data,lat='lat',lon='lon',projection="natural earth")

fig.show()

