# import pandas as pd
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # Read the data
# data = pd.read_csv('train.csv')  # Replace with your CSV file path

# # Create a figure with an appropriate size
# fig = plt.figure(figsize=(10, 5))

# # Create a GeoAxes in the tile's projection
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# # Set the extent (min longitude, max longitude, min latitude, max latitude)
# # You can also use data['longitude'].min(), data['longitude'].max(), etc.
# ax.set_extent([-180, 180, -90, 90])

# # Add natural earth features to the map with cartopy.feature
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')

# # Plot the locations as red dots
# plt.scatter(data['lon'], data['lat'], color='red', s=10, transform=ccrs.Geodetic(), alpha=0.6)

# # Add gridlines
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# # Show the plot
# plt.show()

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

