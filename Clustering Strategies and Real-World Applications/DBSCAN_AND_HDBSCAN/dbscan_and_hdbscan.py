#Comparing DBSCAN and HDBSCAN clustering

# Density-Based Spatial Clustering of Applications with Noise  (DBSCAN)
#Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler

# geographical tools
import geopandas as gpd  # pandas dataframe-like geodataframes for geographical data
import contextily as ctx  # used for obtianing a basemap of Canada
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')


# Introduction
# In this lab, you'll create two clustering models using data curated by StatCan containing the names, types, and locations of cultural and art facilities across Canada. 
# We'll focus on the museum locations provided across Canada.

# Data source: The Open Database of Cultural and Art Facilities (ODCAF)
# A collection of open data containing the names, types, and locations of cultural and art facilities across Canada. It is released under the Open Government License - Canada. 
# The different types of facilities are labeled under 'ODCAF_Facility_Type'.

# Landing page:
# https://www.statcan.gc.ca/en/lode/databases/odcaf

# link to zip file:
# https://www150.statcan.gc.ca/n1/en/pub/21-26-0001/2020001/ODCAF_V1.0.zip?st=brOCT3Ry




import requests
import zipfile
import io
import os

# URL of the ZIP file on the cloud server
zip_file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip'

# Directory to save the extracted TIFF file
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Download the ZIP file
response = requests.get(zip_file_url)
response.raise_for_status()  # Ensure the request was successful
# Step 2: Open the ZIP file in memory
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    # Step 3: Iterate over the files in the ZIP
    for file_name in zip_ref.namelist():
        if file_name.endswith('.tif'):  # Check if it's a TIFF file
            # Step 4: Extract the TIFF file
            # zip_ref.extract(file_name, output_dir)
            print(f"Downloaded and extracted: {file_name}")







# Include a plotting function
# The code for a helper function is provided to help you plot your results. 
# Although you don't need to worry about the details, it's quite instructive as it uses a geopandas dataframe and a basemap to plot coloured cluster points on a map of Canada.

def plot_clustered_locations(df, title='Museums Clustered by Proximity'):
    # Koordinatları GeoDataFrame formatına çeviririk
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)  # Xəritə ilə uyğun sistemə çeviririk

    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Səs-küy (noise) və klasterləşdirilmiş nöqtələri ayırırıq
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]
    
    # Səs-küy nöqtələrini qırmızı haşiyəli qara rəngdə göstəririk
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')
    
    # Klaster nöqtələri fərqli rənglərlə göstərilir
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)
    
    # Kanada xəritəsi qatını əlavə edirik
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)
    
    # Qrafikin başlıqları və dizaynı
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()









# Explore the data and extract what you need from it
# Start by loading the data set into a Pandas DataFrame and displaying the first few rows.

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding = "ISO-8859-1")

df.head()




#Exercise 1. Explore the table. What do missing values look like in this data set?
# Strings consisting of two dots '..' indicate missing values. There miight still be empty fields, or NaNs.print("Unique values indicating missing data in each 



#Exercise 2. Display the facility types and their counts.
df.ODCAF_Facility_Type.value_counts()





#Exercise 3. Filter the data to only include museums.
# Check your results. Did you get as many as you expected?

df = df[df.ODCAF_Facility_Type == 'museum']
df.ODCAF_Facility_Type.value_counts()




#Exercise 4. Select only the Latitude and Longitude features as inputs to our clustering problem.
# Also, display information about the coordinates like counts and data types.
df = df[['Latitude', 'Longitude']]
df.info()




# Exercise 5. We'll need these coordinates to be floats, not objects.
# Remove any museums that don't have coordinates, and convert the remaining coordinates to floats.

# Remove observations with no coordinates 
df = df[df.Latitude!='..']

# Convert to float
df[['Latitude','Longitude']] = df[['Latitude','Longitude']].astype('float')











# Build a DBSCAN model
# Correctly scale the coordinates for DBSCAN (since DBSCAN is sensitive to scale)


# In this case we know how to scale the coordinates. Using standardization would be an error becaues we aren't using the full range of the lat/lng coordinates.
# Since latitude has a range of +/- 90 degrees and longitude ranges from 0 to 360 degrees, the correct scaling is to double the longitude coordinates (or half the Latitudes)
coords_scaled = df.copy()
coords_scaled["Latitude"] = 2*coords_scaled["Latitude"]



#Apply DBSCAN with Euclidean distance to the scaled coordinates
# In this case, reasonable neighbourhood parameters are already chosen for you. Feel free to experiment.

min_samples=3 # minimum number of samples needed to form a neighbourhood
eps=1.0 # neighbourhood search radius
metric='euclidean' # distance measure 

dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(coords_scaled)


# Add cluster labels to the DataFrame
df['Cluster'] = dbscan.fit_predict(coords_scaled)  # Assign the cluster labels

# Display the size of each cluster
df['Cluster'].value_counts()

#As you can see, there are two relatively large clusters and 79 points labelled as noise (-1).

# Plot the museums on a basemap of Canada, colored by cluster label.


plot_clustered_locations(df, title='Museums Clustered by Proximity')



# What do you see?
# What size is the smallest cluster?
# Do you think the clusters make sense in terms of what you expect to see?
# Do you think there should be more clusters in some regions? Why?


# One key thing to notice here is that the clusters are not uniformly dense.

# For example, the points are quite densely packed in a few regions but are relatively sparse in between.

# DBSCAN agglomerates neighboring clusters together when they are close enough.

# Let's see how a hierarchical density-based clustering algorithm like HDBSCAN performs.

















# Build an HDBSCAN clustering model
# At this stage, you've already loaded your data and extracted the museum coordinates into a dataframe, df.

# You've also stored properly scaled coordinates as the 'coords_scaled' array.

# All that remains is to:

# Fit and transform HDBSCAN to your scaled coordinates
# Extract the cluster labels
# Plot the results on the same basemap as before
# Reasonable HDBSCAN parameters have been selected for you to start with.

# Initialize an HDBSCAN model


min_samples=None
min_cluster_size=3
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean')  # You can adjust parameters as needed






#Exercise 6. Assign the cluster labels to your unscaled coordinate dataframe and display the counts of each cluster label
df['Cluster'] = hdb.fit_predict(coords_scaled)  # Another way to assign the labels

# Display the size of each cluster
df['Cluster'].value_counts()
#As you can see, unlike the case for DBSCAN, clusters quite uniformly sized, although there is a quite lot of noise identified.




#Exercise 7. Plot the hierarchically clustered museums on a basemap of Canada, colored by cluster label.
plot_clustered_locations(df, title='Museums Hierarchically Clustered by Proximity')




# Closing remarks
# Take a close look at the map.

# What's different about these results compared to DBSCAN?
# It might seem like there are more points identified as noise, but is that the case?
# Can you see the variations in density that HDBSCAN captures?
# In practice, you would want to investigate much deeper but at least you get the idea here.