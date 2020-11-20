import gdal
import ogr
import numpy as np
import geopandas as gpd
import pandas as pd

# read shapefile to geopandas geodataframe
gdf = gpd.read_file('E://arc//sds//seg//truth_data_subset_utm12.shp')

# get names of land cover classes/labels
class_names = gdf['lctype'].unique()
print('class_name', class_names)

# create a unique id (integer) for each land cover class/label
class_ids =np.arange(class_names.size)+1
print('class ids',class_ids)

# create a pandas data frame of the labels and ids and save to cs0\
df = pd.DataFrame({'lctype': class_names, 'id': class_ids})
df.to_csv('E://arc//sds//seg//class_lookup.csv')
print('gdf without ids', gdf.head())
# add a new column to geodatafame with the id for each class/label
gdf['id'] = gdf['lctype'].map(dict(zip(class_names, class_ids)))
print('gdf with ids', gdf.head())

# split the truth data into training and test data sets and save each to a new shapefile
gdf_train = gdf.sample(frac=0.7)
gdf_test = gdf.drop(gdf_train.index)
print('gdf shape', gdf.shape, 'training shape', gdf_train.shape, 'test', gdf_test.shape)
gdf_train.to_file('E://arc//sds//seg//train.shp')
gdf_test.to_file('E://arc//sds//seg//test.shp')
