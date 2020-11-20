import numpy as np
import gdal
import ogr
from sklearn import metrics

# read image file
naip_fn = "E://arc//sds//m_4308804_ne_16_060_20181017//clipped.TIF"
driverTiff = gdal.GetDriverByName('Gtiff')
naip_ds = gdal.Open(naip_fn)

# open the points file to use for training data
test_fn = ('E://arc//sds//seg//test.shp')
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()

# create a new raster layer in memory
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())

# rasterize the test points
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

truth = target_ds.GetRasterBand(1).ReadAsArray()

pred_ds = gdal.Open('E://arc//Image_naip//classified.TIF')
pred = pred_ds.GetRasterBand(1).ReadAsArray()

idx = np.nonzero(truth)

cm = metrics.confusion_matrix(truth[idx], pred[idx])

# pixel accuracy
print(cm)

print(cm.diagonal())
print(cm.sum(axis=0))

accuracy = cm.diagonal() / cm.sum(axis=0)
print(accuracy)

