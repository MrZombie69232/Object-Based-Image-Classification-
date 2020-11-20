import numpy as np
import gdal
import ogr
from skimage import exposure
from skimage.segmentation import slic
import scipy
import time
from sklearn.ensemble import RandomForestClassifier

# read image file
naip_fn = "E://arc//sds//m_4308804_ne_16_060_20181017//clipped.TIF"


# getting band number , rows and columns to print
driverTiff = gdal.GetDriverByName("GTiff")
naip_ds = gdal.Open(naip_fn)
nbands = naip_ds.RasterCount
band_data = []
print('band', naip_ds.RasterCount, 'rows', naip_ds.RasterYSize, 'column', naip_ds.RasterXSize)

# stacking bands using a loop
for i in range(1,nbands+1):
    band = naip_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)

band_data = np.dstack(band_data)

# rescaling image (0-1)
img = exposure.rescale_intensity(band_data)

# starting segmentation
seg_start = time.time()
# segments = quickshift(img, convert2lab=False)
segments = slic(img, n_segments=10000, compactness=0.1)
print('segments complete', time.time() - seg_start)

# Spectral Properties of Segments
def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # in this case the variance = nan, change it 0.0
            band_stats[3] = 0.0
        features += band_stats
    return features

segment_ids = np.unique(segments)
objects = []
object_ids = []
for id in segment_ids:
    segment_pixels = img[segments == id]
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)

print('created', len(objects), 'objects with', len(objects[0]),'variables')
# save segments to raster
segments_fn = "E://arc//sds//seg//final_seg.TIF"
segments_ds = driverTiff.Create(segments_fn, naip_ds.RasterXSize, naip_ds.RasterYSize,
                                1, gdal.GDT_Float32)
segments_ds.SetGeoTransform(naip_ds.GetGeoTransform())
segments_ds.SetProjection(naip_ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds = None

# open the points file to use for training data
train_fn = ('E://arc//sds//seg//train.shp')
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()

# create a new raster layer in memory
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())

# rasterize the training points
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

# retrieve the rasterized data and print basic stats
data = target_ds.GetRasterBand(1).ReadAsArray()
print('min', data.min(), 'max', data.max(), 'mean', data.mean())

# Get segments representing each land cover classification type and ensure no segment represents more than one class
ground_truth = target_ds.GetRasterBand(1).ReadAsArray()

classes = np.unique(ground_truth)[1:]
print('class values', classes)

segments_per_class = {}

for klass in classes:
    segments_of_class = segments[ground_truth == klass]
    segments_per_class[klass] = set(segments_of_class)
    print("Training segments for class", klass, ":", len(segments_of_class))

intersection = set()
accum = set()

for class_segments in segments_per_class.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, "Segment(s) represent multiple classes"


train_img = np.copy(segments)
threshold = train_img.max()+1

for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label

train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

training_objects =[]
training_labels = []

# loop to assign values to training_objects and training_values
for klass in classes:
    class_train_objects = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_objects)
    training_objects += class_train_objects
    print('Training objects for class', klass, ':', len(class_train_objects))

# calling classifier for the prediction
classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects,training_labels)
print('Fitting Random Forest Classifier')
predicted = classifier.predict(objects)
print('Predicting Classifications')

# adding segments id and predicted data together
clf = np.copy(segments)
for segment_id,klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass

print('Prediction applied to numpy array')

# masking to show no data
mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0
clf = np.multiply(clf, mask)
clf[clf <0] = -9999.0

print('Saving Classification to raster with gdal')

# saving data to view in Qgis
clfds = driverTiff.Create('E://arc//Image_naip//classified.TIF', naip_ds.RasterXSize, naip_ds.RasterYSize,
                          1, gdal.GDT_Float32)

# set transformation and projection
clfds.SetGeoTransform(naip_ds.GetGeoTransform())
clfds.SetProjection(naip_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None

print('Done!')



