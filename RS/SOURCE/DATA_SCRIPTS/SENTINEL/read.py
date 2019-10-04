#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:12:34 2019

@author: ghosh128
"""

import os
from PIL import Image
import numpy as np
from scipy import io
from osgeo import gdal, gdalconst, osr
#%%
DATA_DIR = "../../DATA/MINNEAPOLIS"
SENTINEL_DIR = os.path.join(DATA_DIR, "SENTINEL")
OSM_DIR = os.path.join(DATA_DIR, "OSM")
OUT_DIR = os.path.join(DATA_DIR, "NUMPY")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
im_size = (10980,10980)
n_channel = 12
#%%
im = Image.open(os.path.join(SENTINEL_DIR, "B01.jp2"))
B1 = np.array(im.resize(im_size))
print("B1")

im = Image.open(os.path.join(SENTINEL_DIR, "B02.jp2"))
B2 = np.array(im.resize(im_size))
print("B2")

im = Image.open(os.path.join(SENTINEL_DIR, "B03.jp2"))
B3 = np.array(im.resize(im_size))
print("B3")

im = Image.open(os.path.join(SENTINEL_DIR, "B04.jp2"))
B4 = np.array(im.resize(im_size))
print("B4")

im = Image.open(os.path.join(SENTINEL_DIR, "B05.jp2"))
B5 = np.array(im.resize(im_size))
print("B5")

im = Image.open(os.path.join(SENTINEL_DIR, "B06.jp2"))
B6 = np.array(im.resize(im_size))
print("B6")

im = Image.open(os.path.join(SENTINEL_DIR, "B07.jp2"))
B7 = np.array(im.resize(im_size))
print("B7")

im = Image.open(os.path.join(SENTINEL_DIR, "B08.jp2"))
B8 = np.array(im.resize(im_size))
print("B8")

im = Image.open(os.path.join(SENTINEL_DIR, "B8A.jp2"))
B8A = np.array(im.resize(im_size))
print("B8A")

im = Image.open(os.path.join(SENTINEL_DIR, "B09.jp2"))
B9 = np.array(im.resize(im_size))
print("B9")

im = Image.open(os.path.join(SENTINEL_DIR, "B11.jp2"))
B11 = np.array(im.resize(im_size))
print("B11")

im = Image.open(os.path.join(SENTINEL_DIR, "B12.jp2"))
B12 = np.array(im.resize(im_size))
print("B12")

data = np.stack((B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12), axis=-1)
data = data.astype(np.float32)

for channel in range(n_channel):
    data[:,:,channel] = (data[:,:,channel]-np.mean(data[:,:,channel]))/np.std(data[:,:,channel])
np.save(os.path.join(OUT_DIR, "SENTINEL_data"), data)
#%%
osm_im_array = io.loadmat(os.path.join(OSM_DIR, "SENTINEL.mat"))["gt"]
tif_with_meta = gdal.Open(os.path.join(OSM_DIR, 'SENTINEL.tif'), gdalconst.GA_ReadOnly)
gt = tif_with_meta.GetGeoTransform()
driver = gdal.GetDriverByName("GTiff")
dest = driver.Create(os.path.join(OSM_DIR, "SENTINEL_label.tif"), 10980, 10980, 1, gdal.GDT_UInt16)
dest.GetRasterBand(1).WriteArray(osm_im_array)
dest.SetGeoTransform(gt)
wkt = tif_with_meta.GetProjection()
srs = osr.SpatialReference()
srs.ImportFromWkt(wkt)
dest.SetProjection(srs.ExportToWkt())
dest = None
