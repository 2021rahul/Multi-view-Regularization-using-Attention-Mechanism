#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:17:32 2019

@author: ghosh128
"""

import os
from PIL import Image
import numpy as np
from scipy import io
from osgeo import gdal, gdalconst, osr
#%%
REGION = "MADRID"
DATA_DIR = "../../DATA/"+REGION
LANDSAT_DIR = os.path.join(DATA_DIR, "LANDSAT")
OSM_DIR = os.path.join(DATA_DIR, "OSM")
OUT_DIR = os.path.join(DATA_DIR, "NUMPY")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
im_size = (3660,3660)
n_channel = 11
#%%
im = Image.open(os.path.join(LANDSAT_DIR, "B1.tif"))
B1 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B2.tif"))
B2 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B3.tif"))
B3 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B4.tif"))
B4 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B5.tif"))
B5 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B6.tif"))
B6 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B7.tif"))
B7 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B8.tif"))
B8 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B9.tif"))
B9 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B10.tif"))
B10 = np.array(im.resize(im_size))

im = Image.open(os.path.join(LANDSAT_DIR, "B11.tif"))
B11 = np.array(im.resize(im_size))

data = np.stack((B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11), axis=-1)
data = data.astype(np.float32)
for channel in range(n_channel):
    data[:,:,channel] = (data[:,:,channel]-np.mean(data[:,:,channel]))/np.std(data[:,:,channel])
np.save(os.path.join(OUT_DIR, "LANDSAT_data"), data)
# %%
#osm_im_array = io.loadmat(os.path.join(OSM_DIR, "LANDSAT.mat"))["gt"]
#tif_with_meta = gdal.Open(os.path.join(OSM_DIR, 'LANDSAT.tif'), gdalconst.GA_ReadOnly)
#gt = tif_with_meta.GetGeoTransform()
#driver = gdal.GetDriverByName("GTiff")
#dest = driver.Create(os.path.join(OSM_DIR, "LANDSAT_label.tif"), 3660, 3660, 1, gdal.GDT_UInt16)
#dest.GetRasterBand(1).WriteArray(osm_im_array)
#dest.SetGeoTransform(gt)
#wkt = tif_with_meta.GetProjection()
#srs = osr.SpatialReference()
#srs.ImportFromWkt(wkt)
#dest.SetProjection(srs.ExportToWkt())
#dest = None
