#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 20:55:35 2019

@author: rahul2021
"""

import os

DATA_DIR = "../../DATA/MINNEAPOLIS/"
LANDSAT_DIR = os.path.join(DATA_DIR, "LANDSAT")
SENTINEL_DIR = os.path.join(DATA_DIR, "SENTINEL")

with open("clip.sh", "w") as f:
    sentinel_tile = os.path.join(SENTINEL_DIR,[file for file in os.listdir(SENTINEL_DIR) if file.endswith(".tif")][0])
    command = "gdaltindex MINNEAPOLIS_sentinel.shp " + sentinel_tile + "\n"
    f.write(command)
    landsat_file_lists = sorted([file for file in os.listdir(LANDSAT_DIR) if file.endswith(".TIF")])
    for file in landsat_file_lists:
        in_file = os.path.join(LANDSAT_DIR, file)
        out_file = os.path.join(LANDSAT_DIR, file.split('.')[0].split("_")[-1]+".tif")
        command = "gdalwarp -cutline MINNEAPOLIS_sentinel.shp -crop_to_cutline -ts 3660 3660 " + in_file + " " + out_file + "\n"
        f.write(command)
# %%
DATA_DIR = "../../DATA/MADRID/"
LANDSAT_DIR = os.path.join(DATA_DIR, "LANDSAT")
SENTINEL_DIR = os.path.join(DATA_DIR, "SENTINEL")

with open("clip.sh", "a") as f:
    sentinel_tile = os.path.join(SENTINEL_DIR,[file for file in os.listdir(SENTINEL_DIR) if file.endswith(".tif")][0])
    command = "gdaltindex MADRID_sentinel.shp " + sentinel_tile + "\n"
    f.write(command)
    landsat_file_lists = sorted([file for file in os.listdir(LANDSAT_DIR) if file.endswith(".TIF")])
    for file in landsat_file_lists:
        in_file = os.path.join(LANDSAT_DIR, file)
        out_file = os.path.join(LANDSAT_DIR, file.split('.')[0].split("_")[-1]+".tif")
        command = "gdalwarp -cutline MADRID_sentinel.shp -crop_to_cutline -ts 3660 3660 " + in_file + " " + out_file + "\n"
        f.write(command)
# %%
DATA_DIR = "../../DATA/ROME/"
LANDSAT_DIR = os.path.join(DATA_DIR, "LANDSAT")
SENTINEL_DIR = os.path.join(DATA_DIR, "SENTINEL")

with open("clip.sh", "a") as f:
    sentinel_tile = os.path.join(SENTINEL_DIR,[file for file in os.listdir(SENTINEL_DIR) if file.endswith(".tif")][0])
    command = "gdaltindex ROME_sentinel.shp " + sentinel_tile + "\n"
    f.write(command)
    landsat_file_lists = sorted([file for file in os.listdir(LANDSAT_DIR) if file.endswith(".TIF")])
    for file in landsat_file_lists:
        in_file = os.path.join(LANDSAT_DIR, file)
        reprojected_file = os.path.join(LANDSAT_DIR, file[:-4]+"_reprojected"+file[-4:])
        command = "gdalwarp -t_srs ROME_sentinel.prj " + in_file + " " + reprojected_file + "\n"
        f.write(command)
        out_file = os.path.join(LANDSAT_DIR, file.split('.')[0].split("_")[-1]+".tif")
        command = "gdalwarp -cutline ROME_sentinel.shp -crop_to_cutline -ts 3660 3660 " + reprojected_file + " " + out_file + "\n"
        f.write(command)

