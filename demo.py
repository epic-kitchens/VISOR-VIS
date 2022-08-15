#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:04:58 2022

@author: Ahmad Darkhalil
"""
from vis import *
import os

json_files_path = '../json_files'
output_directory ='../outputs'
output_resolution= (854,480)
is_overlay=False
rgb_frames = '../Images'
generate_video=True

folder_of_jsons_to_masks(json_files_path, output_directory,is_overlay=is_overlay,rgb_frames=rgb_frames,output_resolution=output_resolution,generate_video=generate_video)
