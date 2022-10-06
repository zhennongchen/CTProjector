#!/usr/bin/env python

# %%
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import SimpleITK as sitk
import time
import os
import transformation as transform
import functions_collection as ff
import glob as gb
import nibabel as nb

import ct_projector.projector.cupy as ct_projector
import ct_projector.projector.cupy.fan_equiangular as ct_fan
import ct_projector.projector.numpy as numpy_projector
import ct_projector.projector.numpy.fan_equiangluar as numpy_fan

# %%
# define random range:
max_t = 5 #unit = mm
max_r = 15
total_view = 2400
gantry_rotation_time = 500 #unit ms
time_range = [250,400] # unit ms

# %%
# define the patient list
patient_list = ff.find_all_target_files(['example_CT_volume/*'],'/workspace/CTProjector/data')
print(patient_list)

# %%
for p in patient_list:
    patient_id = os.path.basename(p)
    print(patient_id)
    
    ct = sitk.ReadImage(p)
    spacing_raw = ct.GetSpacing()
    img_raw = sitk.GetArrayFromImage(ct)

    img = (img_raw.astype(np.float32) + 1024) / 1000 * 0.019
    img[img < 0] = 0
    img = img[::-1, ...]
    spacing = np.array(spacing_raw[::-1])

    img = img[np.newaxis, ...]
    projector = ct_projector.ct_projector()
    projector.from_file('./projector_fan.cfg')
    projector.nx = img.shape[3]
    projector.ny = img.shape[2]
    projector.nz = 1
    projector.nv = 1
    projector.dx = spacing[2]
    projector.dy = spacing[1]
    projector.dz = spacing[0]
    projector.nview = total_view

    # very important: make sure that the arrays are saved in C order
    cp.cuda.Device(0).use()
    ct_projector.set_device(0)

    angles = projector.get_angles()
    print(angles.shape)

    # generate random motion
    for i in range(0,2):
        print('\n',i , 'random')
        translation,translation_mm,rotation,start_view,end_view,lasting_view,lasting_time = transform.generate_random_3D_motion(5,15,spacing,time_range = time_range,total_view_num = total_view, gantry_rotation_time = gantry_rotation_time)
        print(translation_mm,[i/np.pi * 180 for i in rotation],start_view,end_view,lasting_time)


    

