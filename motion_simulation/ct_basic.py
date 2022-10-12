#!/usr/bin/env python

import ct_projector.projector.cupy as ct_projector
import ct_projector.projector.cupy.fan_equiangular as ct_fan
import ct_projector.projector.numpy as numpy_projector
import ct_projector.projector.numpy.fan_equiangluar as numpy_fan

import numpy as np
import cupy as cp
import SimpleITK as sitk
import os
import CTProjector.functions_collection as ff
import CTProjector.transformation as transform
import glob as gb
import nibabel as nb
from PIL import Image

# for sitk:
# def basic_image_processing(filename):
#     ct = sitk.ReadImage(filename)
#     spacing_raw = ct.GetSpacing()
#     img_raw = sitk.GetArrayFromImage(ct)

#     img = (img_raw.astype(np.float32) + 1024) / 1000 * 0.019
#     img[img < 0] = 0
#     img = img[::-1, ...]

#     spacing = np.array(spacing_raw[::-1])

#     return img,spacing


# for nibabel:
def basic_image_processing(filename):
    ct = nb.load(filename)
    spacing = ct.header.get_zooms()
    img = ct.get_fdata()
    
    img = (img.astype(np.float32) + 1024) / 1000 * 0.019
    img[img < 0] = 0
    img = np.rollaxis(img,-1,0)

    spacing = np.array(spacing[::-1])


    return img,spacing,ct.affine


def define_forward_projector(img,spacing,total_view):
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
    return projector


def backprojector(img,spacing,):
    fbp_projector = numpy_projector.ct_projector()
    fbp_projector.from_file('./projector_fan.cfg')
    fbp_projector.nx = img.shape[3]
    fbp_projector.ny = img.shape[2]
    fbp_projector.nz = 1
    fbp_projector.nv = 1
    fbp_projector.dx = spacing[2]
    fbp_projector.dy = spacing[1]
    fbp_projector.dz = spacing[0]
    return fbp_projector





def fp_w_motion(img, projector, angles, translation, rotation, start_view, end_view,slice_num, increment_raw):
    projection = np.zeros([slice_num[1] - slice_num[0],angles.shape[0],1,projector.nu])
    assert projection.shape[0] == 60

    t = 0
    transformation_doing = False
    transformation_done = False
    while True:
        if t + increment_raw >= start_view and transformation_doing == False and transformation_done == False:
            increment = start_view - t
            transformation_doing = True
        elif t + increment_raw >= end_view and transformation_doing == True and transformation_done == False:
            increment = end_view - t
            transformation_done = True
        elif t + increment_raw > angles.shape[0] and transformation_done == True:
            increment = angles.shape[0] - t
        else:
            increment = increment_raw

        if start_view == 0 and end_view == 0: # no motion:
            increment = increment_raw


        if t < start_view:
            img_new = np.copy(img)
            print('view: ',t, '~',t+increment, ' keep original img')

        elif t >= start_view and t<end_view:
            translation_ = [i / (end_view - start_view) * (t - start_view) for i in translation]
            rotation_ = [i / (end_view - start_view) * (t - start_view) for i in rotation]
        
            I = img[0,...]
            _,_,_,transformation_matrix = transform.generate_transform_matrix(translation_,rotation_,[1,1,1],I.shape)
            transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
            img_new = transform.apply_affine_transform(I, transformation_matrix)
            img_new = img_new[np.newaxis, ...]

            print('view: ',t, '~',t+increment, ' doing transformation')
                
        else:
            I = img[0,...]
            _,_,_,transformation_matrix = transform.generate_transform_matrix(translation,rotation,[1,1,1],I.shape)
            transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
            img_new = transform.apply_affine_transform(I, transformation_matrix)
            img_new = img_new[np.newaxis, ...]
            print('view: ',t, '~',t+increment,' after transformation')

        origin_img = img_new[0,slice_num[0]:slice_num[1],...]
        origin_img = origin_img[:, np.newaxis, ...]

        cuimg = cp.array(origin_img, cp.float32, order = 'C')
        cuangles = cp.array(angles[t:t+increment], cp.float32, order = 'C')

        projector.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless=True)
        cufp = projector.fp(cuimg, angles = cuangles)

        fp = cufp.get()

        projection[:,t:t+increment,...] = fp

        t = t+increment
        if t >= angles.shape[0]:
            break
    
    return projection


def filtered_backporjection(projection,angles,projector,fbp_projector,back_to_original_value = True):
    # z_axis = True when z_axis is the slice, otherwise x-axis is the slice

    cuangles = cp.array(angles, cp.float32, order = 'C')
    fprj = numpy_fan.ramp_filter(fbp_projector, projection, filter_type='RL')
    projector.set_backprojector(ct_fan.distance_driven_bp, angles=cuangles, is_fbp=True)
    cufprj = cp.array(fprj, cp.float32, order = 'C')
    curecon = projector.bp(cufprj)
    recon = curecon.get()
    recon = recon[:,0,...]

    if back_to_original_value == True:
        recon = recon / 0.019 * 1000 - 1024

    return recon

