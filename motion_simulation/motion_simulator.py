#!/usr/bin/env python

# %%
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import SimpleITK as sitk
import time
import os
import ct_basic as ct
import functions_collection as ff
import transformation as transform
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
no_random  = False # do you need to randomly generate motion parameters?
total_view = 2400
gantry_rotation_time = 500 #unit ms
time_range = [250,400] # unit ms
view_increment = 10 # increment in gantry views
main_save_folder = '/local/data/CT_motion/CT_images/simulated_data'
folder_name = 'random_'

# %%
# define the patient list
patient_list = ff.find_all_target_files(['example_CT_volume/*'],'/local/data/CT_motion/CT_images')
print(patient_list)

# %%
for p in patient_list:
    patient_id = ff.find_timeframe(p,2)
    print('patient: ',patient_id)
    save_folder = os.path.join(main_save_folder,'patient_' + str(patient_id))
    ff.make_folder([save_folder])
    
    img,spacing = ct.basic_image_processing(p)
    print('sitk image: shape:  ',img.shape, ' spacing: ',spacing)
    img_nb = nb.load(p); img_affine = img_nb.affine
    print('nib image shape: ',img_nb.shape)

    # define projectors
    img = img[np.newaxis, ...]
    projector = ct.define_forward_projector(img,spacing,total_view)
    fbp_projector = ct.backprojector(img,spacing)

    # very important: make sure that the arrays are saved in C order
    cp.cuda.Device(0).use()
    ct_projector.set_device(0)

    angles = projector.get_angles()
    print(angles.shape)

    for random_i in range(0,10):
        
        # create folder
        random_folder = os.path.join(save_folder,folder_name +str(random_i + 1))
        ff.make_folder([random_folder])

        print('\n',random_i + 1 , 'random')
        
        # generate random motion
        # 10% pure translation, 10% pure rotation, 80% hybrid:
        if random_i % 10 == 8:
            tt = max_t; rr = 0
        elif random_i % 10 == 9:
            tt = 0; rr = max_r
        else:
            tt = max_t; rr = max_r

        translation,translation_mm,rotation,start_view,end_view,lasting_view,lasting_time = transform.generate_random_3D_motion(
            tt,rr,spacing,
            time_range = time_range,
            total_view_num = total_view, 
            gantry_rotation_time = gantry_rotation_time,
            no_random = no_random)
        print(translation_mm,[i/np.pi * 180 for i in rotation],start_view,end_view,lasting_time)
        # save motion parameters
        parameter_file = os.path.join(random_folder,'motion_parameters.txt')
        ff.txt_writer(parameter_file,[translation_mm,[i/np.pi * 180 for i in rotation], [start_view],[end_view],[lasting_time],[total_view],[gantry_rotation_time]],['translation_mm','rotation degree','start_view','end_view','lasting_time(ms)','total_projection_view','gantry_rotation_time(ms)'])


        # generate forward projection
        slice_num = [int(img.shape[1]/2) - 30, int(img.shape[1]/2) + 30]
        projection = ct.fp_w_motion(img, projector, angles, translation, rotation, start_view, end_view,slice_num, view_increment)
        # save fp
        print(projection.shape)
        a = projection[int(projection.shape[0]/2), :, 0, :]
        print(a.shape)
        ff.save_grayscale_image(a, os.path.join(random_folder,'projection.png'))
        np.save(os.path.join(random_folder,'projection.npy'),projection)


        # generate backprojection
        recon = ct.filtered_backporjection(projection,angles,projector,fbp_projector,back_to_original_value=True)
        print(recon.shape)
        # save recon
        recon_nb = nb.Nifti1Image(np.rollaxis(np.rollaxis(recon,0,3),0,2),img_affine)
        nb.save(recon_nb, os.path.join(random_folder,'recon.nii.gz'))
        ff.save_grayscale_image(recon[int(recon.shape[0]/2),...], os.path.join(random_folder,'recon.png'))

    

