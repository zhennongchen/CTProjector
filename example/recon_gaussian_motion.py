import importlib
import CTProjector.src.ct_projector.recon.cupy 
importlib.reload(CTProjector.src.ct_projector.recon.cupy)

import numpy as np
import cupy as cp
import os
import pandas as pd
import nibabel as nb


import HeadCT_motion_correction_PAR.motion_simulator.motion_simulation.ct_basic as basic
import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform

import CTProjector.src.ct_projector.projector.cupy as ct_projector
import CTProjector.src.ct_projector.recon.cupy as ct_recon
import CTProjector.src.ct_projector.projector.cupy.fan_equiangular as ct_fan

def generate_and_save_sinograms_spline_motion(img, total_angle, amplitude_tx, amplitude_ty, amplitude_tz, amplitude_rx, amplitude_ry, amplitude_rz, file_name , sga = 0, load_file = False, geometry = 'fan', total_view_num = 1000, increment = 100, gantry_rotation_time = 500):
    t = np.linspace(0,gantry_rotation_time, 5, endpoint=True)
    spline_tx = transform.interp_func(t, np.asarray(amplitude_tx))
    spline_ty = transform.interp_func(t, np.asarray(amplitude_ty))
    spline_tz = transform.interp_func(t, np.asarray(amplitude_tz))
    spline_rx = transform.interp_func(t, np.asarray(amplitude_rx))
    spline_ry = transform.interp_func(t, np.asarray(amplitude_ry))
    spline_rz = transform.interp_func(t, np.asarray(amplitude_rz))

    projector = basic.define_forward_projector(img,spacing,total_view_num)
    angles = ff.get_angles_zc(total_view_num, total_angle, sga)

    if load_file == False:
        sinogram = basic.fp_w_spline_motion_model(img, projector, angles,3,  spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz, geometry, total_view = total_view_num, gantry_rotation_time = 500, slice_num = None, increment = increment, order = 3)

        np.save(file_name, sinogram)
    else:
        sinogram = np.load(file_name, allow_pickle = True)

    return angles, sinogram, total_angle, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz, geometry, projector


save_folder = '/mnt/mount_zc_NAS/motion_correction/data/test_sinograms/'

# load image
filename = '/mnt/mount_zc_NAS/motion_correction/data/raw_data/nii-images/thin_slice/MO101701M000006/MO001A000007/img-nii-2.5/img.nii.gz'
img,spacing,affine = basic.basic_image_processing(filename)

new_img = np.zeros([40,234,234])
img = img[20:40,...]
new_img[10:30,...] = img
img_ds = np.copy(new_img)
img_ds = img_ds[np.newaxis, ...]
print(img_ds.shape)


# define motion and make sinogram
total_view_num = 1000
increment = 100
gantry_rotation_time = 500
amplitude_tx = np.linspace(0,0,5)
amplitude_ty = np.linspace(0,0,5)
amplitude_tz = np.linspace(0,5,5)
amplitude_rx = np.linspace(0,0/180*np.pi,5)
amplitude_ry = np.linspace(0,0/180*np.pi,5)
amplitude_rz = np.linspace(0,0/180*np.pi,5)
t = np.linspace(0, gantry_rotation_time, 5, endpoint=True)
sga = 0

file_name = os.path.join(save_folder,'spline1.npy')
geometry = 'fan'
load_file = True
angles, sinogram, total_angle, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz, geometry, projector = generate_and_save_sinograms_spline_motion(img_ds, 360, amplitude_tx, amplitude_ty, amplitude_tz, amplitude_rx, amplitude_ry, amplitude_rz, file_name , sga = sga, load_file = load_file, geometry = geometry, total_view_num = total_view_num, increment = increment)


# load FBP
fbp_motion = nb.load(os.path.join(save_folder, 'motion.nii.gz')).get_fdata(); fbp_motion = np.rollaxis(fbp_motion,2,0)

# load PAR_corrected 
PAR = nb.load(os.path.join(save_folder, 'PAR_corrected.nii.gz' )).get_fdata(); PAR = np.rollaxis(PAR,2,0)


# Iterative recon:
img_ds2 = np.copy(img_ds[:, 20:40,:,:])

# define projector
projector_ir = projector

angles = ff.get_angles_zc(total_view_num, 360, sga)
origin_img = img_ds2[0,...]
origin_img = origin_img[:,np.newaxis,...]

curef = cp.array(origin_img, order='C')
cuangles = cp.array(angles, order='C')

projector_ir.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless = False)
projector_ir.set_backprojector(ct_fan.distance_driven_bp, angles=cuangles, is_fbp = True)


# doing recon
PAR_corrected = (PAR.astype(np.float32) + 1024) / 1000 * 0.019
PAR_corrected[PAR_corrected < 0] = 0
PAR_corrected = PAR_corrected[:,np.newaxis,...]

fbp_motion2 = (fbp_motion.astype(np.float32) + 1024) / 1000 * 0.019
fbp_motion2[fbp_motion2 < 0] = 0
fbp_motion2 = fbp_motion2[:,np.newaxis,...]

# sinogram_part = sinogram[20:40,...]

niter = 900
nos = 12
nesterov = 0.5
beta = 0
zero_init = False

projector_norm = projector_ir.calc_projector_norm()
cunorm_img = projector_ir.calc_norm_img() / projector_norm / projector_norm

cufbp = cp.array(PAR_corrected ,order='C')
cuprj2 = cp.array(sinogram, cp.float32, order = 'C')
cuangles = cp.array(angles, order='C')

Result = []

if zero_init:
    curecon = cp.zeros(cufbp.shape, cp.float32)
    cunesterov = cp.zeros(cufbp.shape, cp.float32)
else:    
    curecon = cp.copy(cufbp)
    cunesterov = cp.copy(curecon)

for i in range(0,niter):
    for n in range(0,nos):

        curecon, cunesterov = ct_recon.nesterov_acceleration_motion(
            ct_recon.sqs_gaussian_one_step_motion,
            img=curecon,
            img_nesterov=cunesterov,
            recon_kwargs={
                'projector': projector_ir,
                'prj': cuprj2,
                'norm_img': cunorm_img,
                'projector_norm': projector_norm,
                'beta': beta,
                't': t,
                'amplitude_tx': amplitude_tx,
                'amplitude_ty': amplitude_ty,
                'amplitude_tz': amplitude_tz,
                'amplitude_rx': amplitude_rx,
                'amplitude_ry': amplitude_ry,
                'amplitude_rz': amplitude_rz,
                'sga': float(sga),
                'total_view_num': total_view_num,
                'increment': increment ,
                'gantry_rotation_time': gantry_rotation_time
            }
        )

    # if (i + 1) % 10 == 0:
    _, data_loss, prior_loss = ct_recon.sqs_gaussian_one_step_motion(
        projector_ir,
        curecon,
        cuprj2,
        cunorm_img,
        projector_norm,
        beta,
        t,
        amplitude_tx,
        amplitude_ty,
        amplitude_tz,
        amplitude_rx,
        amplitude_ry,
        amplitude_rz,
        float(sga),
        total_view_num,
        increment, 
        gantry_rotation_time,
        return_loss=True
    )


    Result.append([i, data_loss, prior_loss])
    print(i, data_loss, prior_loss)

    df = pd.DataFrame(Result, columns = ['step', 'data_loss', 'prior_loss'])

    df.to_excel('/mnt/mount_zc_NAS/motion_correction/data/test_sinograms/loss_record_start_from_PAR.xlsx', index = False)

    recon_ir = curecon.get()[:,0,:,:]
    recon_ir = recon_ir / 0.019 * 1000 - 1024

    nb.save(nb.Nifti1Image(np.rollaxis(recon_ir,0,3),affine), os.path.join('/mnt/mount_zc_NAS/motion_correction/data/test_sinograms/IR-recon','IR_corrected_' + str(i)+'.nii.gz'))