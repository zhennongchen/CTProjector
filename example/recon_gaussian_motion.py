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
#
def generate_and_save_sinograms_spline_motion(img, total_angle, amplitude_tx, amplitude_ty, amplitude_tz, amplitude_rx, amplitude_ry, amplitude_rz, file_name , sga = 0, load_file = False, geometry = 'fan', total_view_num = 1000, increment = 100, gantry_rotation_time = 500):
    t = np.linspace(gantry_rotation_time / 10,gantry_rotation_time, 10, endpoint=True)
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

# load static image
# filename = '/mnt/mount_zc_NAS/motion_correction/data/raw_data/nii-images/thin_slice/MO101701M000006/MO001A000007/img-nii-2.5/img.nii.gz'
filename = os.path.join(save_folder, 'rx4rz4_sga0_increment100_fan_high_z_resolution/static_HR_slice60-100fromoriginal.nii.gz')
static = nb.load(filename).get_fdata()
img,spacing,affine = basic.basic_image_processing(filename)
img = img[np.newaxis,...]
print(img.shape)

# define motion and make sinogram
total_view_num = 1000
increment = 100
gantry_rotation_time = 500
amplitude_tx = np.linspace(0,0,10)
amplitude_ty = np.linspace(0,0,10)
amplitude_tz = np.linspace(0,0,10)
amplitude_rx = np.linspace(0.4/180 * np.pi ,4/180*np.pi,10)
amplitude_ry = np.linspace(0,0/180*np.pi,10)
amplitude_rz = np.linspace(0.4/180 * np.pi ,4/180*np.pi,10)
sga = 0

t = np.linspace(gantry_rotation_time / 10,gantry_rotation_time, 10, endpoint=True)
spline_tx = transform.interp_func(t, amplitude_tx)
spline_ty = transform.interp_func(t, amplitude_ty)
spline_tz = transform.interp_func(t, amplitude_tz)
spline_rx = transform.interp_func(t, amplitude_rx)
spline_ry = transform.interp_func(t, amplitude_ry)
spline_rz = transform.interp_func(t, amplitude_rz)

file_name = os.path.join(save_folder,'rx4rz4_sga0_increment100_fan_high_z_resolution','spline1.npy')
geometry = 'fan'
load_file = True
angles, sinogram, total_angle, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz, geometry, projector = generate_and_save_sinograms_spline_motion(img, 360, amplitude_tx, amplitude_ty, amplitude_tz, amplitude_rx, amplitude_ry, amplitude_rz, file_name , sga = sga, load_file = load_file, geometry = geometry, total_view_num = total_view_num, increment = increment)
print(sinogram.shape)

# load PAR_corrected 
# PAR = nb.load(os.path.join(save_folder,'rx4rz4_sga0_increment100_fan_high_z_resolution','PAR_corrected_HR.nii.gz' )).get_fdata()
PAR = nb.load(os.path.join(save_folder,'IR_recon','IR_corrected_25.nii.gz' )).get_fdata()
print(PAR.shape)

mae_par,_,rmse_par,_, ssim_par = ff.compare(PAR[:,:,10:50], static[:,:,10:50],cutoff_low = -10, extreme = 1000)
print('PAR results: ', mae_par, rmse_par, ssim_par)

# Iterative recon:
# define projector
projector_ir = projector

angles = ff.get_angles_zc(total_view_num, 360, sga)
origin_img = img[0,...]
origin_img = origin_img[:,np.newaxis,...]

curef = cp.array(origin_img, order='C')
cuangles = cp.array(angles, order='C')

projector_ir.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless = False)
projector_ir.set_backprojector(ct_fan.distance_driven_bp, angles=cuangles)  # no FBP!!


# doing recon
PAR_corrected = np.rollaxis(PAR,2,0)
PAR_corrected = (PAR_corrected.astype(np.float32) + 1024) / 1000 * 0.019
PAR_corrected[PAR_corrected < 0] = 0
PAR_corrected = PAR_corrected[:,np.newaxis,...]


niter = 400
nos = 5
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

        curecon, cunesterov, data_loss, _ = ct_recon.nesterov_acceleration_motion(
            ct_recon.sqs_gaussian_one_step_motion,
            img=curecon,
            img_nesterov=cunesterov,
            recon_kwargs={
                'projector': projector_ir,
                'prj': cuprj2,
                'norm_img': cunorm_img,
                'projector_norm': projector_norm,
                'beta': beta,
                'spline_tx': spline_tx,
                'spline_ty': spline_ty,
                'spline_tz': spline_tz,
                'spline_rx': spline_rx,
                'spline_ry': spline_ry,
                'spline_rz': spline_rz,
                'sga': float(sga),
                'total_view_num': total_view_num,
                'increment': increment ,
                'gantry_rotation_time': gantry_rotation_time,
                'return_loss':  True,
                'use_t_end': True,
            }
        )

    recon_ir = curecon.get()[:,0,:,:]
    recon_ir = recon_ir / 0.019 * 1000 - 1024

    recon_ir = np.rollaxis(recon_ir,0,3)
    mae_ir,_,rmse_ir,_, ssim_ir = ff.compare(recon_ir[:,:,10:50], static[:,:,10:50],cutoff_low = -10, extreme = 1000)

    nb.save(nb.Nifti1Image(recon_ir,affine), os.path.join('/mnt/mount_zc_NAS/motion_correction/data/test_sinograms/IR_recon','IR_corrected_' + str(i)+'.nii.gz'))

    print('IR results: ', data_loss, mae_ir, rmse_ir, ssim_ir)

    Result.append([i, data_loss, mae_ir, rmse_ir, ssim_ir])
    df = pd.DataFrame(Result, columns = ['step', 'data_loss', 'mae_ir', 'rmse_ir', 'ssim_ir'])
    df.to_excel('/mnt/mount_zc_NAS/motion_correction/data/test_sinograms/loss_record_start_from_PAR.xlsx', index = False)