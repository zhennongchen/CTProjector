'''
This file provide general utilities in sqs algorithm using generic forward and backprojector.

Unified numpy as cupy through importing.
'''

from scipy.ndimage import shift
import numpy as np
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import HeadCT_motion_correction_PAR.motion_simulator.motion_simulation.ct_basic as basic
import HeadCT_motion_correction_PAR.functions_collection as ff


from typing import Union, Tuple, Callable, Any
from . import BACKEND

if BACKEND == 'cupy':
    import cupy as cp
    import ct_projector.prior.cupy as recon_prior
    from ct_projector.projector.cupy import ct_projector
elif BACKEND == 'numpy':
    import numpy as cp
    import ct_projector.prior.numpy as recon_prior
    from ct_projector.projector.numpy import ct_projector
else:
    raise ValueError('Backend not supported.', BACKEND)


def sqs_fp_w_motion(img, projector,projector_norm, angles, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz,  total_view_num, gantry_rotation_time , increment  , order = 3, use_t_end = True):

    projection = np.zeros([img.shape[1],angles.shape[0],1,projector.nu])
    view_to_time = gantry_rotation_time / total_view_num

    steps = int(total_view_num // increment)


    MVF_list = []

    for step in range(0,steps):

        view_start = increment * step
        view_end = view_start + increment

        t_end = view_end * view_to_time
        
        if use_t_end == True:
            tt = t_end
        else:
            tt = view_start * view_to_time

        translation_ = [spline_tz(np.array([tt])), spline_tx(np.array([tt])), spline_ty(np.array([tt]))]
        rotation_ = [spline_rz(np.array([tt])), spline_rx(np.array([tt])), spline_ry(np.array([tt]))]

        # print('step, tt, translation_, rotation_:',step, tt, translation_, [r/np.pi * 180 for r in rotation_])

        I = img[0,...]
        _,_,_,transformation_matrix = transform.generate_transform_matrix(translation_,rotation_,[1,1,1],I.shape)
        transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
        
        MVF_list.append(transformation_matrix)

        img_new = transform.apply_affine_transform(I, transformation_matrix ,order)
    
        img_new = img_new[np.newaxis, ...]
                
        origin_img = img_new[0,...]
        origin_img = origin_img[:, np.newaxis, ...]

        cuimg = cp.array(origin_img, cp.float32, order = 'C')
        cuangles = cp.array(angles[view_start : view_end], cp.float32, order = 'C')

        cufp = projector.fp(cuimg, angles = cuangles) /projector_norm

        fp = cufp.get()

        projection[:,view_start:view_end,...] = fp

    # print('how many MVF?: ',len(MVF_list))
    # print('projection shape: ',projection.shape)
    return cp.array(projection, cp.float32, order = 'C'), MVF_list



def sqs_bp_w_motion(fp, MVF_list, img, projector,projector_norm, angles, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz, total_view_num , gantry_rotation_time, increment ,  weight = 1, order = 3):
    projection = cp.asnumpy(fp)

    final_img = np.zeros(img.shape)

    view_to_time = gantry_rotation_time / total_view_num

    for step in range(0, len(MVF_list)):

        view_start = increment * step
        view_end = view_start + increment

        projection_part = projection[:,view_start: view_end, :, :]
        angles_partial = angles[view_start : view_end]

        cu_projection_part = cp.array(projection_part, cp.float32, order = 'C')
        cu_angles_partial = cp.array(angles_partial, cp.float32, order = 'C')

        curecon_part = projector.bp(cu_projection_part * weight, angles = cu_angles_partial) / projector_norm
        recon_part = curecon_part.get()[:,0,:,:]
     
       
        # apply motion

        t_end = view_end * view_to_time
        
        translation_ = [-spline_tz(np.array([t_end])), -spline_tx(np.array([t_end])), -spline_ty(np.array([t_end]))]
        rotation_ = [-spline_rz(np.array([t_end])), -spline_rx(np.array([t_end])), -spline_ry(np.array([t_end]))]

        # print('step, t_end, translation_, rotation_:',step, t_end, translation_, [r/np.pi * 180 for r in rotation_])

        _,_,_,transformation_matrix = transform.generate_transform_matrix(translation_,rotation_,[1,1,1],recon_part.shape, which_one_is_first='translation')
        transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, recon_part.shape)

        img_new = transform.apply_affine_transform(recon_part, transformation_matrix ,order)

        final_img += img_new
    
    final_img = final_img / len(MVF_list)
    # print('final_image shape: ',final_img.shape)
    final_img = final_img[:,np.newaxis,:,:]
    # print('final_img_last shape: ', final_img.shape)
    
    return cp.array(final_img, cp.float32, order = 'C')


def sqs_gaussian_one_step_motion(
    projector: ct_projector,
    img: cp.array,
    prj: cp.array,
    norm_img: cp.array,
    projector_norm: float,
    beta: float,
    spline_tx: callable,
    spline_ty: callable,
    spline_tz: callable,
    spline_rx: callable,
    spline_ry: callable, 
    spline_rz: callable, 
    sga: float,
    total_view_num: int, 
    increment: int, 
    gantry_rotation_time: int,
    weight: cp.array = None,
    return_loss: bool = False,
    use_t_end: bool = True,
) -> Union[cp.array, Tuple[cp.array, float, float]]:
    '''
    sqs with gaussian prior. Please see the doc/sqs_equations, section 4.

    Parameters
    -----------------
    projector: ct_projector.
        The projector to perform forward and back projections. Its parameter should be
        compatible with the shape of img and prj.
    img: array(float32) of shape [batch, nz, ny, nx].
        The current image.
    prj: array(float32) of shape [batch, nview, nv, nu].
        The projection data.
    norm_img: array(float32) of shape [batch, nz, ny, nx].
        The norm image A^T*w*A*1.
    projector_norm: float.
        The norm of the system matrix, to make penalty parameter easier to tune.
    beta: float.
        Gaussian prior strength.
    weight: array(float32) of shape [batch, nview, nv, nu].
        The weighting matrix.
    return_loss: bool
        If true, the function will return a tuple [result image, data_loss, penalty_loss].
        If false, only the result image will be returned.

    Returns
    -------------------
    img: array(float32) of shape [batch, nz, ny, nx].
        The updated image.
    data_loss: float.
        Only return if return_loss is True. The data term loss.
    nlm_loss: float.
        Only return if return_loss is True. The penalty term loss.
    '''
    def gaussian_func(img):
        guide = cp.ones(img.shape, cp.float32)
        return recon_prior.nlm(img, guide, 1, [3, 3, 3], [1, 1, 1], 1)

    if weight is None:
        weight = 1

    # turn cupy array [z,1,x,y] to numpy array[z,x,y]
    img_np = cp.asnumpy(img[:,0,:,:]); img_np = img_np[np.newaxis,...]

    # do forward projection:
    angles = ff.get_angles_zc(total_view_num, 360, sga)

    fp, MVF_list = sqs_fp_w_motion(img_np, projector, projector_norm, angles, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz,  total_view_num = total_view_num, gantry_rotation_time = gantry_rotation_time, increment = increment , order = 3, use_t_end = use_t_end)

    # calculate the error:
    fp_delta = fp - prj / projector_norm

    # do backprojection, final bp should have [z,1,x,y]
    bp = sqs_bp_w_motion(fp_delta, MVF_list, img[:,0,:,:], projector,projector_norm, angles, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz,  total_view_num = total_view_num, gantry_rotation_time = gantry_rotation_time, increment = increment, weight = 1,order = 3)

    # sqs
    gauss = 4 * (img - gaussian_func(img))
    img = img - (bp + beta * gauss) / (norm_img + beta * 8)

    if return_loss:
        # fp = projector.fp(img) / projector_norm
        data_loss = 0.5 * cp.sum(weight * (fp - prj / projector_norm)**2)
        # data_loss = cp.mean(cp.abs(fp - prj))

        nlm = gaussian_func(img)
        nlm2 = gaussian_func(img * img)
        nlm_loss = cp.sum(img * img - 2 * img * nlm + nlm2)

        return img, data_loss, nlm_loss
    else:
        return img


def nesterov_acceleration_motion(
    recon_func: Callable[..., Union[cp.array, Tuple[Any, ...]]],
    recon_kwargs: dict,
    img: cp.array,
    img_nesterov: cp.array,
    nesterov: float = 0.5,
) -> Tuple[cp.array, cp.array]:
    '''
    kwargs should contains all params for func except for img.
    func should return img as the only or the first return value.

    Parameters
    -------------------
    func: callable
        The iteration function. It must take a kwarg 'img' as the current image.
        The return value must be a single img array, or a tuple with the first
        element as img.
    img: np.array(float32)
        The current image, whose shape should be compatible with func.
    img_nesterov: np.array(float32)
        The nesterov image (x*), whose shape should be the same with img.
    nesterov: float
        The acceleration parameter.
    **kwargs: dict
        The keyword argument list to be passed to func. Note that 'img' will
        be overridden by img_nesterov.

    Returns
    --------------------
    img: np.array(float32).
        The updated image.
    img_nesterov: np.array(float32).
        The nesterov image (x*) for the next iteration.
    '''

    recon_kwargs['img'] = img
    res = recon_func(**recon_kwargs)

    if type(res) is tuple:
        img = res[0] + nesterov * (res[0] - img_nesterov)
        img_nesterov = res[0]

        return tuple([img, img_nesterov] + list(res[1:]))
    else:
        img = res + nesterov * (res - img_nesterov)
        img_nesterov = res

        return img, img_nesterov