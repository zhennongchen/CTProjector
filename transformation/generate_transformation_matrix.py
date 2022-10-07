from fcntl import DN_MODIFY
import numpy as np
from numpy import random
import math
from .rotation_matrix_from_angle import rotation_matrix


def generate_transform_matrix(t,r,s,img_shape):

    assert type(img_shape) == tuple
   
    ## translation
    # t should be the translation in [x,y,z] directions
    assert len(t) == len(img_shape)

    translation = np.eye(len(img_shape) + 1)
    translation[:len(img_shape),len(img_shape)] = np.transpose(np.asarray(t))

    ## rotation
    # r should be the rotation angle in [x,y,z] if 3D, or a single scalar value if 2D
    if len(img_shape) == 2:
        assert type(r) == float
    if len(img_shape) == 3:
        assert len(r) == 3
    
    rotation = np.eye(len(img_shape) + 1)

    if len(img_shape) == 2:

        rotation[:2, :2] = rotation_matrix(r)

    elif len(img_shape) == 3:
        x = rotation_matrix(r[0],matrix_type="roll")
        y = rotation_matrix(r[1],matrix_type="pitch")
        z = rotation_matrix(r[2],matrix_type="yaw")
        rotation[:3, :3] = np.dot(z, np.dot(y, x))
    else:
        raise Exception("image_dimension must be either 2 or 3.")

    # scale
    scale = np.eye(len(img_shape) + 1)
    for ax in range(0,len(img_shape)):
        scale[ax, ax] = s[ax]

    return translation,rotation,scale,np.dot(scale, np.dot(rotation, translation))

def generate_random_3D_motion(max_t,max_r,pixel_dim = [0.68,0.68,1], time_range = [250,400],total_view_num = 2400, gantry_rotation_time = 500, no_random = False):
    # default, max_t (maximum translation) no more than 5mm
    # default, max_r (maximum rotation) no more than 15 degree
    # time_interval is the range of possible lasting time, a list with [minimum_lasting_time, maximum_lasting_time], default [200,400]ms
    # total_view_new = 2400 for conventional CT
    # gantry_rotation_time, usually 500 ms or 250 ms per gantry rotation

    per_view_time = gantry_rotation_time / total_view_num

    # get the randomized translation
    if no_random == False:
        while True:
            [t_x,t_y,t_z] = [random.rand()*max_t ,random.rand()*max_t,random.rand()*max_t]
            if math.sqrt((t_x**2 + t_y**2 + t_z**2)) <= max_t:
                break
        translation = [t_x/pixel_dim[0], t_y/pixel_dim[1], t_z/pixel_dim[-1]]
        translation_mm = [t_x,t_y,t_z]

        # get the randomized rotation
        rotation = [random.rand() * max_r ,random.rand() * max_r,random.rand() * max_r]
        rotation = [i / 180 * np.pi for i in rotation]
    else:
        translation_mm = [max_t, max_t, max_t]
        translation = [max_t/pixel_dim[0], max_t/pixel_dim[1], max_t/pixel_dim[-1]]
        rotation = [max_r,max_r,max_r]
        rotation = [i / 180 * np.pi for i in rotation]
    
    # get start view and end view
    # no motion situation:
    if sum(translation_mm) + sum(rotation) == 0:
        start_view = 0
        end_view = 0
        lasting_view = 0
        lasting_time = 0
    else:
        # get a randomized lasting time
        lasting_time = random.rand() * (time_range[1] - time_range[0]) + time_range[0]

        lasting_view = int(lasting_time / per_view_time)

        start_view = int(random.rand() * (total_view_num - lasting_view - 1))
        end_view = int(start_view + lasting_view)
    
    return translation,translation_mm,rotation,start_view,end_view,lasting_view,lasting_time

    

# def generate_random_transform(params, shape):

#     ##
#     ## Translation
#     ##

#     translation = np.eye(params.image_dimension + 1)
#     #raw code   
#     for t, ax in zip(params.translation_range, params.img_spatial_indices):
#         random_t=np.random.uniform(-t, t)
#         translation[ax, params.image_dimension] =random_t * shape[ax]

#     #only translate to the first quadrant
#     #for t,ax in zip(params.translation_range,params.img_spatial_indices): 
#      #   random_t=np.random.uniform(-t,0) 
#       #  translation[ax,params.image_dimension]=random_t *shape[ax]


#     ##
#     ## Rotation
#     ##

#     rotation = np.eye(params.image_dimension + 1)

#     if params.image_dimension == 2:

#         theta = (
#             np.pi
#             / 180.0
#             * np.random.uniform(-params.rotation_range, params.rotation_range)
#         )

#         rotation[:2, :2] = dv.rotation_matrix_from_angle(theta)
#     elif params.image_dimension == 3:
#         x = dv.rotation_matrix_from_angle(
#             np.pi
#             / 180.0
#             * np.random.uniform(-params.rotation_range[0], params.rotation_range[0]),
#             matrix_type="roll",
#         )
#         y = dv.rotation_matrix_from_angle(
#             np.pi
#             / 180.0
#             * np.random.uniform(-params.rotation_range[1], params.rotation_range[1]),
#             matrix_type="pitch",
#         )
#         z = dv.rotation_matrix_from_angle(
#             np.pi
#             / 180.0
#             * np.random.uniform(-params.rotation_range[2], params.rotation_range[2]),
#             matrix_type="yaw",
#         )
#         rotation[:3, :3] = np.dot(z, np.dot(y, x))
#     else:
#         raise Exception("image_dimension must be either 2 or 3.")

#     ##
#     ## Scale
#     ##
    
#     scale = np.eye(params.image_dimension + 1)
#     scale_factor = np.random.uniform(1 - params.scale_range, 1 + params.scale_range)
#     for ax in params.img_spatial_indices:
#         scale[ax, ax] = scale_factor
    
#     ##
#     ## Flip
#     ##

#     flip = np.eye(params.image_dimension + 1)

#     for f, ax in zip(params.flip, params.img_spatial_indices):
#         if f and (np.random.random() < 0.5):
#             flip[ax, ax] *= -1

#     ##
#     ## Compose
#     ##

#     return translation,rotation,scale,np.dot(scale, np.dot(rotation, translation))