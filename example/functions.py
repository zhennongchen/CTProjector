import numpy as np

def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width
    low = level - width
    # normalize
    unit = (1-0) / (width*2)
    image[image>high] = high
    image[image<low] = low
    new = (image - low) * unit 
    # for i in range(0,image.shape[0]):
    #     for j in range(0,image.shape[1]):
    #         if image[i,j] > high:
    #             image[i,j] = high
    #         if image[i,j] < low:
    #             image[i,j] = low
    #         norm = (image[i,j] - (low)) * unit
    #         new[i,j] = norm
    return new