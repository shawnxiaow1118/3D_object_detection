import numpy as np 

def make_anchors(bases, stride, image_shape, feature_shape, allowed_border=0):
	H, W                  = feature_shape
    img_height, img_width = image_shape

    # anchors = shifted bases. Generate proposals from box deltas and anchors
    shift_x = np.arange(0, W) * stride
    shift_y = np.arange(0, H) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    B  = len(bases)
    HW = len(shifts)
    anchors   = (bases.reshape((1, B, 4)) + shifts.reshape((1, HW, 4)).transpose((1, 0, 2)))
    anchors   = anchors.reshape((HW * B, 4)).astype(np.int32)
    num_anchors = int(HW * B)

    # only keep anchors inside the image
    inside_inds = np.where(
        (anchors[:, 0] >= -allowed_border) &
        (anchors[:, 1] >= -allowed_border) &
        (anchors[:, 2] < img_width  + allowed_border) &  # width
        (anchors[:, 3] < img_height + allowed_border)    # height
    )[0].astype(np.int32)

    return anchors, inside_inds