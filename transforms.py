import math
from typing import Union, Tuple, List
import tensorflow as tf
import tensorflow_addons as tfa

from base import translate, scale, rotate
from misc import get_bbox_from_keypoints, resize_with_pad


def compose_transforms(translate_xy: Union[Tuple, List]=(0, 0), scale_factor: int=1, angle: float=0., is_inv: bool=False):
    """
    Compose transformation matrix
    reference: https://stackoverflow.com/questions/48413604/map-matrix-returned-by-cv2-getrotationmatrix2d

    Args:
        translate_xy (Union[Tuple, List], optional): translation x-value, y-value. Defaults to (0, 0).
        scale_factor (int, optional): scaling factor. Defaults to 1.
        angle (float, optional): rotation degree(required to be converted into radian). Defaults to 0..
        is_inv (bool, optional): if True, it is used for image transformation otherwise for coordinates. Defaults to False.

    Returns:
        tf.Tensor: 1D tensor meaning perspective transformation matrix
    """
    radian = tf.convert_to_tensor(angle * (math.pi / 180), dtype=tf.float32)
    
    if is_inv:
        m1 = translate(-translate_xy[0], -translate_xy[1])
        m2 = scale(scale_factor, scale_factor, is_inv=is_inv)
        m3 = rotate(-radian)
        m4 = translate(translate_xy[0], translate_xy[1])
        
    else:
        m1 = translate(-translate_xy[0], -translate_xy[1])
        m2 = rotate(radian)
        m3 = scale(scale_factor, scale_factor)
        m4 = translate(translate_xy[0], translate_xy[1])
        
    m1 = tf.reshape(m1, (-1,))[:-1]
    m2 = tf.reshape(m2, (-1,))[:-1]
    m3 = tf.reshape(m3, (-1,))[:-1]
    m4 = tf.reshape(m4, (-1,))[:-1]
    
    transforms = tfa.image.compose_transforms([m4, m3, m2, m1]) # m4 @ m3 @ m2 @ m1
    return transforms


def apply_transforms(image, keypoints, scale_factor: int=1, angle: float=0.):
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    center_point = tf.convert_to_tensor([image_width / 2, image_height / 2], dtype=tf.float32) # 회전 기준점
    tx = center_point[0]
    ty = center_point[1]
    scale_factor = tf.convert_to_tensor(scale_factor, dtype=tf.float32)
    
    transforms = compose_transforms(translate_xy=(tx, ty), scale_factor=scale_factor, angle=angle, is_inv=False)
    transforms_inv = compose_transforms(translate_xy=(tx, ty), scale_factor=scale_factor, angle=angle, is_inv=True)
    image = tfa.image.transform(image, transforms_inv, fill_mode='constant')
    
    transforms = tf.reshape(transforms[0, :-2], (2, 3))
    xy = keypoints[:, :2]
    xy = tf.transpose(tf.matmul(transforms[:, :2], xy - center_point, transpose_b=True)) + center_point
    bbox = get_bbox_from_keypoints(xy, image.shape)
    # bbox = tf.cast(bbox, tf.int32)
    
    image, bbox, xy = resize_with_pad(image, bbox, xy)
    
    vis = keypoints[:, 2]
    vis = tf.math.minimum(1., vis)
    vis *= tf.cast((
            (xy[:, 0] >= 0) &
            (xy[:, 0] < image_width) &
            (xy[:, 1] >= 0) &
            (xy[:, 1] < image_height)), tf.float32)
    keypoints = tf.concat([xy, tf.expand_dims(vis, axis=1)], axis=-1)
    return image, bbox, keypoints

