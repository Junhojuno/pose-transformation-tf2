import tensorflow as tf


def translate(tx, ty):
    """make translation matrix"""
    translate_matrix = [[1, 0, tx],
                        [0, 1, ty],
                        [0, 0, 1]]
    return tf.convert_to_tensor(translate_matrix, dtype=tf.float32)


def scale(sx, sy, is_inv=False):
    """make scale matrix"""
    if is_inv:
        scaling_matrix = [[1 / sx, 0, 0],
                          [0, 1 / sy, 0],
                          [0, 0, 1]]
    else:
        scaling_matrix = [[sx, 0, 0],
                          [0, sy, 0],
                          [0, 0, 1]]
    return tf.convert_to_tensor(scaling_matrix, dtype=tf.float32)


def rotate(radian):
    """
    make rotation matrix
    because of non-standard coordinates, radian is required to be negative.
    """
    rotate_matrix = [[tf.math.cos(-radian), -tf.math.sin(-radian), 0],
                     [tf.math.sin(-radian), tf.math.cos(-radian), 0],
                     [0, 0, 1]]
    return rotate_matrix
