import tensorflow as tf


def resize_with_pad(image, bbox, keypoints, target_shape=[192, 192]):
    """resize with pad on image/bbox/keypoints
    reference: https://github.com/tensorflow/tensorflow/blob/c256c071bb26e1e13b4666d1b3e229e110bc914a/tensorflow/python/ops/image_ops_impl.py#L1726

    Args:
        image (tf.Tensor): original shape image ; (H, W, 3)
        bbox (tf.Tensor): bounding box (x1, y1, x2, y2) ; (4,)
        keypoints (tf.Tensor): keypoints coordinates ; (num_keypoints, 2)
        target_shape (list, optional): input image shape. Defaults to [192, 192].

    Returns:
        Tuple[tf.Tensor]: image, bbox, keypoints with resized
    """
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    target_height, target_width = target_shape[0], target_shape[1]
    
    # convert values to float, to ease divisions
    f_height = tf.cast(height, dtype=tf.float32)
    f_width = tf.cast(width, dtype=tf.float32)
    f_target_height = tf.cast(target_height, dtype=tf.float32)
    f_target_width = tf.cast(target_width, dtype=tf.float32)

    # Find the ratio by which the image must be adjusted
    # to fit within the target
    ratio = tf.math.maximum(f_width / f_target_width, f_height / f_target_height)
    resized_height_float = f_height / ratio
    resized_width_float = f_width / ratio
    resized_height = tf.cast(
        tf.math.floor(resized_height_float), dtype=tf.int32
    )
    resized_width = tf.cast(
        tf.math.floor(resized_width_float), dtype=tf.int32
    )

    padding_height = (f_target_height - resized_height_float) / 2
    padding_width = (f_target_width - resized_width_float) / 2
    f_padding_height = tf.math.floor(padding_height)
    f_padding_width = tf.math.floor(padding_width)
    p_height = tf.math.maximum(0, tf.cast(f_padding_height, dtype=tf.int32))
    p_width = tf.math.maximum(0, tf.cast(f_padding_width, dtype=tf.int32))

    image_resized = tf.image.resize(image, (resized_height, resized_width))
    image_resized_with_pad = tf.pad(image_resized, 
                                    [[p_height, p_height], [p_width, p_width], [0, 0]])
    bbox /= tf.convert_to_tensor([width, height, width, height], dtype=tf.int32)
    bbox = tf.cast(bbox, tf.float32) * tf.convert_to_tensor([resized_width, resized_height, resized_width, resized_height], dtype=tf.float32)
    bbox_with_pad = bbox + tf.convert_to_tensor([p_width, p_height, p_width, p_height], dtype=tf.float32) # assume that bbox format is (x1, y1, x2, y2)
    bbox_with_pad = tf.cast(bbox_with_pad, tf.int32)
    
    keypoints /= tf.convert_to_tensor([[width, height]], dtype=tf.float32)
    keypoints *= tf.convert_to_tensor([[resized_width, resized_height]], dtype=tf.float32)
    keypoints_with_pad = keypoints + tf.reshape(tf.convert_to_tensor([padding_width, padding_height], dtype=tf.float32), (1, 2))
    return image_resized_with_pad, bbox_with_pad, keypoints_with_pad


def get_bbox_from_keypoints(keypoints, image_shape, s=0.2):
    """extract more fitted-bbox from keypoints

    Args:
        keypoints (tf.Tensor): coordinates of keypoints ; (num_keypoints, 2)
        image_shape (Tuple or List): image shape (H, W)
        s (float, optional): enlargement factor. Defaults to 0.2.

    Returns:
        tf.Tensor: bounding box coordinate [x1, y1, x2, y2]
    """
    xmin = tf.math.reduce_min(keypoints[:, 0])
    ymin = tf.math.reduce_min(keypoints[:, 1])
    xmax = tf.math.reduce_max(keypoints[:, 0])
    ymax = tf.math.reduce_max(keypoints[:, 1])
    
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = (xmax - xmin) * (1 + s)
    height = (ymax - ymin) * (1 + s)
    
    bbox = tf.convert_to_tensor([tf.math.maximum(0, center_x - width / 2),
                                 tf.math.maximum(0, center_y - height / 2),
                                 tf.math.minimum(image_shape[1], center_x + width / 2),
                                 tf.math.minimum(image_shape[0], center_y + height / 2)])
    
    return bbox
