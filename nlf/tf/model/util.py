import tensorflow as tf
from simplepyutils import FLAGS


def heatmap_to_image(coords, is_training):
    stride = FLAGS.stride_train if is_training else FLAGS.stride_test

    last_image_pixel = FLAGS.proc_side - 1
    last_receptive_center = last_image_pixel - (last_image_pixel % stride)
    coords_out = coords * last_receptive_center

    if FLAGS.centered_stride:
        coords_out = coords_out + stride // 2

    return coords_out


def heatmap_to_25d(coords, is_training):
    coords2d = heatmap_to_image(coords[..., :2], is_training)
    return tf.concat([coords2d, coords[..., 2:] * FLAGS.box_size_m], axis=-1)


def heatmap_to_metric(coords, is_training):
    coords2d = heatmap_to_image(
        coords[..., :2], is_training) * FLAGS.box_size_m / FLAGS.proc_side
    return tf.concat([coords2d, coords[..., 2:] * FLAGS.box_size_m], axis=-1)
