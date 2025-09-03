import numpy as np
import tensorflow as tf
from simplepyutils import FLAGS
from tensorflow_graphics.math.optimizer.levenberg_marquardt import minimize as levenberg_marquardt

from nlf.tf import tfu


def rigid_align(
        coords_pred, coords_true, *, joint_validity_mask=None, scale_align=False,
        reflection_align=False):
    """Returns the predicted coordinates after rigid alignment to the ground truth."""
    if joint_validity_mask is None:
        joint_validity_mask = tf.ones_like(coords_pred[..., 0], dtype=tf.bool)
    return procrustes_tf(
        coords_true, coords_pred, joint_validity_mask, allow_scaling=scale_align,
        allow_reflection=reflection_align)


def linear_combine_points(coords, weights):
    return tf.matmul(weights, coords, transpose_a=True)


def procrustes_tf(X, Y, validity_mask, allow_scaling=False, allow_reflection=False):
    """Register the points in Y by rotation, translation, uniform scaling (optional) and
    reflection (optional)
    to be closest to the corresponding points in X, in a least-squares sense.

    This function operates on batches. For each item in the batch a separate
    transform is computed independently of the others.

    Arguments:
       X: Tensor with shape [batch_size, n_points, point_dimensionality]
       Y: Tensor with shape [batch_size, n_points, point_dimensionality]
       validity_mask: Boolean Tensor with shape [batch_size, n_points] indicating
         whether a point is valid in Y
       allow_scaling: boolean, specifying whether uniform scaling is allowed
       allow_reflection: boolean, specifying whether reflections are allowed

    Returns the transformed version of Y.
    """
    meanY, T, output_scale, meanX = procrustes_tf_transf(
        X, Y, validity_mask, allow_scaling, allow_reflection)
    return ((Y - meanY) @ T) * output_scale + meanX


def procrustes_tf_transf(X, Y, validity_mask, allow_scaling=False, allow_reflection=False):
    validity_mask = validity_mask[..., np.newaxis]
    _0 = tf.constant(0, X.dtype)
    n_points_per_example = tf.math.count_nonzero(
        validity_mask, axis=1, dtype=tf.float32, keepdims=True)
    denominator_correction_factor = tf.cast(
        tf.shape(validity_mask)[1], tf.float32) / n_points_per_example

    def normalize(Z):
        Z = tf.where(validity_mask, Z, _0)
        mean = tf.reduce_mean(Z, axis=1, keepdims=True) * denominator_correction_factor
        centered = tf.where(validity_mask, Z - mean, _0)
        norm = tf.norm(centered, axis=(1, 2), ord='fro', keepdims=True)
        normalized = centered / norm
        return mean, norm, normalized

    meanX, normX, normalizedX = normalize(X)
    meanY, normY, normalizedY = normalize(Y)
    A = tf.matmul(normalizedY, normalizedX, transpose_a=True)
    s, U, V = tf.linalg.svd(A)
    T = tf.matmul(U, V, transpose_b=True)
    s = tf.expand_dims(s, axis=-1)

    if allow_scaling:
        relative_scale = normX / normY
        output_scale = relative_scale * tf.reduce_sum(s, axis=1, keepdims=True)
    else:
        relative_scale = None
        output_scale = 1

    if not allow_reflection:
        # Check if T has a reflection component. If so, then remove it by flipping
        # across the direction of least variance, i.e. the last singular value/vector.
        has_reflection = (tf.linalg.det(T) < 0)[..., np.newaxis, np.newaxis]
        T_mirror = T - 2 * tf.matmul(U[..., -1:], V[..., -1:], transpose_b=True)
        T = tf.where(has_reflection, T_mirror, T)

        if allow_scaling:
            output_scale_mirror = output_scale - 2 * relative_scale * s[:, -1:]
            output_scale = tf.where(has_reflection, output_scale_mirror, output_scale)

    return meanY, T, output_scale, meanX


def reconstruct_absolute(
        coords2d, coords3d_rel, intrinsics, mix_3d_inside_fov=None, weak_perspective=None,
        point_validity_mask=None, border_factor1=0.75, border_factor2=None, mix_based_on_3d=True):
    inv_intrinsics = tf.linalg.inv(tf.cast(intrinsics, coords2d.dtype))
    coords2d_normalized = tf.matmul(
        to_homogeneous(coords2d), inv_intrinsics, transpose_b=True)[..., :2]
    if weak_perspective is None:
        weak_perspective = FLAGS.weak_perspective

    reconstruct_ref_fn = (
        reconstruct_ref_weakpersp if weak_perspective else reconstruct_ref_fullpersp)

    if border_factor2 is None:
        border_factor2 = border_factor1

    is_predicted_to_be_in_fov1 = is_within_fov(coords2d, border_factor1)

    if point_validity_mask is not None:
        is_predicted_to_be_in_fov1 = tf.logical_and(is_predicted_to_be_in_fov1, point_validity_mask)

    ref = reconstruct_ref_fn(coords2d_normalized, coords3d_rel, is_predicted_to_be_in_fov1)
    coords_abs_3d_based = coords3d_rel + tf.expand_dims(ref, 1)
    reference_depth = ref[:, 2]
    relative_depths = coords3d_rel[..., 2]

    coords_abs_2d_based = back_project(coords2d_normalized, relative_depths, reference_depth)

    if mix_3d_inside_fov is not None:
        coords_abs_2d_based = (
                mix_3d_inside_fov * coords_abs_3d_based +
                (1 - mix_3d_inside_fov) * coords_abs_2d_based)

    if mix_based_on_3d:
        coords_proj_for_mix_decision = project_pose(coords_abs_3d_based, intrinsics)
    else:
        coords_proj_for_mix_decision = project_pose(coords_abs_2d_based, intrinsics)

    is_predicted_to_be_in_fov2 = tf.logical_and(
        is_within_fov(coords_proj_for_mix_decision, border_factor2),
        coords_abs_3d_based[..., 2] > 0.001)

    return tf.where(
        is_predicted_to_be_in_fov2[..., tf.newaxis], coords_abs_2d_based, coords_abs_3d_based)



def reconstruct_ref_weakpersp(normalized_2d, coords3d_rel, validity_mask):
    mean3d, stdev3d = tfu.mean_stdev_masked(
        coords3d_rel[..., :2], validity_mask, items_axis=1, dimensions_axis=2)

    mean2d, stdev2d = tfu.mean_stdev_masked(
        normalized_2d[..., :2], validity_mask, items_axis=1, dimensions_axis=2)

    stdev2d = tf.maximum(stdev2d, 1e-5)
    stdev3d = tf.maximum(stdev3d, 1e-5)

    old_mean = tfu.reduce_mean_masked(coords3d_rel, validity_mask, axis=1, keepdims=True)
    new_mean_z = tf.math.divide_no_nan(stdev3d, stdev2d)
    new_mean = to_homogeneous(mean2d) * new_mean_z
    return tf.squeeze(new_mean - old_mean, 1)


def to_homogeneous(x):
    return tf.concat([x, tf.ones_like(x[..., :1])], axis=-1)


def map_fn_via_tensorarray(fn, elems, fn_output_signature):
    # alternative to map_fn via TensorArray, which allows gradients for ragged tensors
    # unlike map_fn
    batch_size = tf.shape(tf.nest.flatten(elems)[0])[0]
    arr = tf.TensorArray(
        fn_output_signature.dtype, size=batch_size, element_shape=fn_output_signature.shape)
    for i in tf.range(batch_size):
        tf.autograph.experimental.set_loop_options(parallel_iterations=1024)
        arr = arr.write(i, fn(tf.nest.map_structure(lambda x: x[i], elems)))
    return arr.stack()


@tf.function
def reconstruct_ref_fullpersp(normalized_2d, coords3d_rel, validity_mask):
    """Reconstructs the reference point location.

    Args:
      normalized_2d: normalized image coordinates of the joints
         (without intrinsics applied), shape [batch_size, n_points, 2]
      coords3d_rel: 3D camera coordinate offsets relative to the unknown reference
         point which we want to reconstruct, shape [batch_size, n_points, 3]
      validity_mask: boolean mask of shape [batch_size, n_points] containing True
         where the point is reliable and should be used in the reconstruction

    Returns:
      The 3D reference point in camera coordinates, shape [batch_size, 3]
    """

    if isinstance(normalized_2d, tf.RaggedTensor):
        return map_fn_via_tensorarray(
            lambda x: reconstruct_ref_fullpersp(
                x[0][tf.newaxis], x[1][tf.newaxis], x[2][tf.newaxis])[0],
            (normalized_2d, coords3d_rel, validity_mask),
            fn_output_signature=tf.TensorSpec([3], tf.float32))

    n_batch = tf.shape(normalized_2d)[0]
    n_points = tf.shape(normalized_2d)[1]

    def rms_normalize_and_reshape(x):
        scale = tf.sqrt(tfu.reduce_mean_masked(
            tf.square(x), validity_mask, axis=(1, 2), keepdims=True) + np.float32(1e-10))
        normalized = x / scale
        reshaped = tf.reshape(normalized, [-1, n_points * 2, 1])
        return scale, reshaped

    eyes = tf.tile(tf.expand_dims(tf.eye(2), 0), [n_batch, n_points, 1])
    scale2d, reshaped2d = rms_normalize_and_reshape(normalized_2d)
    A = tf.concat([eyes, -reshaped2d], axis=2)  # A is [batch_size, n_points * 2, 3]

    rel_backproj = normalized_2d * coords3d_rel[:, :, 2:] - coords3d_rel[:, :, :2]
    scale_rel_backproj, b = rms_normalize_and_reshape(rel_backproj)
    # b is [batch_size, n_points * 2, 1]

    weights = tf.repeat(
        tf.cast(validity_mask, tf.float32), 2, axis=1)[..., tf.newaxis] + np.float32(1e-4)
    ref = tf.linalg.lstsq(
        A * weights, b * weights, l2_regularizer=FLAGS.fullpersp_l2_regul, fast=True)
    # ref is [batch_size, 3, 1]

    ref = tf.concat([ref[:, :2], ref[:, 2:] / scale2d], axis=1) * scale_rel_backproj
    return tf.squeeze(ref, axis=-1)


def project(points):
    return points[..., :2] / points[..., 2:3]


def back_project(camcoords2d, delta_z, z_offset):
    return to_homogeneous(camcoords2d) * tf.expand_dims(delta_z + tf.expand_dims(z_offset, -1), -1)


def is_within_fov(imcoords, border_factor=0.75):
    stride_train = FLAGS.stride_train
    offset = -stride_train / 2 if not FLAGS.centered_stride else 0
    lower = tf.cast(stride_train * border_factor + offset, tf.float32)
    upper = tf.cast(FLAGS.proc_side - stride_train * border_factor + offset, tf.float32)
    proj_in_fov = tf.reduce_all(tf.logical_and(imcoords >= lower, imcoords <= upper), axis=-1)
    return proj_in_fov


def project_pose(coords3d, intrinsic_matrix):
    projected = coords3d / tf.maximum(np.float32(0.001), coords3d[..., 2:])
    return tf.matmul(projected, intrinsic_matrix[..., :2, :], transpose_b=True)


def intrinsic_matrix_from_field_of_view(fov_degrees, imshape):
    imshape = tf.cast(imshape, tf.float32)
    fov_radians = fov_degrees * tf.constant(np.pi / 180, tf.float32)
    larger_side = tf.reduce_max(imshape)
    focal_length = larger_side / (tf.math.tan(fov_radians / 2) * 2)
    return tf.convert_to_tensor(
        [[[focal_length, 0, imshape[1] / 2],
          [0, focal_length, imshape[0] / 2],
          [0, 0, 1]]], tf.float32)


def get_new_rotation_matrix(forward_vector, up_vector):
    # Z will point forwards, towards the box center
    new_z = tf.linalg.l2_normalize(forward_vector, axis=-1)
    # Get the X (right direction) as the cross of forward and up.
    new_x = tf.linalg.cross(new_z, up_vector)
    # Get alternative X by rotating the new Z around the old Y by 90 degrees
    # in case lookdir happens to align with the up vector and the above cross product is zero.
    new_x_alt = tf.stack([new_z[:, 2], tf.zeros_like(new_z[:, 2]), -new_z[:, 0]], axis=1)
    new_x = tf.linalg.l2_normalize(
        tf.where(tf.linalg.norm(new_x, axis=-1, keepdims=True) == 0, new_x_alt, new_x), axis=-1)
    # Complete the right-handed coordinate system to get Y
    new_y = tf.linalg.cross(new_z, new_x)
    # Stack the axis vectors to get the rotation matrix
    return tf.stack([new_x, new_y, new_z], axis=1)


def rotation_mat(angle, rot_axis):
    sin = tf.math.sin(angle)
    cos = tf.math.cos(angle)
    _0 = tf.zeros_like(angle)
    _1 = tf.ones_like(angle)

    if rot_axis == 'x':
        entries = [
            _1, _0, _0,
            _0, cos, sin,
            _0, -sin, cos]
    elif rot_axis == 'y':
        entries = [
            cos, _0, -sin,
            _0, _1, _0,
            sin, _0, cos]
    else:
        entries = [
            cos, -sin, _0,
            sin, cos, _0,
            _0, _0, _1]

    return tf.reshape(tf.stack(entries, axis=-1), tf.concat([tf.shape(angle), [3, 3]], axis=0))


def decompose_rotation_to_twist_and_swing(rotation, twist_axis):
    """Decompose a rotation into a twist and a swing component."""
    # The rotation is to be understood as an active rotation that rotates points.
    # The output is R_twist and R_swing, such that
    #   R = R_swing R_twist
    #   The axis of R_twist is twist_axis
    #   The axis of R_swing is orthogonal to twist_axis
    rotated_twist_axis = tf.linalg.matvec(rotation, twist_axis)
    swing_axis = tf.linalg.cross(twist_axis, rotated_twist_axis)
    swing_rotation = project_to_SO3(
        outer_product(rotated_twist_axis, twist_axis) +
        outer_product(swing_axis, swing_axis))
    twist_rotation = tf.linalg.matmul(swing_rotation, rotation, transpose_a=True)
    return twist_rotation, swing_rotation


def project_to_SO3(A):
    _, U, V = tf.linalg.svd(A)
    T = tf.matmul(U, V, transpose_b=True)
    has_reflection = (tf.linalg.det(T) < 0)[..., tf.newaxis, tf.newaxis]
    T_mirror = T - 2 * tf.matmul(U[..., -1:], V[..., -1:], transpose_b=True)
    return tf.where(has_reflection, T_mirror, T)


def outer_product(x, y):
    return tf.einsum('...i,...j->...ij', x, y)


def haversine_rotmat(rot1, rot2=None):
    # the haversine is equal to (1-cos(x))/2 = sin(x/2)**2
    # It turns out that the haversine of the angle between two rotations
    # (ie the angle of the axis-angle representation) is just sum of squared differences
    # in 3x3 rotation matrix representation divided by 8.
    if rot2 is None:
        rot2 = tf.eye(3, batch_shape=tf.shape(rot1)[:-2], dtype=rot1.dtype)

    squared_diff = tf.math.squared_difference(rot1, rot2)
    return tf.reduce_sum(squared_diff, axis=(-2, -1)) * 0.125


def center_relative_pose(
        coords3d, joint_validity_mask=None, center_is_mean=False, center_joints=None):
    if center_is_mean:
        if isinstance(coords3d, np.ndarray):
            if joint_validity_mask is None:
                center = np.mean(coords3d, axis=1, keepdims=True)
            else:
                coords3d = coords3d.copy()
                coords3d[~joint_validity_mask] = np.nan
                center = np.nanmean(coords3d, axis=1, keepdims=True)
        else:
            if joint_validity_mask is None:
                center = tf.reduce_mean(coords3d, axis=1, keepdims=True)
            else:
                if center_joints is not None:
                    center = tfu.reduce_mean_masked(
                        tf.gather(coords3d, center_joints, axis=1),
                        tf.gather(joint_validity_mask, center_joints, axis=1),
                        axis=1, keepdims=True)
                else:
                    center = tfu.reduce_mean_masked(
                        coords3d, joint_validity_mask, axis=1, keepdims=True)
    else:
        center = coords3d[:, -1:]
    return coords3d - center
