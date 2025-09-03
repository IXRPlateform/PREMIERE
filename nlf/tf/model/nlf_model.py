import numpy as np
import simplepyutils as spu
import tensorflow as tf
from simplepyutils import FLAGS

from nlf.paths import PROJDIR
from nlf.tf import tfu, tfu3d
from nlf.tf.model import util as model_util


class NLFModel(tf.keras.Model):
    def __init__(self, backbone, weight_field, normalizer):
        super().__init__(dtype=tf.float32)
        self.backbone = backbone
        self.normalizer = normalizer
        self.heatmap_head = LocalizerHead(weight_field, normalizer)
        self.input_resolution = tf.Variable(np.int32(FLAGS.proc_side), trainable=False)

        joint_info = spu.load_pickle(f'{PROJDIR}/joint_info_866.pkl')
        i_left_joints = [i for i, n in enumerate(joint_info.names) if n[0] == 'l']
        i_right_joints = [joint_info.ids['r' + joint_info.names[i][1:]] for i in i_left_joints]
        i_center_joints = [i for i in range(joint_info.n_joints) if
                           i not in i_left_joints and
                           i not in i_right_joints]
        permutation = tf.constant(i_left_joints + i_right_joints + i_center_joints, tf.int32)
        self.inv_permutation = tf.math.invert_permutation(permutation)

        self.canonical_lefts = self.add_weight(
            shape=(len(i_left_joints), 3), dtype=tf.float32,
            trainable=FLAGS.trainable_canonical_joints,
            initializer=tf.keras.initializers.Zeros(),
            constraint=tf.keras.constraints.MinMaxNorm(0, 0.07, axis=1),
            name='canonical_locs_left')
        self.canonical_centers = self.add_weight(
            shape=(len(i_center_joints), 2), dtype=tf.float32,
            trainable=FLAGS.trainable_canonical_joints,
            initializer=tf.keras.initializers.Zeros(),
            constraint=tf.keras.constraints.MinMaxNorm(0, 0.07, axis=1),
            name='canonical_locs_centers')

        self.canonical_locs_init = np.load(f'{PROJDIR}/canonical_loc_symmetric_init_866.npy')
        self.canonical_delta_mask = np.array(
            [not is_hand_joint(n) for n in joint_info.names]).astype(np.float32)

    def canonical_locs(self):
        canonical_rights = tf.concat([
            -1 * self.canonical_lefts[:, :1],
            self.canonical_lefts[:, 1:]], axis=1)
        canonical_centers = tf.concat([
            tf.zeros_like(self.canonical_centers[:, :1]),
            self.canonical_centers], axis=1)
        permuted = tf.concat([self.canonical_lefts, canonical_rights, canonical_centers], axis=0)
        return (tf.gather(permuted, self.inv_permutation, axis=0) *
                self.canonical_delta_mask[:, np.newaxis] +
                self.canonical_locs_init)

    @tf.function(input_signature=[])
    def get_canonical_locs(self):
        return self.canonical_locs()

    def call(self, inp, training=None):
        image, intrinsics, canonical_points = inp
        image = tf.cast(image, tfu.get_dtype())
        tf.debugging.assert_type(image, tfu.get_dtype())
        canonical_points_tensor = canonical_points.to_tensor()
        tf.debugging.assert_type(canonical_points_tensor, tf.float32)
        point_validity = tfu.to_mask_tensor(canonical_points[..., 0])

        features = self.backbone(image, training=training)
        coords2d, coords3d, uncertainties = self.heatmap_head(
            features, canonical_points_tensor, training=training)

        coords3d_abs = tfu3d.reconstruct_absolute(
            coords2d, coords3d, intrinsics,
            mix_3d_inside_fov=FLAGS.mix_3d_inside_fov, point_validity_mask=point_validity,
            border_factor1=0.55 if training else 0.6)
        return tf.RaggedTensor.from_tensor(
            coords3d_abs, lengths=canonical_points.row_lengths(),
            row_splits_dtype=canonical_points.row_splits.dtype, ragged_rank=1)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float16),
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),
        tf.RaggedTensorSpec(
            shape=(None, None, 3), dtype=tf.float32, ragged_rank=1, row_splits_dtype=tf.int32)
    ])
    def predict_multi(self, image, intrinsic_matrix, canonical_points):
        # This function is needed to avoid having to go through Keras' __call__
        # in the exported SavedModel, which causes all kinds of problems.
        return self.call((image, intrinsic_matrix, canonical_points), training=False) * 1000

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float16),
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
    ])
    def predict_multi_same_canonicals(
            self, image, intrinsic_matrix, canonical_points, flip_canonicals_per_image=()):
        image = tf.cast(image, tfu.get_dtype())
        tf.debugging.assert_type(image, tfu.get_dtype())
        tf.debugging.assert_type(canonical_points, tf.float32)
        features = self.backbone(image, training=False)

        coords2d, coords3d, uncertainties = self.heatmap_head.predict_same_canonicals(
            features, canonical_points, training=False)
        reconstr_kwargs = dict(
            mix_3d_inside_fov=FLAGS.mix_3d_inside_fov,
            border_factor1=1,
            border_factor2=0.6,
            mix_based_on_3d=True)

        coords3d_abs = tfu3d.reconstruct_absolute(
            coords2d, coords3d, intrinsic_matrix,
            point_validity_mask=(uncertainties < 0.3 if FLAGS.nll_loss else None),
            **reconstr_kwargs) * 1000

        if tf.math.reduce_any(flip_canonicals_per_image):
            fl_coords2d, fl_coords3d, fl_uncertainties = self.heatmap_head.predict_same_canonicals(
                features, canonical_points * tf.constant([-1, 1, 1], tf.float32), training=False)
            fl_coords3d_abs = tfu3d.reconstruct_absolute(
                fl_coords2d, fl_coords3d, intrinsic_matrix,
                point_validity_mask=(fl_uncertainties < 0.3 if FLAGS.nll_loss else None),
                **reconstr_kwargs) * 1000
            coords3d_abs = tf.where(
                flip_canonicals_per_image[:, tf.newaxis, tf.newaxis], fl_coords3d_abs, coords3d_abs)
            uncertainties = tf.where(
                flip_canonicals_per_image[:, tf.newaxis], fl_uncertainties,
                uncertainties)

        factor = tf.convert_to_tensor(1 if FLAGS.fix_uncert_factor else 3, dtype=tf.float32)
        return coords3d_abs, uncertainties * factor

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
    ])
    def get_weights_for_canonical_points(self, canonical_points):
        weights = self.heatmap_head.weight_field(canonical_points)
        w_tensor, b_tensor = transpose_weights(
            weights, FLAGS.backbone_link_dim, 2 + FLAGS.depth, tfu.get_dtype())
        weights_fl = self.heatmap_head.weight_field(
            canonical_points * tf.constant([-1, 1, 1], tf.float32))
        w_tensor_fl, b_tensor_fl = transpose_weights(
            weights_fl, FLAGS.backbone_link_dim, 2 + FLAGS.depth, tfu.get_dtype())
        return dict(
            w_tensor=w_tensor, b_tensor=b_tensor,
            w_tensor_flipped=w_tensor_fl, b_tensor_flipped=b_tensor_fl)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float16),
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),
        dict(
            w_tensor=tf.TensorSpec(
                shape=(FLAGS.backbone_link_dim, None, 2 + FLAGS.depth), dtype=tf.float16),
            b_tensor=tf.TensorSpec(shape=(None, 2 + FLAGS.depth), dtype=tf.float16),
            w_tensor_flipped=tf.TensorSpec(
                shape=(FLAGS.backbone_link_dim, None, 2 + FLAGS.depth), dtype=tf.float16),
            b_tensor_flipped=tf.TensorSpec(shape=(None, 2 + FLAGS.depth), dtype=tf.float16),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
    ])
    def predict_multi_same_weights(
            self, image, intrinsic_matrix, weights, flip_canonicals_per_image=()):
        features_processed = self.get_features(image)
        return self.decode_features_multi_same_weights(
            features_processed, intrinsic_matrix, weights, flip_canonicals_per_image)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float16)])
    def get_features(self, image):
        image = tf.cast(image, tfu.get_dtype())
        features = self.backbone(image, training=False)
        features_processed = self.heatmap_head.layer(features)
        return features_processed

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float16),
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),
        dict(
            w_tensor=tf.TensorSpec(
                shape=(FLAGS.backbone_link_dim, None, 2 + FLAGS.depth), dtype=tf.float16),
            b_tensor=tf.TensorSpec(shape=(None, 2 + FLAGS.depth), dtype=tf.float16),
            w_tensor_flipped=tf.TensorSpec(
                shape=(FLAGS.backbone_link_dim, None, 2 + FLAGS.depth), dtype=tf.float16),
            b_tensor_flipped=tf.TensorSpec(shape=(None, 2 + FLAGS.depth), dtype=tf.float16),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.bool),
    ])
    def decode_features_multi_same_weights(
            self, features, intrinsic_matrix, weights, flip_canonicals_per_image=()):
        features_processed = features
        flip_canonicals_per_image_ind = tf.cast(flip_canonicals_per_image, tf.int32)
        nfl_features_processed, fl_features_processed = tf.dynamic_partition(
            features_processed, flip_canonicals_per_image_ind, 2)
        partitioned_indices = tf.dynamic_partition(
            tf.range(tf.shape(features_processed)[0]), flip_canonicals_per_image_ind, 2)
        nfl_coords2d, nfl_coords3d, nfl_uncertainties = apply_weights3d_same_canonicals_impl(
            nfl_features_processed, weights['w_tensor'], weights['b_tensor'],
            n_out_channels=2 + FLAGS.depth)
        fl_coords2d, fl_coords3d, fl_uncertainties = apply_weights3d_same_canonicals_impl(
            fl_features_processed, weights['w_tensor_flipped'], weights['b_tensor_flipped'],
            n_out_channels=2 + FLAGS.depth)
        coords2d = tf.dynamic_stitch(partitioned_indices, [nfl_coords2d, fl_coords2d])
        coords3d = tf.dynamic_stitch(partitioned_indices, [nfl_coords3d, fl_coords3d])
        uncertainties = tf.dynamic_stitch(
            partitioned_indices, [nfl_uncertainties, fl_uncertainties])

        coords2d = model_util.heatmap_to_image(coords2d, is_training=False)
        coords3d = model_util.heatmap_to_metric(coords3d, is_training=False)
        coords3d_abs = tfu3d.reconstruct_absolute(
            coords2d, coords3d, intrinsic_matrix,
            point_validity_mask=(uncertainties < 0.3 if FLAGS.nll_loss else None),
            mix_3d_inside_fov=FLAGS.mix_3d_inside_fov, border_factor1=1, border_factor2=0.6,
            mix_based_on_3d=True) * 1000
        return coords3d_abs, uncertainties


class LocalizerHead(tf.keras.layers.Layer):
    def __init__(self, weight_field, normalizer):
        super().__init__()
        self.weight_field = weight_field

        from nlf.tf.backbones.efficientnet.effnetv2_model import conv_kernel_initializer

        conv = tf.keras.layers.Conv2D(
            filters=FLAGS.backbone_link_dim, kernel_size=1,
            kernel_constraint=tf.keras.constraints.MinMaxNorm(
                0, FLAGS.constrain_kernel_norm, axis=[0, 1, 2],
                rate=FLAGS.constraint_rate),
            use_bias=False, kernel_initializer=conv_kernel_initializer)

        self.layer = tf.keras.Sequential([
            conv,
            normalizer(axis=-1, momentum=0.9, epsilon=1e-3),
            tf.keras.layers.Activation(tf.nn.silu)])

    def call(self, features, canonical_positions, training=None):
        weights = self.weight_field(canonical_positions)  # NP[C(c+1)]
        tf.debugging.assert_type(weights, tfu.get_dtype())
        return self.call_with_weights(features, weights, training=training)

    def call_with_weights(self, features, weights, training=None):
        features_processed = self.layer(features)  # NHWc
        coords2d, coords3d_rel_pred, uncertainties = apply_weights3d(
            features_processed, weights, n_out_channels=2 + FLAGS.depth)
        coords2d_pred = model_util.heatmap_to_image(coords2d, training)
        coords3d_rel_pred = model_util.heatmap_to_metric(coords3d_rel_pred, training)
        return coords2d_pred, coords3d_rel_pred, uncertainties

    def predict_same_canonicals(
            self, features, canonical_positions, training=None):
        weights = self.weight_field(canonical_positions)  # NP[C(c+1)]
        tf.debugging.assert_type(weights, tfu.get_dtype())
        features_processed = self.layer(features)  # NHWc

        coords2d, coords3d_rel_pred, uncertainties = apply_weights3d_same_canonicals(
            features_processed, weights, n_out_channels=2 + FLAGS.depth)
        coords2d_pred = model_util.heatmap_to_image(coords2d, training)
        coords3d_rel_pred = model_util.heatmap_to_metric(coords3d_rel_pred, training)
        return coords2d_pred, coords3d_rel_pred, uncertainties


def apply_weights3d(features, weights, n_out_channels):
    # features: NHWc 128,8,8,1280
    # weights:  NP[(c+1)C] 128,768,10*1281
    weights = tf.cast(weights, features.dtype)
    weights_resh = tfu.unmerge_last_axis(
        weights, features.shape[-1] + 1, n_out_channels)  # NPC(c+1)
    w_tensor = weights_resh[..., :-1, :]  # NPCc
    b_tensor = weights_resh[..., -1, :]  # NPC

    logits = (tf.einsum(
        'nhwc,npcC->nphwC', features, w_tensor) +
              b_tensor[:, :, tf.newaxis, tf.newaxis, :])

    uncertainty_map = tf.cast(logits[..., 0], tf.float32)
    coords_metric_xy = tfu.soft_argmax(tf.cast(logits[..., 1], tf.float32), axis=[3, 2])
    heatmap25d = tfu.softmax(tf.cast(logits[..., 2:], tf.float32), axis=[3, 2, 4])
    heatmap2d = tf.reduce_sum(heatmap25d, axis=4)
    uncertainties = tf.reduce_sum(uncertainty_map * tf.stop_gradient(heatmap2d), axis=[3, 2])
    uncertainties = tf.nn.softplus(uncertainties + FLAGS.uncert_bias) + FLAGS.uncert_bias2
    coords25d = tfu.decode_heatmap(heatmap25d, axis=[3, 2, 4])
    coords2d = coords25d[..., :2]
    coords3d = tf.concat([coords_metric_xy, coords25d[..., 2:]], axis=-1)
    return coords2d, coords3d, uncertainties


def transpose_weights(weights, n_in_channels, n_out_channels, feature_dtype):
    weights = tf.cast(weights, feature_dtype)
    weights_resh = tfu.unmerge_last_axis(
        weights, n_in_channels + 1, n_out_channels)  # P(c+1)C
    w_tensor = weights_resh[..., :-1, :]  # PcC
    b_tensor = weights_resh[..., -1, :]  # PC
    w_tensor = tf.transpose(w_tensor, perm=[1, 0, 2])  # PcC-> cPC
    return w_tensor, b_tensor


def apply_weights3d_same_canonicals(features, weights, n_out_channels):
    # features: NHWc 128,8,8,1280
    # weights:  P[C(c+1)] 768,10*1281
    w_tensor, b_tensor = transpose_weights(
        weights, features.shape[-1], n_out_channels, features.dtype)
    return apply_weights3d_same_canonicals_impl(features, w_tensor, b_tensor, n_out_channels)


def apply_weights3d_same_canonicals_impl(features, w_tensor, b_tensor, n_out_channels):
    w_tensor = tfu.merge_last_two_axes(w_tensor)[tf.newaxis, tf.newaxis]
    b_tensor = tf.reshape(b_tensor, [-1])
    logits = tf.nn.conv2d(features, w_tensor, strides=1, padding='SAME') + b_tensor
    logits = tfu.unmerge_last_axis(logits, -1, n_out_channels)  # nhwpC
    uncertainty_map = tf.cast(logits[..., 0], tf.float32)

    coords_metric_xy = tfu.soft_argmax(tf.cast(logits[..., 1], tf.float32), axis=[2, 1])
    heatmap25d = tfu.softmax(tf.cast(logits[..., 2:], tf.float32), axis=[2, 1, 4])
    heatmap2d = tf.reduce_sum(heatmap25d, axis=4)
    uncertainties = tf.reduce_sum(uncertainty_map * tf.stop_gradient(heatmap2d), axis=[2, 1])
    uncertainties = tf.nn.softplus(uncertainties + FLAGS.uncert_bias) + FLAGS.uncert_bias2
    coords25d = tfu.decode_heatmap(heatmap25d, axis=[2, 1, 4])
    coords2d = coords25d[..., :2]
    coords3d = tf.concat([coords_metric_xy, coords25d[..., 2:]], axis=-1)
    return coords2d, coords3d, uncertainties


def is_hand_joint(name):
    n = name.split('_')[0]
    if any(x in n for x in ['thumb', 'index', 'middle', 'ring', 'pinky']):
        return True

    return (
            (n.startswith('lhan') or n.startswith('rhan')) and len(n) > 4)