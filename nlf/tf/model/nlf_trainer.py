import addict
import fleras.exceptions
import numpy as np
import smplfitter.tf
import tensorflow as tf
from fleras import EasyDict
from simplepyutils import FLAGS

from nlf.tf import tfu, tfu3d


class NLFTrainer(fleras.ModelTrainer):
    def __init__(self, localizer_model, **kwargs):
        super().__init__(**kwargs)
        self.model = localizer_model
        self.body_model_kinds = [
            ('smpl', 'female'), ('smpl', 'male'), ('smpl', 'neutral'),
            ('smplh', 'female'), ('smplh', 'male'), ('smplx', 'female'), ('smplx', 'male'),
            ('smplx', 'neutral'), ('smplxmoyo', 'female')]
        self.body_models = {
            (n, g): smplfitter.tf.SMPLBodyModel(model_name=n, gender=g)
            for n, g in self.body_model_kinds}

    def prepare_inputs(self, inps, training):
        # FORWARD BODY MODELS ON GPU
        inps = addict.Dict(inps)

        # First we need to group the different body model types (smpl, smplh, smplx) and
        # genders together, so we can forward them in a batched way.
        bm_kinds = self.body_model_kinds

        # These selectors contain the indices where a certain body model appears in the batch
        # e.g. body_model_selectors['smpl', 'neutral'] == [0, 2, 5, 7, ...]
        body_model_selectors = {
            (n, g): tf.where(inps.param.body_model == f'{n}_{g[0]}')[:, 0] for (n, g) in bm_kinds}
        # The permutation is the one needed to sort the batch so that all body models of the same
        # kind are grouped together.
        permutation = tf.concat([body_model_selectors[k] for k in bm_kinds], axis=0)
        # The inverse permutation is the one needed to undo the permutation, we will use it later
        invperm = tf.math.invert_permutation(permutation)
        # The sizes tensor contains how many of each body model type we have in the batch
        sizes = tf.stack([tf.shape(body_model_selectors[k])[0] for k in bm_kinds])

        # This function is similar to tf.dynamic_partition
        def permute_and_split(x):
            parts = tf.split(tf.gather(x, permutation, axis=0), sizes)
            return dict(zip(bm_kinds, parts))

        # Each of these become dictionaries that map to e.g. the pose of each body model type
        # Since different body models have different number of joints, the tensor sizes
        # may be different in each value of the dictionary.
        pose = permute_and_split(inps.param.pose)
        shape = permute_and_split(inps.param.shape)
        trans = permute_and_split(inps.param.trans)
        kid_factor = permute_and_split(inps.param.kid_factor)
        scale = permute_and_split(inps.param.scale)
        interp_weights_indices = permute_and_split(inps.param.interp_weights.indices)
        interp_weights_data = permute_and_split(inps.param.interp_weights.data)
        interp_weights_dense_shape = permute_and_split(inps.param.interp_weights.dense_shape)

        # Now we determine the GT points for each body model type
        def decode_points_for_body_model(k):
            bm = self.body_models[k]
            # We first forward the corresponding body model to get the vertices and joints
            result = bm(pose[k][:, :bm.num_joints * 3], shape[k], trans[k], kid_factor[k])
            # We concatenate the vertices and joints
            # (and scale them, which is not part of the body model definition, but some datasets
            # specify a scale factor for their fits, so we have to use it. AGORA's kid factor
            # is probably a better way.)
            verts_and_joints = (
                    tf.concat([result['vertices'], result['joints']], axis=1) *
                    scale[k][:, tf.newaxis, tf.newaxis])

            # The random internal and surface points of the current batch are specified by
            # nlf.tf.loading.parametric as vertex and joint weightings. The weights are
            # the Sibson coordinates of the points in canonical space w.r.t. the mesh vertices
            # and joints. About a million points were precomputed and the data loader
            # samples from that. Anyways, we now apply the same weights (Sibson coordinates)
            # to the posed and shaped vertices and joints to get the GT points that NLF
            # should learn to predict based on the image and the canonical points.
            # This function does the interpolation using CSR sparse representation.
            return interpolate_sparse(
                verts_and_joints, interp_weights_indices[k], interp_weights_data[k],
                interp_weights_dense_shape[k])

        # We now put the GT points of the different body model kinds back together
        # in the original order before we sorted them according to body model type.
        # For this we use the inverse permutation.
        gt_points = tf.concat([decode_points_for_body_model(k) for k in bm_kinds], axis=0)
        inps.param.coords3d_true = tf.gather(gt_points, invperm, axis=0)

        # LOOK UP CANONICAL POINTS
        # This gets the canonical points of the joints of all the different skeleton definitions.
        # We enforce symmetry on these (i.e., only the right side is simply the mirror of the left,
        # there are no variables for the right side; and the x of the middle joints like the spine
        # are forced to be zero, there is no variable for that).
        # model.canonical_locs() assembles the canonical points from the underlying trainable
        # variables and constructs this symmetric version.
        canonical_locs = self.model.canonical_locs()

        # Since each example can define a custom set of points (from the global pool of possible
        # skeleton points for which we have trainable canonicals), we now need to pick out the ones
        # for which each example provides GT annotation. The idea is that each skeleton-based
        # example specifies *indices* into the global set of canonical points.
        inps.kp3d.canonical_points = tf.concat([
            tf.gather(canonical_locs, inps.kp3d.point_ids, axis=0),
            inps.kp3d.canonical_points], axis=1)
        inps.dense.canonical_points = tf.concat([
            tf.gather(canonical_locs, inps.dense.point_ids, axis=0),
            inps.dense.canonical_points], axis=1)
        inps.kp2d.canonical_points = tf.concat([
            tf.gather(canonical_locs, inps.kp2d.point_ids, axis=0),
            inps.kp2d.canonical_points], axis=1)

        # STACKING AND SPLITTING
        # We have four different sub-batches corresponding to the different annotation categories.
        # For efficient handling, e.g. passing though the backbone network, we have to concat
        # them all.
        inps.image = tf.concat([
            inps.param.image,
            inps.kp3d.image,
            inps.dense.image,
            inps.kp2d.image], axis=0)
        inps.intrinsics = tf.concat([
            inps.param.intrinsics,
            inps.kp3d.intrinsics,
            inps.dense.intrinsics,
            inps.kp2d.intrinsics], axis=0)
        inps.canonical_points = tf.concat([
            inps.param.canonical_points,
            inps.kp3d.canonical_points,
            inps.dense.canonical_points,
            inps.kp2d.canonical_points], axis=0)

        # Some of the tensors are actually ragged tensors, which need to be converted to
        # dense tensors for efficient processing.
        # Besides the dense (padded) version, also produce dense (non-ragged) boolean mask tensors
        # that specify which elements were part of the ragged tensor and which are just paddings
        # This will be important to know when averaging in the loss computation.
        def to_mask_tensor(x):
            return tfu.to_mask_tensor(x[..., 0], length=FLAGS.num_points)

        inps.kp3d.point_validity = to_mask_tensor(inps.kp3d.coords3d_true)
        inps.dense.point_validity = to_mask_tensor(inps.dense.coords2d_true)
        inps.kp2d.point_validity = to_mask_tensor(inps.kp2d.coords2d_true)
        inps.point_validity_mask = tf.concat([
            inps.param.point_validity,
            inps.kp3d.point_validity,
            inps.dense.point_validity,
            inps.kp2d.point_validity], axis=0)

        def to_tensor(x):
            return x.to_tensor(shape=[None, FLAGS.num_points, None])

        inps.canonical_points_tensor = to_tensor(inps.canonical_points)
        inps.kp3d.coords3d_true = to_tensor(inps.kp3d.coords3d_true)
        inps.dense.coords2d_true = to_tensor(inps.dense.coords2d_true)
        inps.kp2d.coords2d_true = to_tensor(inps.kp2d.coords2d_true)

        inps.coords2d_true = tf.concat([
            tfu3d.project_pose(inps.param.coords3d_true, inps.param.intrinsics),
            tfu3d.project_pose(inps.kp3d.coords3d_true, inps.kp3d.intrinsics),
            inps.dense.coords2d_true,
            inps.kp2d.coords2d_true], axis=0)

        # Check if Z coord is larger than 1 mm. this will be used to filter out points behind the
        # the camera, which are not visible in the image, and should not be part of the loss.
        is_in_front_true = tf.concat([
            inps.param.coords3d_true[..., 2] > 0.001,
            inps.kp3d.coords3d_true[..., 2] > 0.001,
            tf.ones_like(inps.dense.coords2d_true[..., 0], dtype=tf.bool),
            tf.ones_like(inps.kp2d.coords2d_true[..., 0], dtype=tf.bool)], axis=0)

        # Check if the 2D GT points are within the field of view of the camera.
        # The border factor parameter is measured in terms of stride units of the backbone.
        # E.g., 0.5 means that we only consider points "within the fov" if it's at least half
        # a stride inside the image border. This is because the network is not able to output
        # 2D coordinates at the border of the image.
        inps.is_within_fov_true = tf.logical_and(
            tf.logical_and(
                tfu3d.is_within_fov(inps.coords2d_true, 0.5),
                inps.point_validity_mask),
            is_in_front_true)
        return inps

    def forward_train(self, inps, training):
        preds = EasyDict(param=EasyDict(), kp3d=EasyDict(), dense=EasyDict(), kp2d=EasyDict())

        def backbone_and_head(image, canonical_points):
            # We perform horizontal flipping augmentation here.
            # We randomly decide to flip each example. This means that
            # 1) the image is flipped horizontally (tf.image.flip_left_right) and
            # 2) the canonical points' x coord is flipped (canonical_points * [-1, 1, 1]).
            # At the end the predictions will also be flipped back accordingly.
            flip_mask = tf.random.uniform([tf.shape(image)[0], 1, 1]) > 0.5
            image = tf.where(
                flip_mask[..., tf.newaxis],
                tf.image.flip_left_right(image),
                image)
            canonical_points = tf.where(
                flip_mask,
                canonical_points * tf.constant([-1, 1, 1], tf.float32),
                canonical_points)

            # Flipped or not, the image is passed through the backbone network.
            features = self.model.backbone(image, training=training)
            tf.debugging.assert_all_finite(features, 'Nonfinite features!')
            tf.debugging.assert_type(features, tfu.get_dtype())
            tf.debugging.assert_type(canonical_points, tf.float32)

            # The heatmap head (which contains the localizer field that dynamically constructs
            # the weights) now outputs the 2D and 3D predictions, as well as the uncertainties.
            head2d, head3d, uncertainties = self.model.heatmap_head(
                features, canonical_points, training=training)

            # The results need to be flipped back if they were flipped for the input at the start
            # of this function.
            head3d = tf.where(
                flip_mask,
                head3d * tf.constant([-1, 1, 1], tf.float32),
                head3d)
            head2d = tf.where(
                flip_mask,
                tf.concat([FLAGS.proc_side - 1 - head2d[..., :1], head2d[..., 1:]], axis=-1),
                head2d)
            return head2d, head3d, uncertainties

        head2d, head3d, uncertainties = backbone_and_head(inps.image, inps.canonical_points_tensor)

        # Now we perform the reconstruction of the absolute 3D coordinates.
        # We have 2D coords in pixel space and 3D coords at metric scale but up to unknown
        # translation. These are redundant to some extent and we have two choices to create the
        # output for the points inside the field of view: either back-projectingt the 2D points
        # or translating the 3D points. The mix factor here is controlling the weighting of
        # these two options.
        mix_3d_inside_fov = (
            tf.random.uniform([tf.shape(head3d)[0], 1, 1])
            if training else FLAGS.mix_3d_inside_fov)

        # The reconstruction is only performed using those points that are within the field of
        # view according to the GT.
        validity = inps.is_within_fov_true
        if FLAGS.nll_loss:
            # Furthermore, if we have reliable uncertainties (not too early in training)
            # then we also pick only those points where the uncertainty is below a certain
            # threshold.
            is_early = (tf.cast(self.adjusted_train_counter(), tf.float32) <
                        tf.cast(0.1 * FLAGS.training_steps, tf.float32))
            validity = tf.logical_and(validity, tf.logical_or(is_early, uncertainties < 0.3))
        preds.coords3d_abs = self.reconstruct_absolute(
            head2d, head3d, inps.intrinsics, mix_3d_inside_fov, validity)

        # SPLITTING
        # The prediction was more efficient to do in one big batch, but now
        # for loss and metric computation it's better to split the different annotation
        # categories (i.e., parametric, 3D skeleton, densepose and 2D skeleton) into separate
        # tensors again.
        batch_parts = [inps.param, inps.kp3d, inps.dense, inps.kp2d]
        batch_sizes = [tf.shape(p.image)[0] for p in batch_parts]
        (preds.param.coords3d_abs, preds.kp3d.coords3d_abs, preds.dense.coords3d_abs,
         preds.kp2d.coords3d_abs) = tf.split(preds.coords3d_abs, batch_sizes, axis=0)

        preds.param.uncert, preds.kp3d.uncert, preds.dense.uncert, preds.kp2d.uncert = tf.split(
            uncertainties, batch_sizes, axis=0)

        return preds

    def compute_losses(self, inps, preds):
        losses = EasyDict()

        # Parametric
        losses_parambatch, losses.loss_parambatch = self.compute_loss_with_3d_gt(
            inps.param, preds.param)
        # We also store the individual component losses for each category to track them in
        # Weights and Biases (wandb).
        losses.loss_abs_param = losses_parambatch.loss3d_abs
        losses.loss_rel_param = losses_parambatch.loss3d
        losses.loss_px_param = losses_parambatch.loss2d

        # 3D keypoints
        losses_kpbatch, losses.loss_kpbatch = self.compute_loss_with_3d_gt(inps.kp3d, preds.kp3d)
        losses.loss_abs_kp = losses_kpbatch.loss3d_abs
        losses.loss_rel_kp = losses_kpbatch.loss3d
        losses.loss_px_kp = losses_kpbatch.loss2d

        # Densepose
        losses.loss_densebatch = self.compute_loss_with_2d_gt(inps.dense, preds.dense)

        # 2D keypoints
        losses.loss_2dbatch = self.compute_loss_with_2d_gt(inps.kp2d, preds.kp2d)

        # AGGREGATE
        # The final loss is a weighted sum of the losses computed on the four different
        # training example categories.
        losses.loss = tf.add_n([
            FLAGS.loss_factor_param * losses.loss_parambatch,
            FLAGS.loss_factor_kp * losses.loss_kpbatch,
            FLAGS.loss_factor_dense * losses.loss_densebatch,
            FLAGS.loss_factor_2d * losses.loss_2dbatch,
        ])

        for name, value in losses.items():
            tf.debugging.assert_all_finite(value, f'Nonfinite {name}!')

        return losses

    def compute_loss_with_3d_gt(self, inps, preds):
        losses = EasyDict()

        if inps.point_validity is None:
            inps.point_validity = tf.ones_like(preds.coords3d_abs[..., 0], dtype=tf.bool)

        diff = inps.coords3d_true - preds.coords3d_abs

        # CENTER-RELATIVE 3D LOSS
        # We now compute a "center-relative" error, which is either root-relative
        # (if there is a root joint present), or mean-relative (i.e. the mean is subtracted).
        meanrel_diff = tfu3d.center_relative_pose(
            diff, joint_validity_mask=inps.point_validity, center_is_mean=True)
        # root_index is a [batch_size] int tensor that holds which one is the root
        # diff is [batch_size, joint_cound, 3]
        # we now need to select the root joint from each batch element
        if inps.root_index.shape.ndims == 0:
            inps.root_index = tf.fill(tf.shape(diff)[:1], inps.root_index)

        # diff has shape N,P,3 for batch, point, coord
        # and root_index has shape N
        # and we want to select the root joint from each batch element
        sanitized_root_index = tf.where(
            inps.root_index == -1, tf.zeros_like(inps.root_index), inps.root_index)
        root_diff = tf.expand_dims(tf.gather_nd(diff, tf.stack(
            [tf.range(tf.shape(diff)[0]), sanitized_root_index], axis=1)), axis=1)

        rootrel_diff = diff - root_diff
        # Some elements of the batch do not have a root joint, which is marked as -1 as root_index.
        center_relative_diff = tf.where(
            inps.root_index[:, tf.newaxis, tf.newaxis] == -1, meanrel_diff, rootrel_diff)

        losses.loss3d = tfu.reduce_mean_masked(
            self.my_norm(center_relative_diff, preds.uncert), inps.point_validity)

        # ABSOLUTE 3D LOSS (camera-space)
        absdiff = tf.abs(diff)

        # Since the depth error will naturally scale linearly with distance, we scale the z-error
        # down to the level that we would get if the person was 5 m away.
        scale_factor_for_far = tf.minimum(
            np.float32(1), 5 / tf.abs(inps.coords3d_true[..., 2:]))
        absdiff_scaled = tf.concat(
            [absdiff[..., :2], absdiff[..., 2:] * scale_factor_for_far], axis=-1)

        # There are numerical difficulties for points too close to the camera, so we only
        # apply the absolute loss for points at least 30 cm away from the camera.
        is_far_enough = inps.coords3d_true[..., 2] > 0.3
        is_valid_and_far_enough = tf.logical_and(inps.point_validity, is_far_enough)

        # To make things simpler, we estimate one uncertainty and automatically
        # apply a factor of 4 to get the uncertainty for the absolute prediction
        # this is just an approximation, but it works well enough.
        # The uncertainty does not need to be perfect, it merely serves as a
        # self-gating mechanism, and the actual value of it is less important
        # compared to the relative values between different points.
        losses.loss3d_abs = tfu.reduce_mean_masked(
            self.my_norm(absdiff_scaled, preds.uncert * 4.),
            is_valid_and_far_enough)

        # 2D PROJECTION LOSS (pixel-space)
        # We also compute a loss in pixel space to encourage good image-alignment in the model.
        coords2d_pred = tfu3d.project_pose(preds.coords3d_abs, inps.intrinsics)
        coords2d_true = tfu3d.project_pose(inps.coords3d_true, inps.intrinsics)

        # Balance factor which considers the 2D image size equivalent to the 3D box size of the
        # volumetric heatmap. This is just a factor to get a rough ballpark.
        # It could be tuned further.
        scale_2d = 1 / FLAGS.proc_side * FLAGS.box_size_m

        # We only use the 2D loss for points that are in front of the camera and aren't
        # very far out of the field of view. It's not a problem that the point is outside
        # to a certain extent, because this will provide training signal to move points which
        # are outside the image, toward the image border. Therefore those point predictions
        # will gather up near the border and we can mask them out when doing the absolute
        # reconstruction.
        is_in_fov_pred = tf.logical_and(
            tfu3d.is_within_fov(coords2d_pred, border_factor=-20 * (FLAGS.proc_side / 256)),
            preds.coords3d_abs[..., 2] > 0.001)
        is_near_fov_true = tf.logical_and(
            tfu3d.is_within_fov(coords2d_true, border_factor=-20 * (FLAGS.proc_side / 256)),
            inps.coords3d_true[..., 2] > 0.001)
        losses.loss2d = tfu.reduce_mean_masked(
            self.my_norm((coords2d_true - coords2d_pred) * scale_2d, preds.uncert),
            tf.logical_and(
                is_valid_and_far_enough,
                tf.logical_and(is_in_fov_pred, is_near_fov_true)))

        return losses, tf.add_n([
            losses.loss3d,
            losses.loss2d,
            FLAGS.absloss_factor * self.stop_grad_before_step(
                losses.loss3d_abs, FLAGS.absloss_start_step)])

    def compute_loss_with_2d_gt(self, inps, preds):
        scale_2d = 1 / FLAGS.proc_side * FLAGS.box_size_m
        coords2d_pred = tfu3d.project_pose(preds.coords3d_abs, inps.intrinsics)

        is_in_fov_pred2d = tfu3d.is_within_fov(
            coords2d_pred,
            border_factor=-20 * (FLAGS.proc_side / 256))
        is_near_fov_true2d = tfu3d.is_within_fov(
            inps.coords2d_true,
            border_factor=-20 * (FLAGS.proc_side / 256))

        return tfu.reduce_mean_masked(
            self.my_norm(
                (inps.coords2d_true - coords2d_pred) * scale_2d, preds.uncert),
            tf.logical_and(
                tf.logical_and(is_in_fov_pred2d, is_near_fov_true2d),
                inps.point_validity))

    def compute_metrics(self, inps, preds, training):
        metrics = EasyDict()
        metrics_parambatch = self.compute_metrics_with_3d_gt(inps.param, preds.param)
        if not training:
            return metrics_parambatch

        metrics_kp = self.compute_metrics_with_3d_gt(inps.kp3d, preds.kp3d, '_kp')
        metrics_dense = self.compute_metrics_with_2d_gt(
            inps.dense, preds.dense, '_dense')
        metrics_2d = self.compute_metrics_with_2d_gt(inps.kp2d, preds.kp2d, '_2d')

        metrics.update(**metrics_parambatch, **metrics_kp, **metrics_dense, **metrics_2d)
        return metrics

    def compute_metrics_with_3d_gt(self, inps, preds, suffix=''):
        metrics = EasyDict()
        diff = inps.coords3d_true - preds.coords3d_abs

        # ABSOLUTE
        metrics['mean_error_abs' + suffix] = tfu.reduce_mean_masked(
            tf.norm(diff, axis=-1), inps.point_validity) * 1000

        # RELATIVE
        meanrel_absdiff = tf.abs(tfu3d.center_relative_pose(
            diff, joint_validity_mask=inps.point_validity, center_is_mean=True))
        dist = tf.norm(meanrel_absdiff, axis=-1)
        metrics['mean_error' + suffix] = tfu.reduce_mean_masked(dist, inps.point_validity) * 1000

        # PCK/AUC
        threshold = np.float32(0.1)
        auc_score = tfu.auc(dist, 0, threshold)
        metrics['auc' + suffix] = tfu.reduce_mean_masked(auc_score, inps.point_validity) * 100
        is_correct = tf.cast(dist <= threshold, tf.float32)
        metrics['pck' + suffix] = tfu.reduce_mean_masked(is_correct, inps.point_validity) * 100

        # PROCRUSTES
        coords3d_pred_procrustes = tfu3d.rigid_align(
            preds.coords3d_abs, inps.coords3d_true, joint_validity_mask=inps.point_validity,
            scale_align=True)
        dist_procrustes = tf.norm(coords3d_pred_procrustes - inps.coords3d_true, axis=-1)
        metrics['mean_error_procrustes' + suffix] = tfu.reduce_mean_masked(
            dist_procrustes, inps.point_validity) * 1000

        # PROJECTION
        coords2d_pred = tfu3d.project_pose(preds.coords3d_abs, inps.intrinsics)
        coords2d_true = tfu3d.project_pose(inps.coords3d_true, inps.intrinsics)
        scale = 256 / FLAGS.proc_side
        metrics['mean_error_px' + suffix] = tfu.reduce_mean_masked(
            tf.norm((coords2d_true - coords2d_pred) * scale, axis=-1), inps.point_validity)

        return metrics

    def compute_metrics_with_2d_gt(self, inps, preds, suffix=''):
        metrics = EasyDict()
        scale = 256 / FLAGS.proc_side
        coords2d_pred = tfu3d.project_pose(preds.coords3d_abs, inps.intrinsics)
        metrics['mean_error_px' + suffix] = tfu.reduce_mean_masked(
            tf.norm((inps.coords2d_true - coords2d_pred) * scale, axis=-1), inps.point_validity)
        return metrics

    def my_norm(self, x, uncert=None):
        if uncert is None:
            return tfu.charbonnier(x, epsilon=2e-2, axis=-1)

        if FLAGS.nll_loss:
            dim = tf.cast(tf.shape(x)[-1], tf.float32)
            beta_comp_factor = tf.stop_gradient(
                uncert) ** FLAGS.beta_nll if FLAGS.beta_nll else 1

            factor = tf.math.rsqrt(dim) if FLAGS.fix_uncert_factor else tf.math.sqrt(dim)
            return (tfu.charbonnier(
                x / tf.expand_dims(uncert, -1),
                epsilon=FLAGS.charb_eps, axis=-1) + factor * tf.math.log(
                uncert)) * beta_comp_factor
        else:
            return tfu.charbonnier(x, epsilon=FLAGS.charb_eps, axis=-1)  # +

    def my_euc_norm(self, x):
        dim = tf.cast(tf.shape(x)[-1], tf.float32)
        norm = tfu.reduce_euclidean(x, keepdims=True) * tf.math.rsqrt(dim)
        return norm

    def adjusted_train_counter(self):
        return self.train_counter // FLAGS.grad_accum_steps

    def reconstruct_absolute(
            self, head2d, head3d, intrinsics, mix_3d_inside_fov, point_validity_mask=None):
        return tf.cond(
            self.adjusted_train_counter() < 500,
            lambda: tfu3d.reconstruct_absolute(
                head2d, head3d, intrinsics, mix_3d_inside_fov=mix_3d_inside_fov,
                weak_perspective=True, point_validity_mask=point_validity_mask,
                border_factor1=1, border_factor2=0.55, mix_based_on_3d=False),
            lambda: tfu3d.reconstruct_absolute(
                head2d, head3d, intrinsics, mix_3d_inside_fov=mix_3d_inside_fov,
                weak_perspective=False, point_validity_mask=point_validity_mask,
                border_factor1=1, border_factor2=0.55, mix_based_on_3d=False))

    def stop_grad_before_step(self, x, step):
        if self.adjusted_train_counter() >= step:
            return x

        if tf.reduce_all(tf.math.is_finite(x)):
            return tf.stop_gradient(x)
        else:
            return tf.zeros_like(x)


def interpolate_sparse(source_points, indices, data, dense_shape):
    nrows = indices.nrows(tf.int32)
    if nrows == 0:
        return tf.zeros([0, FLAGS.num_points, 3], tf.float32)

    row_ids = indices.value_rowids()
    stacked_indices = tf.concat([row_ids[:, tf.newaxis], indices.values], axis=1)
    stacked_shape = tf.concat([nrows[tf.newaxis], dense_shape[0]], axis=0)
    stacked_values = data.values
    stacked_sparse_weights_csr = tf.raw_ops.SparseTensorToCSRSparseMatrix(
        indices=tf.cast(stacked_indices, tf.int64),
        values=stacked_values,
        dense_shape=tf.cast(stacked_shape, tf.int64))
    result = tf.raw_ops.SparseMatrixMatMul(a=stacked_sparse_weights_csr, b=source_points)
    return tf.stop_gradient(result)
