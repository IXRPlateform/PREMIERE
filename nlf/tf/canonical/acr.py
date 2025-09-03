# Affine-combining regressor with regularization to minimize spatial support of weights.
import tensorflow as tf

import fleras
from affine_combining_autoencoder import AffineCombinationLayer
from fleras.easydict import EasyDict


class AffineCombiningRegressor(tf.keras.Model):
    def __init__(
            self, n_sided_joints, n_center_joints, n_latent_points_sided, n_latent_points_center,
            w_init=None, chiral=True):
        super().__init__()

        self.layer = AffineCombinationLayer(
            n_sided_joints, n_center_joints, n_latent_points_sided, n_latent_points_center,
            transposed=False, use_w_x=True, use_w_q=True, w_init=w_init, chiral=chiral)

    def call(self, inp):
        return self.layer(inp)


class AffineCombiningRegressorTrainer(fleras.ModelTrainer):
    def __init__(self, model, regul_lambda, supp_lambda, pose3d_template, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.regul_lambda = regul_lambda
        self.supp_lambda = supp_lambda
        self.pose3d_template = tf.convert_to_tensor(pose3d_template, dtype=tf.float32)

    def forward_train(self, inps, training):
        return dict(pose3d=self.model(inps.pose3d_in))

    def compute_losses(self, inps, preds):
        losses = EasyDict()
        x, y = splat(inps.pose3d_out, preds.pose3d)
        losses.main_loss = tf.reduce_mean(tf.abs(x - y))

        w = self.model.layer.get_w()
        mean_w = tf.reduce_mean(tf.reduce_sum(tf.abs(w), axis=0))

        losses.regul = mean_w
        losses.supp = mean_spatial_support(self.pose3d_template, w) * 1e-6
        losses.loss = (
                losses.main_loss +
                self.regul_lambda * losses.regul +
                self.supp_lambda * losses.supp)
        return losses

    def compute_metrics(self, inps, preds, training):
        m = EasyDict()
        x, y = splat(inps.pose3d_out, preds.pose3d)
        dist = tf.linalg.norm(x - y, axis=-1)
        m.pck1 = pck(dist, 0.01)
        m.pck2 = pck(dist, 0.02)
        m.pck3 = pck(dist, 0.03)
        m.pck7 = pck(dist, 0.07)
        m.euclidean = tf.reduce_mean(dist)
        m.l1 = tf.reduce_mean(tf.abs(x - y))
        m.max_supp = tf.reduce_max(
            tf.sqrt(spatial_support(self.pose3d_template, self.model.layer.get_w()))) / 1e3
        return m


def sum_squared_difference(a, b, axis=-1):
    return tf.reduce_sum(tf.math.squared_difference(a, b), axis=axis)


def mean_spatial_support(template, weights):
    weighted_mean = convert_pose(template, weights)
    sum_squared_diff = sum_squared_difference(
        template[tf.newaxis, :],
        weighted_mean[:, tf.newaxis])  # Jj
    return tf.einsum('Jj,jJ->', sum_squared_diff, tf.abs(weights)) / weights.shape[1]


def spatial_support(template, weights):
    weighted_mean = convert_pose(template, weights)
    sq_diff = tf.math.squared_difference(
        template[tf.newaxis, :],
        weighted_mean[:, tf.newaxis])  # Jjc
    sq_dists = tf.reduce_sum(sq_diff, axis=-1)  # Jj
    return tf.einsum('Jj,jJ->J', sq_dists, tf.abs(weights))



def convert_pose(pose, weights):
    return tf.matmul(weights, pose, transpose_a=True)


def pck(x, t):
    return tf.reduce_mean(tf.cast(x <= t, tf.float32))


def splat(x, y):
    z_mean = tf.reduce_mean(x[..., 2:], axis=1, keepdims=True)
    return x[..., :2] / x[..., 2:] * z_mean / 1000, y[..., :2] / y[..., 2:] * z_mean / 1000
