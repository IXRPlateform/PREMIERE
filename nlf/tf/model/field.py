import numpy as np
import tensorflow as tf
from simplepyutils import FLAGS

from nlf.paths import PROJDIR
from nlf.tf import tfu


def build_field():
    layer_dims = (
            [FLAGS.field_hidden_size] * FLAGS.field_hidden_layers +
            [(FLAGS.backbone_link_dim + 1) * (FLAGS.depth + 2)])

    if FLAGS.field_type == 'vanilla':
        gps_mlp = BaseModelVanilla(
            pos_enc_dim=512, hidden_dim=2048, output_dim=FLAGS.field_posenc_dim)
    else:
        gps_mlp = GPSBaseModel(
            pos_enc_dim=512, hidden_dim=2048, output_dim=FLAGS.field_posenc_dim)

    canonical_points = tf.keras.Input(shape=(None, 3), dtype=tfu.get_dtype())
    gps_mlp(canonical_points, training=False)

    if FLAGS.field_type == 'gps':
        gps_mlp_loaded = tf.keras.models.load_model(
            f'{PROJDIR}/lbo_mlp_512fourier_2048gelu_1024', compile=False)
        gps_mlp.set_weights(gps_mlp_loaded.get_weights())

    return GPSField(gps_mlp, layer_dims=layer_dims)


# Field with positional encoding based on the global point signature (GPS).
class GPSField(tf.keras.Model):
    def __init__(self, lbo_mlp, layer_dims):
        super().__init__()
        self.lbo_mlp = lbo_mlp
        self.lbo_mlp.trainable = FLAGS.finetune_gps

        self.eigva = np.load(f'{PROJDIR}/canonical_eigval3.npy')[1:].astype(np.float32)
        first_init = 'glorot_uniform'
        first_regul = FirstRegul(FLAGS.field_posenc_dim, FLAGS.lbo_eigval_regul)

        self.hidden_layers = [
            tf.keras.layers.Dense(
                d, activation=tf.nn.gelu,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=first_regul if i == 0 else None)
            for i, d in enumerate(layer_dims[:-1])]
        self.output_layer = tf.keras.layers.Dense(
            layer_dims[-1], activation=None,
            kernel_initializer=first_init if not self.hidden_layers else 'glorot_uniform',
            kernel_regularizer=first_regul if not self.hidden_layers else None)

        self.out_dim = layer_dims[-1]
        self.r_sqrt_eigva = np.sqrt(1.0 / np.load(f'{PROJDIR}/canonical_eigval3.npy')[1:]).astype(
            np.float32)

    def call(self, inp, training):
        lbo = self.lbo_mlp(
            tf.reshape(tf.cast(inp, tf.float16), [-1, 3]), training=FLAGS.finetune_gps)[...,
              :FLAGS.field_posenc_dim]
        if not FLAGS.finetune_gps:
            lbo = tf.stop_gradient(lbo)

        inp_shape = tf.shape(inp)
        lbo = tf.reshape(lbo, tf.concat([inp_shape[:-1], [FLAGS.field_posenc_dim]], axis=0))
        tf.debugging.assert_type(lbo, tfu.get_dtype())
        lbo = lbo * tf.cast(self.r_sqrt_eigva[:FLAGS.field_posenc_dim], lbo.dtype) * 0.1

        x = lbo
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

# This is a model that is separately pretrained to predict the GPS positional encoding.
# We first computed the GPS at randomly sampled discrete positions inside the canonical volume,
# then trained this MLP architecture with learnable Fourier features to predict the GPS at these
# positions. After training this network, we can compute the GPS at any point in the canonical
# volume. Furthermore, the weights of this network can also be fine-tuned during the training of
# the main model.
class GPSBaseModel(tf.keras.Model):
    def __init__(self, pos_enc_dim=512, hidden_dim=2048, output_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.pos_enc_dim = pos_enc_dim
        self.factor = 1 / np.sqrt(np.float32(self.pos_enc_dim))
        self.W_r = tf.keras.layers.Dense(
            self.pos_enc_dim // 2, use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=12))
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.dense2 = tf.keras.layers.Dense(output_dim)

        nodes = np.load(f'{PROJDIR}/canonical_nodes3.npy')
        self.mini = np.min(nodes, axis=0)
        self.maxi = np.max(nodes, axis=0)
        self.center = (self.mini + self.maxi) / 2

    def call(self, inp, training):
        x = (inp - self.center) / (self.maxi - self.mini)
        x = self.W_r(x)
        x = tf.sin(tf.concat([x, x + np.pi / 2], axis=-1)) * self.factor
        x = self.dense1(x)
        return self.dense2(x)


# The  model uses no positional encoding, it is just an MLP on top of the XYZ coordinates.
# The field names are kept in analogy with the GPS model.
class BaseModelVanilla(tf.keras.Model):
    def __init__(self, pos_enc_dim=512, hidden_dim=2048, output_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.pos_enc_dim = pos_enc_dim
        self.factor = 1 / np.sqrt(np.float32(self.pos_enc_dim))
        self.W_r = tf.keras.layers.Dense(self.pos_enc_dim, activation=tf.nn.gelu)
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.dense2 = tf.keras.layers.Dense(output_dim)

        nodes = np.load(f'{PROJDIR}/canonical_nodes3.npy')
        self.mini = np.min(nodes, axis=0)
        self.maxi = np.max(nodes, axis=0)
        self.center = (self.mini + self.maxi) / 2

    def call(self, inp, training):
        x = (inp - self.center) / (self.maxi - self.mini)
        x = self.W_r(x) * self.factor
        x = self.dense1(x)
        return self.dense2(x)



class FirstRegul(tf.keras.regularizers.Regularizer):
    def __init__(self, n_regularized, factors):
        self.n_regularized = n_regularized
        self.factors = factors

    def __call__(self, x):
        return tf.reduce_mean(
            tf.reduce_sum(self.factors * tf.square(x[:self.n_regularized]), axis=1), axis=0)
