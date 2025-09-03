import functools

import keras.layers
import fleras.layers
import numpy as np
import tensorflow as tf
from fleras.layers import GhostBatchNormalization, GhostBatchRenormalization
from keras.layers import Lambda
from keras.models import Sequential
from simplepyutils import FLAGS

from nlf.tf.backbones import mobilenet_v3, resnet
from nlf.tf.backbones.efficientnet import effnetv2_model
from nlf.tf.backbones.efficientnet import effnetv2_utils


def build_backbone():
    build_fn = get_build_fn()
    normalizer = get_normalizer()
    backbone, preproc_fn = build_fn(normalizer)

    preproc_layer = Lambda(preproc_fn, output_shape=lambda x: x)
    result = Sequential([preproc_layer, backbone])
    return result, normalizer


def get_build_fn():
    prefix_to_build_fn = dict(
        efficientnetv2=build_effnetv2, resnet=build_resnet, mobilenet=build_mobilenet)
    for prefix, build_fn in prefix_to_build_fn.items():
        if FLAGS.backbone.startswith(prefix):
            return build_fn

    raise Exception(f'No backbone builder found for {FLAGS.backbone}.')


def build_resnet(bn):
    try:
        from keras.src.layers import VersionAwareLayers
    except ModuleNotFoundError:
        from keras.layers import VersionAwareLayers

    class MyLayers(VersionAwareLayers):
        def __getattr__(self, name):
            if name == 'BatchNormalization':
                return bn
            return super().__getattr__(name)

    classname = f'ResNet{FLAGS.backbone[len("resnet"):]}'.replace('-', '_')
    backbone = getattr(resnet, classname)(
        include_top=False, weights='imagenet',
        input_shape=(None, None, 3), layers=MyLayers())
    if 'V2' in FLAGS.backbone:
        preproc_fn = tf_preproc
    elif 'V1-5' in FLAGS.backbone or 'V1_5' in FLAGS.backbone:
        preproc_fn = torch_preproc
    else:
        preproc_fn = caffe_preproc
    return backbone, preproc_fn


def build_effnetv2(bn):
    effnetv2_utils.set_batchnorm(bn)
    if FLAGS.constrain_kernel_norm != np.inf:
        model_config = dict(
            kernel_constraint=tf.keras.constraints.MinMaxNorm(
                0, FLAGS.constrain_kernel_norm, axis=[0, 1, 2],
                rate=FLAGS.constraint_rate),
            depthwise_constraint=tf.keras.constraints.MinMaxNorm(
                0, FLAGS.constrain_kernel_norm, axis=[0, 1],
                rate=FLAGS.constraint_rate))
    else:
        model_config = {}

    backbone = effnetv2_model.get_model(
        FLAGS.backbone, model_config=model_config, include_top=False)
    return backbone, tf_preproc


def build_mobilenet(bn):
    class MyLayers(mobilenet_v3.VersionAwareLayers):
        def __getattr__(self, name):
            if name == 'BatchNormalization':
                return bn
            return super().__getattr__(name)

    arch = FLAGS.backbone
    arch = arch[:-4] if arch.endswith('mini') else arch
    classname = f'MobileNet{arch[len("mobilenet"):]}'
    backbone = getattr(mobilenet_v3, classname)(
        include_top=False, weights='imagenet', minimalistic=FLAGS.backbone.endswith('mini'),
        input_shape=(FLAGS.proc_side, FLAGS.proc_side, 3), layers=MyLayers(),
        centered_stride=FLAGS.centered_stride, pooling=None)
    return backbone, mobilenet_preproc


def get_normalizer():
    if FLAGS.ghost_bn:
        clazz = GhostBatchRenormalization if FLAGS.batch_renorm else GhostBatchNormalization
        split = [int(x) for x in FLAGS.ghost_bn.split(',')]
        bn = functools.partial(clazz, split=split)
    elif FLAGS.batch_renorm:
        bn = fleras.layers.BatchRenormalization
    else:
        bn = keras.layers.BatchNormalization

    if FLAGS.backbone.startswith('efficientnetv2'):
        bn = functools.partial(bn, name='tpu_batch_normalization')

    def adjust_momentum(m):
        # Adjust for GPU count
        n_gpus = tf.distribute.get_strategy().num_replicas_in_sync
        m = 1 - ((1 - m) / n_gpus)

        # Adjust for gradient accumulation
        m **= 1 / FLAGS.grad_accum_steps
        return m

    def result(*args, momentum=None, **kwargs):
        if momentum is None:
            momentum = 0.99
        return bn(*args, momentum=adjust_momentum(momentum), **kwargs)

    if FLAGS.batch_renorm:
        renorm_clipping = dict(
            rmax=tf.Variable(1, dtype=tf.float32, trainable=False, name='rmax'),
            rmin=tf.Variable(1, dtype=tf.float32, trainable=False, name='rmin'),
            dmax=tf.Variable(0, dtype=tf.float32, trainable=False, name='dmax'))
        result = functools.partial(result, renorm_clipping=renorm_clipping)
        result.renorm_clipping = renorm_clipping
    return result


def torch_preproc(x):
    mean_rgb = tf.convert_to_tensor(np.array([0.485, 0.456, 0.406]), x.dtype)
    stdev_rgb = tf.convert_to_tensor(np.array([0.229, 0.224, 0.225]), x.dtype)
    normalized = (x - mean_rgb) / stdev_rgb
    return normalized


def caffe_preproc(x):
    mean_rgb = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68]), x.dtype)
    return tf.cast(255, x.dtype) * x - mean_rgb


def tf_preproc(x):
    x = tf.cast(2, x.dtype) * x - tf.cast(1, x.dtype)
    return x


def mobilenet_preproc(x):
    return tf.cast(255, x.dtype) * x
