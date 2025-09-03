import argparse

import numpy as np
import simplepyutils as spu
import smpl.tensorflow.full_fitting
import tensorflow as tf
from simplepyutils import FLAGS, logger
import h5py


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-pred-path', type=str, required=True)
    parser.add_argument('--body-model', type=str, default='smpl')
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--enable-kid', action=spu.argparse.BoolAction)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-betas', type=int, default=32)
    parser.add_argument('--l2-regul', type=float, default=5e-2)
    spu.argparse.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    fit_fn = get_fit_fn(
        body_model_name=FLAGS.body_model, gender=FLAGS.gender, num_betas=FLAGS.num_betas,
        enable_kid=FLAGS.enable_kid,
        l2_regularizer=FLAGS.l2_regul)

    ds, n_items = hdf5_as_tf_dataset(FLAGS.in_path, ('verts', 'joints'))
    ds = ds.batch(FLAGS.batch_size)
    ds = ds.apply(tf.data.experimental.prefetch_to_device('GPU:0', 1))

    results_all = []
    for batch in spu.progressbar(ds, total=n_items, step=FLAGS.batch_size):
        res = fit_fn(batch['verts'], batch['joints'])
        res = tf.nest.map_structure(lambda x: x.numpy(), res)
        results_all.append(res)

    betas = np.concatenate([r['shape_betas'] for r in results_all], axis=0)
    pose = np.concatenate([r['pose_rotvecs'] for r in results_all], axis=0)
    trans = np.concatenate([r['trans'] for r in results_all], axis=0)
    joints = np.concatenate([r['joints'] for r in results_all], axis=0)
    np.savez(FLAGS.out_pred_path, betas=betas, pose=pose, trans=trans, joints=joints)


def get_fit_fn(
        body_model_name='smpl', gender='neutral', num_betas=10, enable_kid=False,
        requested_keys=('pose_rotvecs', 'trans', 'shape_betas', 'joints'), l2_regularizer=5e-3):
    body_model = smpl.tensorflow.get_cached_body_model(body_model_name, gender)

    fitter = smpl.tensorflow.full_fitting.Fitter(
        body_model, body_model.J_regressor, num_betas=num_betas, enable_kid=enable_kid)

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, body_model.num_vertices, 3], tf.float32),
            tf.TensorSpec([None, body_model.num_joints, 3], tf.float32)])
    def fit_fn(verts, joints):
        res = fitter.fit(
            verts, 3, l2_regularizer=l2_regularizer, joints_to_fit=joints,
            requested_keys=requested_keys)
        return {k: v for k, v in res.items() if v is not None}

    return fit_fn


def hdf5_as_tf_dataset(filepath, tensor_names):
    def generator():
        with h5py.File(filepath, 'r') as f:
            for arrays in zip(*[f[name] for name in tensor_names]):
                yield dict(zip(tensor_names, arrays))

    with h5py.File(filepath, 'r') as f:
        output_signature = {
            name: tf.TensorSpec(shape=f[name].shape[1:], dtype=f[name].dtype)
            for name in tensor_names}
        print(f['verts'].shape)
        n_elements = f[tensor_names[0]].shape[0]

    ds = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature)
    return ds, n_elements


if __name__ == '__main__':
    main()
