import argparse
import os.path as osp

import numpy as np
import simplepyutils as spu
import smpl.numpy
import smpl.tensorflow.fitting
import smpl.tensorflow.full_fitting
import tensorflow as tf
from simplepyutils import FLAGS

from nlf.paths import DATA_ROOT


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-pred-path', type=str, required=True)
    parser.add_argument('--out-pred-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-betas', type=int, default=10)
    parser.add_argument('--l2-regul', type=float, default=1)
    parser.add_argument('--l2-regul2', type=float, default=0)
    parser.add_argument('--num-iter', type=int, default=3)
    parser.add_argument('--testset-only', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    fit_fn = smpl.tensorflow.get_fit_fn(
        'smpl', 'neutral', num_betas=FLAGS.num_betas, enable_kid=False,
        weighted=True, l2_regularizer=FLAGS.l2_regul, l2_regularizer2=FLAGS.l2_regul2,
        num_iter=FLAGS.num_iter, final_adjust_rots=True)

    seq_filepaths = spu.sorted_recursive_glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    if FLAGS.testset_only:
        seq_filepaths = [p for p in seq_filepaths if spu.split_path(p)[-2] == 'test']
    seq_relpaths = [osp.relpath(p, f'{DATA_ROOT}/3dpw/sequenceFiles') for p in seq_filepaths]

    for seq_relpath in spu.progressbar(seq_relpaths):
        dicti = spu.load_pickle(f'{FLAGS.in_pred_path}/{seq_relpath}')

        def reshape(x):
            return np.reshape(x, [*dicti['vertices'].shape[:2], *x.shape[1:]])

        ds = tf.data.Dataset.from_tensor_slices(
            (dicti['vertices'].reshape(-1, 6890, 4),
             dicti['jointPositions'].reshape(-1, 24, 4),
             )).batch(FLAGS.batch_size).prefetch(1)

        results_all = []
        for vertices, joints in ds:
            v_w = get_weights(vertices[..., 3])
            j_w = get_weights(joints[..., 3])
            res = fit_fn(vertices[..., :3], joints[..., :3], v_w, j_w)
            res = tf.nest.map_structure(lambda x: x.numpy(), res)
            results_all.append(res)

        betas = np.concatenate([r['shape_betas'] for r in results_all], axis=0)
        pose = np.concatenate([r['pose_rotvecs'] for r in results_all], axis=0)
        trans = np.concatenate([r['trans'] for r in results_all], axis=0)
        spu.dump_pickle(
            dict(pose=reshape(pose), betas=reshape(betas), trans=reshape(trans)),
            f'{FLAGS.out_pred_path}/{seq_relpath}')


def get_weights(uncerts):
    w = uncerts ** -1.5
    return w / tf.reduce_mean(w, axis=-1, keepdims=True)


if __name__ == '__main__':
    main()
