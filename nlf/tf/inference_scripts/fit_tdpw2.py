import argparse
import os.path as osp

import numpy as np
import simplepyutils as spu
import tensorflow as tf
from humcentr_cli.util.serialization import simple_dataset_from_file
from simplepyutils import FLAGS

import smpl.numpy
import smpl.tensorflow.fitting
import smpl.tensorflow.full_fitting
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
        fit_sequence(
            fit_fn, spu.replace_extension(f'{FLAGS.in_pred_path}/{seq_relpath}', '.xz'),
            f'{FLAGS.out_pred_path}/{seq_relpath}')


def fit_sequence(fit_fn, in_path, out_path):
    ds = simple_dataset_from_file(in_path).batch(FLAGS.batch_size).prefetch(1)
    results_all = []
    for d in ds:
        res = fit_fn(
            d['vertices']/1000, d['joints']/1000,
            get_weights(d['vertex_uncertainties']), get_weights(d['joint_uncertainties']))
        res = tf.nest.map_structure(lambda x: x.numpy(), res)
        results_all.append(res)

    betas = np.concatenate([r['shape_betas'] for r in results_all], axis=0).transpose(1, 0, 2)
    pose = np.concatenate([r['pose_rotvecs'] for r in results_all], axis=0).transpose(1, 0, 2)
    trans = np.concatenate([r['trans'] for r in results_all], axis=0).transpose(1, 0, 2)
    spu.dump_pickle(dict(pose=pose, betas=betas, trans=trans), out_path)


def get_weights(uncerts):
    w = uncerts ** -1.5
    return w / tf.reduce_mean(w, axis=-1, keepdims=True)


if __name__ == '__main__':
    main()
