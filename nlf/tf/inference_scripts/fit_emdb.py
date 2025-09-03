import argparse

import numpy as np
import simplepyutils as spu
import smplfitter.tf
import tensorflow as tf
from simplepyutils import FLAGS


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-pred-path', type=str, required=True)
    parser.add_argument('--out-pred-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-betas', type=int, default=10)
    parser.add_argument('--l2-regul', type=float, default=1)  # e-1)
    parser.add_argument('--num-iter', type=int, default=3)
    spu.argparse.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()

    fit_fn = smplfitter.tf.get_fit_fn(
        'smpl', 'neutral', num_betas=FLAGS.num_betas, enable_kid=False,
        weighted=True, l2_regularizer=FLAGS.l2_regul, num_iter=FLAGS.num_iter,
        final_adjust_rots=True)

    data = spu.load_pickle(FLAGS.in_pred_path)

    output = {}
    for seq_id, dicti in spu.progressbar_items(data):
        ds = tf.data.Dataset.from_tensor_slices(
            (dicti['vertices'], dicti['joints'])).batch(FLAGS.batch_size).prefetch(1)

        results_all = []
        for vertices, joints in ds:
            v_w = get_weights(vertices[..., 3])
            j_w = get_weights(joints[..., 3])
            fit = fit_fn(vertices[..., :3] / 1000, joints[..., :3] / 1000, v_w, j_w)
            fit = tf.nest.map_structure(lambda x: x.numpy(), fit)
            results_all.append(fit)

        betas = np.concatenate([r['shape_betas'] for r in results_all], axis=0)
        pose = np.concatenate([r['pose_rotvecs'] for r in results_all], axis=0)
        trans = np.concatenate([r['trans'] for r in results_all], axis=0)
        output[seq_id] = dict(pose=pose, betas=betas, trans=trans)

    spu.dump_pickle(output, FLAGS.out_pred_path)


def get_weights(uncerts):
    w = uncerts ** -1.5
    return w / tf.reduce_mean(w, axis=-1, keepdims=True)


if __name__ == '__main__':
    main()
