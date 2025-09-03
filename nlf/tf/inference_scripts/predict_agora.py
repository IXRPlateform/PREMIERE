import argparse
import functools
import glob
import os.path as osp
import pickle
import random
import zipfile

import cameralib
import humcentr_cli.util.serialization
import numpy as np
import poseviz
import simplepyutils as spu
import smpl.numpy
import smpl.tensorflow
import smpl.tensorflow.fitting
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_inputs as tfinp
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS, logger

from nlf.paths import PROJDIR


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--body-model', type=str, default='smpl')
    parser.add_argument('--default-fov', type=float, default=55)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--internal-batch-size', type=int, default=64)
    parser.add_argument('--antialias-factor', type=int, default=2)
    parser.add_argument('--detector-threshold', type=float, default=0.12)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    logger.info('Loading model...')
    model = tfhub.load(FLAGS.model_path)
    logger.info('Model loaded.')
    if FLAGS.body_model == 'smplx':
        cano_verts = np.load(f'{PROJDIR}/canonical_vertices_smplx.npy')
        cano_joints = np.load(f'{PROJDIR}/canonical_joints_smplx_144.npy')[:127]
        body_model = smpl.numpy.get_cached_body_model('smplx', 'neutral')
    else:
        cano_verts = np.load(f'{PROJDIR}/canonical_vertices_smpl.npy')
        cano_joints = np.load(f'{PROJDIR}/canonical_joints/smpl.npy')
        body_model = smpl.numpy.get_cached_body_model('smpl', 'neutral')

    cano_all = np.concatenate([cano_verts, cano_joints], axis=0)
    predict_fn = functools.partial(
        model.detect_poses_batched, internal_batch_size=FLAGS.internal_batch_size,
        detector_threshold=FLAGS.detector_threshold, detector_nms_iou_threshold=0.8,
        detector_flip_aug=True,
        antialias_factor=FLAGS.antialias_factor, num_aug=FLAGS.num_aug,
        weights=model.get_weights_for_canonical_points(cano_all),
        suppress_implausible_poses=True, default_fov_degrees=FLAGS.default_fov)

    image_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/agora/test/*.png')
    random.shuffle(image_paths)
    faces = body_model.faces

    image_path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    viz = poseviz.PoseViz(
        body_model_faces=faces, resolution=(1920, 1080), paused=True) if FLAGS.viz else None

    frame_batches_gpu, frame_batches_cpu = tfinp.image_files(
        image_paths, extra_data=image_path_ds, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz)
    camera = cameralib.Camera.from_fov(FLAGS.default_fov, (2160, 3840))

    with humcentr_cli.util.serialization.WriterPXZ(FLAGS.output_path, metadata=vars(FLAGS)) as writer:
        for (frame_batch_gpu, image_path_batch), frame_batch_cpu in spu.progressbar(
                zip(frame_batches_gpu, frame_batches_cpu),
                total=len(image_paths), step=FLAGS.batch_size):
            pred = predict_fn(frame_batch_gpu)
            pred['vertices'], pred['joints'] = tf.split(
                pred['poses3d'], [body_model.num_vertices, cano_joints.shape[0]], axis=2)
            pred['vertex_uncertainties'], pred['joint_uncertainties'] = tf.split(
                pred['uncertainties'], [body_model.num_vertices, cano_joints.shape[0]],
                axis=2)
            pred['vertices2d'], pred['joints2d'] = tf.split(
                pred['poses2d'], [body_model.num_vertices, cano_joints.shape[0]], axis=2)
            pred = tf.nest.map_structure(lambda x: x.numpy(), pred)

            for (verts_pred, joints_pred, verts_uncert, joints_uncert, joints2d_pred, boxes, frame,
                 image_path) in zip(
                pred['vertices'], pred['joints'], pred['vertex_uncertainties'],
                pred['joint_uncertainties'], pred['joints2d'], pred['boxes'], frame_batch_cpu,
                image_path_batch):

                image_name = osp.splitext(osp.basename(image_path.numpy().decode('utf8')))[0]
                writer.write_frame(
                    name=f'{image_name}.msg', camera=camera,
                    joints=joints_pred, vertices=verts_pred,
                    joint_uncertainties=joints_uncert, vertex_uncertainties=verts_uncert)

                if viz is not None:
                    viz.update(frame, boxes=boxes, camera=camera, vertices=verts_pred)

    if viz is not None:
        viz.close()


if __name__ == '__main__':
    main()
