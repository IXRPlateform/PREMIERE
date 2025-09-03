import argparse
import functools
import os.path as osp
import random

import cameralib
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_inputs as tfinp
from posepile.paths import DATA_ROOT

import poseviz
import simplepyutils as spu
import smpl.numpy
import smpl.tensorflow
import smpl.tensorflow.fitting
from nlf.paths import PROJDIR
from simplepyutils import FLAGS, logger


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--default-fov', type=float, default=55)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--internal-batch-size', type=int, default=64)
    parser.add_argument('--antialias-factor', type=int, default=2)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    logger.info('Loading model...')
    model = tfhub.load(FLAGS.model_path)
    logger.info('Model loaded.')

    cano_verts = np.load(f'{PROJDIR}/canonical_vertices_smplx.npy')
    cano_joints = np.load(f'{PROJDIR}/canonical_joints_smplx_144.npy')[:127]
    body_model = smpl.numpy.get_cached_body_model('smplx', 'neutral')

    cano_all = np.concatenate([cano_verts, cano_joints], axis=0)
    camera = get_camera()
    predict_fn = tf.function(functools.partial(
        model.detect_poses_batched, internal_batch_size=FLAGS.internal_batch_size,
        detector_threshold=0.2, detector_nms_iou_threshold=0.7, detector_flip_aug=True,
        antialias_factor=FLAGS.antialias_factor, num_aug=FLAGS.num_aug,
        suppress_implausible_poses=True, intrinsic_matrix=camera.intrinsic_matrix[np.newaxis],
        max_detections=1, extrinsic_matrix=camera.get_extrinsic_matrix()[np.newaxis],
        world_up_vector=(0, 1, 0),
        weights=model.get_weights_for_canonical_points(cano_all)))

    image_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/ehf/*_img.png')
    random.shuffle(image_paths)

    image_path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    viz = poseviz.PoseViz(
        [], [], body_model_faces=body_model.faces,
        resolution=(1920, 1080), paused=True, ground_plane_height=-791.782,
        world_up=(0, 1, 0)) if FLAGS.viz else None

    ds, frame_batches_cpu = tfinp.image_files(
        image_paths, extra_data=image_path_ds, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz)

    results = {}
    for (frame_batch_gpu, image_path_batch), frame_batch_cpu in spu.progressbar(
            zip(ds, frame_batches_cpu),
            total=len(image_paths) // FLAGS.batch_size):
        pred = predict_fn(frame_batch_gpu)
        verts_b, joints_b = tf.split(
            pred['poses3d'], [cano_verts.shape[0], cano_joints.shape[0]], axis=-2)
        boxes_b = pred['boxes']
        uncerts_verts_b, uncerts_joints_b = tf.split(
            pred['uncertainties'], [cano_verts.shape[0], cano_joints.shape[0]], axis=-1)

        for frame, boxes, joints, vertices, uncerts_verts, uncerts_joints, image_path in zip(
                frame_batch_cpu, boxes_b, joints_b, verts_b, uncerts_verts_b, uncerts_joints_b,
                image_path_batch):

            imname = osp.basename(image_path.numpy().decode('utf-8'))
            results[imname] = dict(
                vertices=vertices[0].numpy(),
                joints=joints[0].numpy(),
                vertex_uncertainties=uncerts_verts[0].numpy(),
                joint_uncertainties=uncerts_joints[0].numpy()) if len(joints) > 0 else None

            if viz is not None:
                viz.update(frame, boxes=boxes, camera=camera, vertices=vertices)

    spu.dump_pickle(results, FLAGS.output_path)

    if viz is not None:
        viz.close()


def get_camera():
    f = 1498.22426237
    cx = 790.263706
    cy = 578.90334
    R = cv2.Rodrigues(np.array([-2.98747896, 0.01172457, -0.05704687], np.float32))[0]
    return cameralib.Camera(
        intrinsic_matrix=np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32),
        rot_world_to_cam=R,
        trans_after_rot=np.array([-0.03609917, 0.43416458, 2.37101226], np.float32),
        world_up=(0, 1, 0))


if __name__ == '__main__':
    main()
