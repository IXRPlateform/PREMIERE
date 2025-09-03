import argparse
import functools

import cameralib
import numpy as np
import poseviz
import simplepyutils as spu
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_inputs as tfinp
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS, logger

from nlf.tf.loading.common import recolor_border
from nlf.paths import PROJDIR


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--default-fov', type=float, default=55)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
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

    body_model_name = 'smpl'
    cano_verts = np.load(f'{PROJDIR}/canonical_verts/smpl.npy')
    cano_joints = np.load(f'{PROJDIR}/canonical_joints/{body_model_name}.npy')
    #vertex_subset = np.load(f'{DATA_ROOT}/body_models/smpl/vertex_subset.npy')
    vertex_subset = np.arange(6890)
    cano_all = np.concatenate([cano_verts[vertex_subset], cano_joints], axis=0)

    K = np.array([[5000, 0., 512 / 2.0], [0., 5000, 512 / 2.0], [0., 0., 1.]], np.float32)

    predict_fn = functools.partial(
        model.estimate_poses_batched, intrinsic_matrix=K[np.newaxis],
        internal_batch_size=FLAGS.internal_batch_size,
        antialias_factor=FLAGS.antialias_factor, num_aug=FLAGS.num_aug,
        weights=model.get_weights_for_canonical_points(cano_all))

    labels = np.load(f'{DATA_ROOT}/ssp_3d/labels.npz')
    centers = labels['bbox_centres']
    size = labels['bbox_whs'][:, np.newaxis]
    boxes = np.concatenate([centers - size / 2, size, size], axis=1).astype(np.float32)

    image_paths = [f'{DATA_ROOT}/ssp_3d/images/{n}' for n in labels['fnames']]
    image_path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    boxes_ds = tf.data.Dataset.from_tensor_slices(boxes)
    extra_data = tf.data.Dataset.zip((image_path_ds, boxes_ds))

    # the letterboxing border should be white for this model
    frame_batches_gpu, frame_batches_cpu = tfinp.image_files(
        image_paths, extra_data=extra_data, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz,
        frame_preproc_fn=functools.partial(recolor_border, border_value=(255, 255, 255))
    )

    result_verts_batches = []
    result_joints_batches = []
    result_vert_uncert_batches = []
    result_joint_uncert_batches = []

    camera = cameralib.Camera(intrinsic_matrix=K)
    faces = np.load(f'{PROJDIR}/smpl_faces.npy')
    viz = poseviz.PoseViz(body_model_faces=faces, paused=True) if FLAGS.viz else None

    for (frame_batch_gpu, (image_path_batch, boxes_batch)), frame_batch_cpu in spu.progressbar(
            zip(frame_batches_gpu, frame_batches_cpu), total=len(image_paths),
            step=FLAGS.batch_size):
        boxes_b = tf.RaggedTensor.from_tensor(boxes_batch[:, tf.newaxis])
        pred = predict_fn(frame_batch_gpu, boxes=boxes_b)
        pred['vertices'], pred['joints'] = tf.split(
            pred['poses3d'], [len(vertex_subset), len(cano_joints)], axis=2)
        pred['vertex_uncertainties'], pred['joint_uncertainties'] = tf.split(
            pred['uncertainties'], [len(vertex_subset), len(cano_joints)], axis=2)

        pred = tf.nest.map_structure(lambda x: tf.squeeze(x, 1).numpy(), pred)
        result_verts_batches.append(pred['vertices'])
        result_joints_batches.append(pred['joints'])
        result_vert_uncert_batches.append(pred['vertex_uncertainties'])
        result_joint_uncert_batches.append(pred['joint_uncertainties'])
        if viz is not None:
            for frame_cpu, box, vertices in zip(frame_batch_cpu, boxes_batch, pred['vertices']):
                viz.update(
                    frame=frame_cpu, boxes=box[np.newaxis], vertices=vertices[np.newaxis],
                    # vertices_alt=fit_vertices[np.newaxis],
                    camera=camera)
    if viz is not None:
        viz.close()

    np.savez(
        FLAGS.output_path,
        vertices=np.concatenate(result_verts_batches, axis=0) / 1000,
        joints=np.concatenate(result_joints_batches, axis=0) / 1000,
        vertex_uncertainties=np.concatenate(result_vert_uncert_batches, axis=0),
        joint_uncertainties=np.concatenate(result_joint_uncert_batches, axis=0),
    )


if __name__ == '__main__':
    main()
