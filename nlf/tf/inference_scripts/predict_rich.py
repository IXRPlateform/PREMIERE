import argparse
import datetime
import functools
import os.path as osp
import queue
import sys
import threading

import cameralib
import numpy as np
import poseviz
import simplepyutils as spu
import smplfitter.np
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_inputs as tfinp
from humcentr_cli.util.serialization import Writer
from simplepyutils import FLAGS

from nlf.paths import DATA_ROOT, PROJDIR


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=40)
    parser.add_argument('--antialias-factor', type=int, default=2)
    parser.add_argument('--downscale-factor', type=int, default=1)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    parser.add_argument('--skip-existing', action=spu.argparse.BoolAction, default=True)
    spu.argparse.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    model = tfhub.load(FLAGS.model_path)

    cano_verts = np.load(f'{PROJDIR}/canonical_vertices_smplx.npy')
    cano_joints = np.load(f'{PROJDIR}/canonical_joints_smplx_144.npy')
    body_model = smplfitter.np.get_cached_body_model('smplx', 'neutral')
    faces = body_model.faces
    cano_all = np.concatenate([cano_verts, cano_joints], axis=0)

    predict_fn = tf.function(functools.partial(
        model.detect_poses_batched, internal_batch_size=FLAGS.internal_batch_size,
        detector_threshold=0.2, detector_nms_iou_threshold=0.7, detector_flip_aug=True,
        antialias_factor=FLAGS.antialias_factor, num_aug=FLAGS.num_aug,
        weights=model.get_weights_for_canonical_points(cano_all)))
    viz = poseviz.PoseViz(body_model_faces=faces, downscale=8) if FLAGS.viz else None

    all_frame_relpaths = get_all_frame_relpaths()
    all_frame_paths = [f'{DATA_ROOT}/rich/test/{p}' for p in all_frame_relpaths]
    all_frame_paths = sorted(all_frame_paths)

    image_path_ds = tf.data.Dataset.from_tensor_slices(all_frame_paths)
    ds, frame_batches_cpu = tfinp.image_files(
        all_frame_paths, extra_data=image_path_ds, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz,
        downscale_factor=FLAGS.downscale_factor)

    q = queue.Queue(FLAGS.batch_size * 2)
    writer = threading.Thread(target=write_main, args=(q,))
    writer.start()

    for (frame_batch_gpu, image_path_batch), frame_batch_cpu in zip(
            spu.progressbar(ds, total=len(all_frame_paths), step=FLAGS.batch_size),
            frame_batches_cpu):
        pred = predict_fn(frame_batch_gpu)
        verts_b, joints_b = tf.split(
            pred['poses3d'], [cano_verts.shape[0], cano_joints.shape[0]], axis=-2)
        boxes_b = pred['boxes']
        uncerts_verts_b, uncerts_joints_b = tf.split(
            pred['uncertainties'], [cano_verts.shape[0], cano_joints.shape[0]], axis=-1)

        for frame, boxes, joints, vertices, uncerts_verts, uncerts_joints, image_path in zip(
                frame_batch_cpu, boxes_b, joints_b, verts_b, uncerts_verts_b, uncerts_joints_b,
                image_path_batch):
            camera = cameralib.Camera.from_fov(
                55, imshape=frame_batch_gpu.shape[1:3])
            image_path = image_path.numpy().decode('utf-8')
            q.put((image_path, dict(
                boxes=boxes.numpy() * FLAGS.downscale_factor,
                joints=joints.numpy(),
                vertices=vertices.numpy(),
                joint_uncertainties=uncerts_joints.numpy(),
                vertex_uncertainties=uncerts_verts.numpy(),
                camera=camera.scale_output(FLAGS.downscale_factor, inplace=False))))

            if viz is not None:
                viz.update(frame, boxes=boxes, camera=camera, vertices=vertices)

    q.put(None)
    writer.join()

    if FLAGS.viz:
        viz.close()


def write_main(q):
    writer = None
    while (data := q.get()) is not None:
        image_path, frame_data = data
        dirpath = osp.dirname(image_path)
        seq_id = osp.relpath(dirpath, f'{DATA_ROOT}/rich/test')
        output_path = osp.normpath(osp.join(FLAGS.output_path, f'{seq_id}.xz'))
        if writer is None or writer.f.name != output_path:
            if writer is not None:
                writer.close()
            metadata = dict(
                config=vars(FLAGS), argv=sys.argv,
                created_time=datetime.datetime.now().isoformat())
            spu.ensure_parent_dir_exists(output_path)
            spu.logger.info(f'Writing {seq_id}')
            writer = Writer(output_path, metadata=metadata)
        writer.write_frame(**frame_data)
        q.task_done()

    if writer is not None:
        writer.close()
    q.task_done()


@spu.picklecache('rich_frame_paths.pkl')
def get_all_frame_relpaths():
    paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/rich/test/**/*.jpeg')
    return [osp.relpath(p, f'{DATA_ROOT}/rich/test') for p in paths]


if __name__ == '__main__':
    main()
