import argparse
import collections
import functools
import os.path as osp

import boxlib
import cameralib
import numpy as np
import poseviz
import simplepyutils as spu
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_inputs as tfinp
from simplepyutils import FLAGS, logger

from nlf.paths import DATA_ROOT, PROJDIR


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--out-video-dir', type=str)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=0)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    model = tfhub.load(FLAGS.model_path)

    faces = np.load(f'{PROJDIR}/smpl_faces.npy')
    cano_verts = np.load(f'{PROJDIR}/canonical_verts/smpl.npy')
    cano_joints = np.load(f'{PROJDIR}/canonical_joints/smpl.npy')
    cano_all = np.concatenate([cano_verts, cano_joints], axis=0)
    predict_fn = tf.function(functools.partial(
        model.estimate_poses_batched, internal_batch_size=FLAGS.internal_batch_size,
        num_aug=FLAGS.num_aug, antialias_factor=2,
        weights=model.get_weights_for_canonical_points(cano_all)))
    detect_fn = functools.partial(
        model.detector.predict_multi_image, threshold=0.2, nms_iou_threshold=0.7, flip_aug=True)

    viz = poseviz.PoseViz(body_model_faces=faces) if FLAGS.viz else None

    all_emdb_pkl_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/emdb/**/*_data.pkl')
    emdb1_sequence_roots = [
        osp.dirname(p) for p in all_emdb_pkl_paths
        if spu.load_pickle(p)['emdb1']]

    results_all = collections.defaultdict(list)

    for seq_root in emdb1_sequence_roots:
        seq_name = osp.basename(seq_root)
        logger.info(f'Predicting {seq_name}...')
        subj = seq_root.split('/')[-2]
        seq_data = spu.load_pickle(f'{seq_root}/{subj}_{seq_name}_data.pkl')
        frame_paths = [f'{seq_root}/images/{i_frame:05d}.jpg'
                       for i_frame in range(seq_data['n_frames'])]
        bboxes = seq_data['bboxes']['bboxes']
        bboxes = np.concatenate(
            [bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]], axis=1).astype(np.float32)

        if FLAGS.viz:
            viz.reinit_camera_view()
            if FLAGS.out_video_dir:
                viz.new_sequence_output(
                    f'{FLAGS.out_video_dir}/{seq_name}.mp4', fps=30)

        box_ds = tf.data.Dataset.from_tensor_slices(bboxes)
        ds, frame_batches_cpu = tfinp.image_files(
            frame_paths, extra_data=box_ds, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz)

        results_seq = predict_sequence(
            predict_fn, detect_fn, ds, frame_batches_cpu, len(frame_paths), viz)
        results_all[f'{subj}_{seq_name}'] = results_seq

    spu.dump_pickle(results_all, FLAGS.output_path)

    if FLAGS.viz:
        viz.close()


def predict_sequence(predict_fn, detect_fn, dataset, frame_batches_cpu, n_frames, viz):
    result_batches = dict(vertices=[], joints=[])
    camera = cameralib.Camera.from_fov(
        55, imshape=dataset.element_spec[0].shape[1:3])

    for (frames_b, box_b), frames_b_cpu in zip(
            spu.progressbar(dataset, total=n_frames, step=FLAGS.batch_size), frame_batches_cpu):
        boxes_det = detect_fn(frames_b)[:, :, :4]
        boxes_selected = select_boxes(box_b.numpy(), boxes_det)
        boxes_b = tf.RaggedTensor.from_tensor(boxes_selected[:, tf.newaxis])
        pred = predict_fn(frames_b, boxes_b)
        pred = tf.nest.map_structure(lambda x: tf.squeeze(x, 1), pred)
        vertices_b, joints_b = tf.split(pred['poses3d'], [6890, 24], axis=1)
        vu_b, ju_b = tf.split(pred['uncertainties'], [6890, 24], axis=1)
        vertices_b = tf.concat([vertices_b, vu_b[..., tf.newaxis]], axis=-1)
        joints_b = tf.concat([joints_b, ju_b[..., tf.newaxis]], axis=-1)
        result_batches['vertices'].append(vertices_b.numpy())
        result_batches['joints'].append(joints_b.numpy())

        if FLAGS.viz:
            for frame, box, vertices in zip(frames_b_cpu, boxes_selected, vertices_b[..., :3]):
                viz.update(frame, box[np.newaxis], vertices=vertices[np.newaxis], camera=camera)

    return {k: np.concatenate(v, axis=0) for k, v in result_batches.items()}


def select_boxes(boxes_gt_batch, boxes_det_batch):
    result = []
    for box_gt, boxes_det in zip(boxes_gt_batch, boxes_det_batch):
        if boxes_det.shape[0] == 0:
            result.append(box_gt)
            continue

        ious = [boxlib.iou(box_gt, box_det) for box_det in boxes_det]
        max_iou = np.max(ious)
        if max_iou > 0.1:
            result.append(boxes_det[np.argmax(ious)].numpy())
        else:
            result.append(box_gt)
    return np.array(result)


if __name__ == '__main__':
    main()
