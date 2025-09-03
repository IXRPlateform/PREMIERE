import argparse
import os.path as osp
import pickle
import zipfile
from posepile.ds.rich.main import make_joint_info
import cameralib
import cv2
import numpy as np
import poseviz
import simplepyutils as spu
import smplfitter.np
import smplfitter.tf
import tensorflow as tf
from humcentr_cli.util.serialization import Reader, Writer
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-pred-path', type=str, required=True)
    parser.add_argument('--out-pred-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-betas', type=int, default=10)
    parser.add_argument('--l2-regul', type=float, default=10)
    parser.add_argument('--fov', type=float, default=55)
    parser.add_argument('--body-model', type=str, default='smpl')
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    fit_fn = smplfitter.tf.get_fit_fn(
        FLAGS.body_model, 'neutral', num_betas=FLAGS.num_betas, enable_kid=True,
        weighted=True, kid_regularizer=0,
        l2_regularizer=FLAGS.l2_regul, l2_regularizer2=0,
        requested_keys=('vertices', 'joints'), final_adjust_rots=True, num_iter=3)

    body_model = smplfitter.np.get_cached_body_model(FLAGS.body_model, 'neutral')
    camera = cameralib.Camera.from_fov(FLAGS.fov, (2160, 3840))
    ds = dataset_from_file(FLAGS.in_pred_path, FLAGS.batch_size)
    ji = make_joint_info()
    ji = ji.select_joints(range(55))

    viz = poseviz.PoseViz(
        ji.names, ji.stick_figure_edges,
        body_model_faces=body_model.faces, resolution=(1920, 1080),
        paused=True) if FLAGS.viz else None

    with Writer(FLAGS.out_pred_path, vars(FLAGS)) as writer:
        for batch in spu.progressbar(ds, step=FLAGS.batch_size):
            verts = batch['vertices'] / 1000
            joints = batch['joints'] / 1000
            vertex_weights = get_weights(batch['vertex_uncertainties'])
            joint_weights = get_weights(batch['joint_uncertainties'][:, :, :body_model.num_joints])

            fit = fit_fn(
                verts, joints[:, :, :body_model.num_joints],
                vertex_weights, joint_weights)

            fit = tf.nest.map_structure(lambda x: x.numpy(), fit)

            for filename, v_np, j_np, v_fit, j_fit in zip(
                    batch['filename'], verts, joints, fit['vertices'], fit['joints']):
                filename = filename.numpy().decode("utf8")

                j_others_fitaligned = j_np[:, body_model.num_joints:] - j_np[:, :1] + j_fit[:, :1]
                j_fit = np.concatenate([j_fit, j_others_fitaligned], axis=-2)

                writer.write_frame(filename=filename, vertices=v_fit * 1000, joints=j_fit * 1000)

                if viz is not None:
                    im_filename = spu.replace_extension(filename, '.png')
                    im_filepath = osp.join(DATA_ROOT, 'agora/test', im_filename)
                    image = cv2.imread(im_filepath)[..., ::-1]
                else:
                    image = None

                if viz is not None:
                    viz.update(
                        image, camera=camera, vertices=v_fit * 1000, poses=j_fit * 1000,
                        vertices_alt=v_np * 1000, poses_alt=j_np * 1000)
                    viz._join()

    if viz is not None:
        viz.close()


def get_weights(x):
    w = x ** -1.5
    return w / tf.reduce_mean(w, axis=-1, keepdims=True)


def dataset_from_file(filepath, batch_size):
    import fleras.parallel_map2

    def generator():
        ragged_tensor_names = ['vertices', 'joints', 'vertex_uncertainties', 'joint_uncertainties']
        with Reader(filepath) as f:
            for frame_data in f:
                out_frame_data = {}
                for name in ragged_tensor_names:
                    out_frame_data[f'_ragged_{name}'] = frame_data[name].astype(np.float32)
                out_frame_data['filename'] = frame_data['filename']
                yield out_frame_data

    return fleras.parallel_map2.iterable_to_batched_tf_dataset(
        generator(), drop_remainder=False, batch_size=batch_size)


if __name__ == '__main__':
    main()
