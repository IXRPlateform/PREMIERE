import argparse

import numpy as np
import simplepyutils as spu
import tensorflow as tf
import tensorflow_hub as hub
from simplepyutils import FLAGS, logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model-path', type=str, required=True)
    parser.add_argument('--output-model-path', type=str, required=True)
    parser.add_argument('--detector-path', type=str)
    parser.add_argument('--bone-length-dataset', type=str)
    parser.add_argument('--bone-length-file', type=str)
    parser.add_argument('--skeleton-types-file', type=str)
    parser.add_argument('--joint-transform-file', type=str)
    parser.add_argument('--border-value', type=int, default=0)
    parser.add_argument('--backbone-link-dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--rot-aug', type=float, default=25)
    parser.add_argument('--rot-aug-360', action=spu.argparse.BoolAction)
    parser.add_argument('--rot-aug-360-half', action=spu.argparse.BoolAction)
    parser.add_argument('--detector-flip-vertical-too', action=spu.argparse.BoolAction)
    parser.add_argument('--return-crops', action=spu.argparse.BoolAction)
    parser.add_argument('--noise-border', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    #tf.config.run_functions_eagerly(True)

    crop_model = hub.load(FLAGS.input_model_path)
    detector = hub.load(FLAGS.detector_path) if FLAGS.detector_path else None

    skeleton_infos = spu.load_pickle(FLAGS.skeleton_types_file)
    import nlf.tf.multiperson.multiperson_model as multiperson_modellib
    model = multiperson_modellib.MultipersonNLF(crop_model, detector, skeleton_infos)

    tf.saved_model.save(
        model, FLAGS.output_model_path,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
        signatures=dict(
            detect_poses_batched=model.detect_poses_batched,
            estimate_poses_batched=model.estimate_poses_batched,
            detect_poses=model.detect_poses,
            estimate_poses=model.estimate_poses))
    #
    # model_loaded = tf.saved_model.load(FLAGS.output_model_path)
    # result = model_loaded.detect_poses_batched(im, weights=model.smpl_weights,
    #                                     intrinsic_matrix=cam.intrinsic_matrix[np.newaxis],
    #                                     num_aug=5,
    #                                     extrinsic_matrix=cam.get_extrinsic_matrix()[np.newaxis],
    #                                     distortion_coeffs=np.zeros(5, dtype=np.float32)[np.newaxis],
    #                                     world_up_vector=np.array([0, 1, 0], dtype=np.float32))
    # print(result)
    # print(result['poses3d'].shape, result['poses3d'].flat_values.shape)

    logger.info(f'Full image model has been exported to {FLAGS.output_model_path}')


if __name__ == '__main__':
    main()
