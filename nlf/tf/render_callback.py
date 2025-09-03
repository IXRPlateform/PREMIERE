import multiprocessing

import cameralib
import cv2
import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu
import tensorflow as tf
from simplepyutils import FLAGS

from nlf.rendering import Renderer
from nlf.paths import DATA_ROOT, PROJDIR


class RenderPredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_step=0, interval=100):
        super().__init__()
        image_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/wild_crops/*.*')
        self.image_stack_np = np.stack([
            cv2.resize(imageio.imread(p)[..., :3], (FLAGS.proc_side, FLAGS.proc_side))
            for p in image_paths], axis=0)
        self.camera = cameralib.Camera.from_fov(
            30, [FLAGS.proc_side, FLAGS.proc_side])
        self.image_stack_tf = tf.constant(self.image_stack_np, tf.float16) / 255

        self.intrinsics = tf.repeat(
            tf.constant(self.camera.intrinsic_matrix, tf.float32)[tf.newaxis],
            len(image_paths), axis=0)
        self.camera.scale_output(512 / FLAGS.proc_side)
        self.canonical_points = tf.constant(
            np.load(f'{PROJDIR}/canonical_vertices_smplx.npy'), tf.float32)
        self.faces = np.load(f'{PROJDIR}/smplx_faces.npy')
        self.q = multiprocessing.Queue(10)
        self.renderer_process = multiprocessing.Process(
            target=smpl_render_loop,
            args=(self.q, self.image_stack_np, self.camera, self.faces, FLAGS.logdir))
        self.renderer_process.start()
        self.start_step = start_step
        self.interval = interval

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.interval == 0 and batch >= self.start_step:
            step = batch // FLAGS.grad_accum_steps
            pred_vertices, uncerts = self.model.model.predict_multi_same_canonicals(
                self.image_stack_tf, self.intrinsics, self.canonical_points)
            pred_vertices = pred_vertices.numpy() / 1000
            self.q.put((step, pred_vertices))

    def on_train_end(self, logs=None):
        self.q.put(None)
        self.renderer_process.join()

    def __del__(self):
        if self.renderer_process.is_alive():
            self.q.put(None)
            self.renderer_process.join()


def smpl_render_loop(q, image_stack, camera, faces, logdir):
    renderer = Renderer(imshape=(512, 512), faces=faces)
    # import wandb
    # id_path = f'{logdir}/run_id'
    # with open(id_path) as f:
    #     run_id = f.read()
    # wandb.init(id=run_id, resume=True)

    image_stack = np.array(
        [cv2.resize(im, (512, 512), interpolation=cv2.INTER_CUBIC) for im in image_stack])

    while (elem := q.get()) is not None:
        batch, pred_vertices = elem
        triplets = [
            make_triplet(im, verts, renderer, camera)
            for im, verts in zip(image_stack, pred_vertices)]
        grid = np.concatenate(
            [np.concatenate(triplets[:3], axis=1),
             np.concatenate(triplets[3:6], axis=1),
             np.concatenate(triplets[6:9], axis=1)], axis=0)
        path = f'{logdir}/pred_{batch:07d}.jpg'
        imageio.imwrite(path, grid, quality=93)
        # wandb.log({'preds': wandb.Image(path)}, step=batch)


def alpha_blend(im1, im2, w1):
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    w1 = w1.astype(np.float32) / 255
    w1 = np.expand_dims(w1, axis=-1)
    res = im1 * w1 + im2 * (1 - w1)
    return np.clip(res, 0, 255).astype(np.uint8)


def make_triplet(image_original, pred_vertices, renderer, camera_front):
    mean = np.mean(pred_vertices, axis=0)
    image_front = renderer.render(pred_vertices, camera_front, RGBA=True)
    image_front = alpha_blend(image_front[..., :3], image_original, image_front[..., 3])
    camera_side = camera_front.orbit_around(mean, np.pi / 2, inplace=False)
    # move a bit further back from the mean, by 1.5 times the distance from the mean to the camera
    camera_side.t = camera_side.t + (camera_side.t - mean)
    image_side = renderer.render(pred_vertices, camera_side)
    return np.concatenate([image_original, image_front, image_side], axis=1)
