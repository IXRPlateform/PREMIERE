from nlf.tf import init

"separator"
import fleras
from simplepyutils import logger
import numpy as np
import posepile.datasets2d as ds2d
import posepile.datasets3d as ds3d
import simplepyutils as spu
import tensorflow as tf
from simplepyutils import FLAGS

import nlf.tf.model.field as lf_field
import nlf.tf.render_callback as render_callback
import nlf.tf.model.nlf_trainer as lf_trainer
import nlf.tf.backbones.builder as backbone_builder
from nlf.tf.loading.densepose import load_dense
from nlf.tf.loading.keypoints2d import load_2d
from nlf.tf.loading.keypoints3d import load_kp
from nlf.tf.loading.parametric import load_parametric
from nlf.paths import DATA_ROOT, PROJDIR
from nlf.tf import tfu
from nlf.tf.util import TRAIN, VALID, TEST

np.complex = complex


def load_simple(ex, name, *args, **kwargs):
    return {name: dict(image_path=ex.image_path)}


def main():
    init.initialize()

    if FLAGS.train:
        job = LocalizerFieldJob()
        job.train()
    elif FLAGS.export_file:
        job = LocalizerFieldJob()
        job.restore()
        job.export(FLAGS.export_file)


class LocalizerFieldJob(fleras.TrainingJob):
    def __init__(self):
        super().__init__(
            wandb_project=FLAGS.wandb_project, wandb_config=FLAGS, logdir=FLAGS.logdir,
            init_path=FLAGS.init_path, load_path=FLAGS.load_path,
            training_steps=FLAGS.training_steps, grad_accum_steps=FLAGS.grad_accum_steps,
            force_grad_accum=FLAGS.force_grad_accum, loss_scale=FLAGS.loss_scale,
            dynamic_loss_scale=FLAGS.dynamic_loss_scale, ema_momentum=FLAGS.ema_momentum,
            finetune_in_inference_mode=FLAGS.finetune_in_inference_mode,
            validate_period=FLAGS.validate_period, checkpoint_dir=FLAGS.checkpoint_dir,
            checkpoint_period=FLAGS.checkpoint_period, multi_gpu=FLAGS.multi_gpu, seed=FLAGS.seed,
            n_completed_steps=FLAGS.completed_steps, parallel_build_data=True,
            stop_step=FLAGS.stop_step)

    def build_data(self):
        if 'finetune_3dpw' in FLAGS.custom:
            ds_parts_param = dict(
                tdpw=2 * 97, agora=8, bedlam=30, rich=6, behave=5, spec=4, surreal=8, moyo=8,
                arctic=5, intercap=5, genebody=0, egobody=3, hi4d_down=3, hi4d_rerender=5,
                humman=2, synbody_humannerf=5, thuman2=6, zjumocap=3, dfaust_render=3)
        elif 'finetune_agora' in FLAGS.custom:
            ds_parts_param = dict(
                agora=8 + 2 * 97, bedlam=30, rich=6, behave=5, spec=4, surreal=8, moyo=8,
                arctic=5, intercap=5, genebody=0, egobody=3, hi4d_down=3, hi4d_rerender=5,
                humman=2, synbody_humannerf=5, thuman2=6, zjumocap=3, dfaust_render=3)
        elif 'subset_real' in FLAGS.custom:
            ds_parts_param = dict(
                rich=5, behave=5, moyo=8,
                arctic=3, intercap=3, genebody=1, egobody=1, hi4d_down=3,
                humman=2, zjumocap=1)
        elif 'subset_synth' in FLAGS.custom:
            ds_parts_param = dict(
                agora=10, bedlam=16, spec=3, surreal=8,
                hi4d_rerender=8, synbody_humannerf=6, thuman2=6, dfaust_render=8)
        else:
            ds_parts_param = dict(
                agora=10, bedlam=16, rich=5, behave=5, spec=3, surreal=8, moyo=8,
                arctic=3, intercap=3, genebody=1, egobody=1, hi4d_down=3, hi4d_rerender=8,
                humman=2, synbody_humannerf=6, thuman2=6, zjumocap=1, dfaust_render=8)


        if 'no_egobody' in FLAGS.custom:
            ds_parts_param['egobody'] = 0

        if 'subset_real' in FLAGS.custom:
            ds_parts3d = {
                'h36m_': 10,
                'muco_downscaled': 6, 'humbi': 5, '3doh_down': 3,
                'panoptic_': 7, 'aist_': 6, 'aspset_': 4, 'gpa_': 4,
                'bml_movi': 5, 'mads_down': 2, 'umpm_down': 2,
                'bmhad_down': 3, '3dhp_full_down': 3, 'totalcapture': 3,
                'ikea_down': 2,
                'human4d': 1, 'fit3d_': 2, 'chi3d_': 1, 'humansc3d_': 1,
                'egohumans': 6, 'dna_rendering': 6,
            }
        elif 'subset_synth' in FLAGS.custom:
            ds_parts3d = {'3dpeople': 4, 'sailvos': 5, 'jta_down': 3, 'hspace_': 3}
        else:
            ds_parts3d = {
                'h36m_': 10,
                'muco_downscaled': 6, 'humbi': 5, '3doh_down': 3,
                'panoptic_': 7, 'aist_': 6, 'aspset_': 4, 'gpa_': 4, '3dpeople': 4, 'sailvos': 5,
                'bml_movi': 5, 'mads_down': 2, 'umpm_down': 2,
                'bmhad_down': 3, '3dhp_full_down': 3, 'totalcapture': 3, 'jta_down': 3,
                'ikea_down': 2,
                'human4d': 1, 'fit3d_': 2, 'chi3d_': 1, 'humansc3d_': 1, 'hspace_': 3,
                'egohumans': 6, 'dna_rendering': 6,
            }

        ds_parts2d = dict(
            mpii_down=4, coco_=4, jrdb_down=4, posetrack_down=4, aic_down=4, halpe=4)
        ds_parts_dense = dict(densepose_coco_=14, densepose_posetrack_=7)

        # PARAMETRIC
        dataset3d_param = ds3d.Pose3DDatasetBarecat(FLAGS.dataset3d, FLAGS.image_barecat_path)
        examples3d_param = get_examples(dataset3d_param, TRAIN)
        example_sections_param, roundrobin_sizes_param = get_sections_and_sizes(
            examples3d_param, ds_parts_param)
        if FLAGS.dataset3d_kp is None:
            FLAGS.dataset3d_kp = f'{DATA_ROOT}/posepile_28ds/annotations_28ds.barecat'

        dataset3d_kp = ds3d.Pose3DDatasetBarecat(FLAGS.dataset3d_kp, FLAGS.image_barecat_path)
        huge8_2_joint_info = spu.load_pickle(f'{PROJDIR}/huge8_2_joint_info.pkl')
        stream_parametric = self.build_roundrobin_stream(
            example_sections=example_sections_param, load_fn=load_parametric,
            extra_args=(
                FLAGS.num_surface_points, FLAGS.num_internal_points, TRAIN, dataset3d_kp.joint_info),
            batch_size=FLAGS.batch_size_parametric, roundrobin_sizes=roundrobin_sizes_param)

        # 3D KEYPOINTS
        bad_paths = spu.load_pickle(f'{DATA_ROOT}/posepile_28ds/bad_annos_28ds.pkl')
        bad_impaths = [
            'dna_rendering_downscaled/1_0018_05/02/000090.jpg',
            'dna_rendering_downscaled/1_0018_05/02/000079.jpg']
        bad_impaths = set(bad_impaths)
        examples3d = [
            ex for ex in dataset3d_kp.examples[0]
            if ex.path not in bad_paths and ex.image_path not in bad_impaths]
        example_sections3d, roundrobin_sizes3d = get_sections_and_sizes(examples3d, ds_parts3d)
        stream_kp = self.build_roundrobin_stream(
            example_sections=example_sections3d, load_fn=load_kp,
            extra_args=(dataset3d_kp.joint_info, TRAIN),
            batch_size=FLAGS.batch_size, roundrobin_sizes=roundrobin_sizes3d)

        # 2D KEYPOINTS
        dataset2d = ds2d.Pose2DDatasetBarecat(FLAGS.dataset2d, FLAGS.image_barecat_path)
        examples2d = [*dataset2d.examples[TRAIN], *dataset2d.examples[VALID]]
        example_sections2d, roundrobin_sizes2d = get_sections_and_sizes(examples2d, ds_parts2d)
        stream_2d = self.build_roundrobin_stream(
            example_sections=example_sections2d, load_fn=load_2d,
            extra_args=(dataset2d.joint_info, huge8_2_joint_info, TRAIN),
            batch_size=FLAGS.batch_size_2d, roundrobin_sizes=roundrobin_sizes2d)

        # DENSEPOSE
        dataset_dense = ds2d.Pose2DDatasetBarecat(FLAGS.dataset_dense, FLAGS.image_barecat_path)
        example_sections_dense, roundrobin_sizes_dense = get_sections_and_sizes(
            dataset_dense.examples[TRAIN], ds_parts_dense)
        stream_dense = self.build_roundrobin_stream(
            example_sections=example_sections_dense,
            load_fn=load_dense,
            extra_args=(dataset_dense.joint_info, huge8_2_joint_info, TRAIN),
            batch_size=FLAGS.batch_size_densepose, roundrobin_sizes=[14, 7])

        # COMBINE
        data_train = self.merge_streams_to_tf_dataset_train(
            streams=[stream_parametric, stream_kp, stream_dense, stream_2d],
            batch_sizes=[
                FLAGS.batch_size_parametric, FLAGS.batch_size, FLAGS.batch_size_densepose,
                FLAGS.batch_size_2d])

        # VALIDATION
        if FLAGS.validate_period:
            examples3d_val = get_examples(dataset3d_param, VALID)
            validation_steps = len(examples3d_val) // FLAGS.batch_size_test
            examples3d_val = examples3d_val[:validation_steps * FLAGS.batch_size_test]
            stream_val = self.build_stream(
                examples=examples3d_val, load_fn=load_parametric,
                extra_args=(
                    FLAGS.num_surface_points, FLAGS.num_internal_points, VALID, dataset3d_kp.joint_info),
                shuffle_before_each_epoch=False)
            data_val = self.stream_to_tf_dataset_test(stream_val, FLAGS.batch_size_test)
        else:
            validation_steps = 0
            data_val = None

        return data_train, data_val, validation_steps

    def build_model(self):
        backbone, normalizer = backbone_builder.build_backbone()
        weight_field = lf_field.build_field()
        import nlf.tf.model.nlf_model as nlf_model
        model = nlf_model.NLFModel(backbone, weight_field, normalizer)
        inp = tf.keras.Input(shape=(None, None, 3), dtype=tfu.get_dtype())
        intr = tf.keras.Input(shape=(3, 3), dtype=tf.float32)
        canonical_loc = tf.keras.Input(shape=(None, 3), dtype=tf.float32, ragged=True)
        model((inp, intr, canonical_loc), training=False)
        if not self.get_load_path() and FLAGS.load_backbone_from:
            logger.info(f'Loading backbone from {FLAGS.load_backbone_from}')
            loaded_model = tf.keras.models.load_model(FLAGS.load_backbone_from, compile=False)
            backbone_weights = [
                w for w in model.backbone.weights if
                not any(w.name.endswith(x) for x in ['rmin:0', 'rmax:0', 'dmax:0'])]
            for w1, w2 in zip(backbone_weights, loaded_model.weights):
                w1.assign(w2)

        return model

    def build_trainer(self, model):
        return lf_trainer.NLFTrainer(model)

    def build_optimizer(self, default_kwargs):
        if FLAGS.dual_finetune_lr:
            return self.build_optimizer_dual(default_kwargs)

        weight_decay = FLAGS.weight_decay / np.sqrt(self.training_steps) / FLAGS.base_learning_rate
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.wrap_learning_rate(self.learning_rate_schedule),
            weight_decay=weight_decay, epsilon=1e-8, **default_kwargs)

        excluded = [self.model.canonical_lefts, self.model.canonical_centers]
        if not FLAGS.decay_field:
            excluded += self.model.heatmap_head.weight_field.trainable_variables
        optimizer.exclude_from_weight_decay(excluded)
        return optimizer

    def build_optimizer_dual(self, default_kwargs):
        weight_decay = FLAGS.weight_decay / np.sqrt(self.training_steps) / FLAGS.base_learning_rate
        optimizer_backbone = tf.keras.optimizers.Adam(
            learning_rate=self.wrap_learning_rate(self.learning_rate_schedule_finetune_low),
            weight_decay=weight_decay, epsilon=1e-8, **default_kwargs)
        optimizer_head = tf.keras.optimizers.Adam(
            learning_rate=self.wrap_learning_rate(self.learning_rate_schedule),
            weight_decay=weight_decay, epsilon=1e-8, **default_kwargs)

        excluded = [self.model.canonical_lefts, self.model.canonical_centers]
        if not FLAGS.decay_field:
            excluded += self.model.heatmap_head.weight_field.trainable_variables

        optimizer_head.exclude_from_weight_decay(excluded)
        optimizer = fleras.optimizers.MultiOptimizer(
            [(optimizer_backbone, self.model.backbone),
             (optimizer_head,
              [self.model.heatmap_head, self.model.canonical_lefts, self.model.canonical_centers])])
        return optimizer

    def learning_rate_schedule(self, step):
        n_warmup_steps = 1000
        n_phase1_steps = (1 - FLAGS.lr_cooldown_fraction) * self.training_steps - n_warmup_steps
        n_phase2_steps = self.training_steps - n_warmup_steps - n_phase1_steps
        step_float = tf.cast(step, tf.float32)
        b = tf.constant(FLAGS.base_learning_rate, tf.float32)

        if step_float < n_warmup_steps:
            return b * FLAGS.field_lr_factor * step_float / n_warmup_steps
        elif step_float < n_warmup_steps + n_phase1_steps:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                b * FLAGS.field_lr_factor, decay_rate=1 / 3, decay_steps=n_phase1_steps,
                staircase=False)(step_float)
        else:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                b * tf.cast(1 / 30, tf.float32), decay_rate=0.3, decay_steps=n_phase2_steps,
                staircase=False)(step_float - n_warmup_steps - n_phase1_steps)

    def learning_rate_schedule_finetune_low(self, step):
        step_float = tf.cast(step, tf.float32)
        n_frozen_steps = 3000
        n_warmup_steps = 2000

        n_phase1_steps = ((
                                  1 - FLAGS.lr_cooldown_fraction) * self.training_steps -
                          n_frozen_steps - n_warmup_steps)
        n_phase2_steps = self.training_steps - n_frozen_steps - n_warmup_steps - n_phase1_steps

        b = tf.constant(FLAGS.base_learning_rate, tf.float32)

        if step_float < n_frozen_steps:
            return tf.convert_to_tensor(0, dtype=tf.float32)
        elif step_float < n_frozen_steps + n_warmup_steps:
            return b * FLAGS.backbone_lr_factor * (step_float - n_frozen_steps) / n_warmup_steps
        elif step_float < n_frozen_steps + n_warmup_steps + n_phase1_steps:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                b * FLAGS.backbone_lr_factor, decay_rate=1 / 3, decay_steps=n_phase1_steps,
                staircase=False)(step_float - n_warmup_steps - n_frozen_steps)
        else:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                b * tf.cast(1 / 30, tf.float32), decay_rate=0.3, decay_steps=n_phase2_steps,
                staircase=False)(step_float - n_warmup_steps - n_frozen_steps - n_phase1_steps)

    def build_callbacks(self):
        cbacks = [
            render_callback.RenderPredictionCallback(start_step=500, interval=100)]
        if FLAGS.batch_renorm:
            cbacks.append(AdjustRenormClipping(7500, 10000))

        return cbacks


class AdjustRenormClipping(tf.keras.callbacks.Callback):
    def __init__(self, ramp_start_step, ramp_length):
        super().__init__()
        self.ramp_start_step = ramp_start_step
        self.ramp_length = ramp_length

    def on_train_batch_begin(self, batch, logs=None):
        batch = tf.convert_to_tensor(batch // FLAGS.grad_accum_steps, dtype=tf.float32)
        renorm_clipping = self.model.model.normalizer.renorm_clipping
        ramp = tfu.ramp_function(batch, self.ramp_start_step, self.ramp_length)
        rmax = 1 + ramp * 2 * FLAGS.renorm_limit_scale  # ramps from 1 to 3
        dmax = ramp * 5 * FLAGS.renorm_limit_scale  # ramps from 0 to 5
        renorm_clipping['rmax'].assign(rmax)
        renorm_clipping['rmin'].assign(tf.math.reciprocal(rmax))
        renorm_clipping['dmax'].assign(dmax)


def get_examples(dataset, learning_phase):
    if learning_phase == TRAIN:
        str_example_phase = FLAGS.train_on
    elif learning_phase == VALID:
        str_example_phase = FLAGS.validate_on
    elif learning_phase == TEST:
        str_example_phase = FLAGS.test_on
    else:
        raise Exception(f'No such learning_phase as {learning_phase}')

    if str_example_phase == 'train':
        examples = dataset.examples[TRAIN]
    elif str_example_phase == 'valid':
        examples = dataset.examples[VALID]
    elif str_example_phase == 'test':
        examples = dataset.examples[TEST]
    elif str_example_phase == 'trainval':
        examples = [*dataset.examples[TRAIN], *dataset.examples[VALID]]
    else:
        raise Exception(f'No such phase as {str_example_phase}')
    return examples


def get_sections_and_sizes(examples, section_name_to_size, verify_all_included=False):
    section_names, section_sizes = zip(*section_name_to_size.items())

    sections = [[] for _ in section_names]
    for ex in spu.progressbar(examples, desc='Building dataset sections'):
        for i, name in enumerate(section_names):
            if ex.image_path.startswith(name):
                sections[i].append(ex)
                break
        else:
            if verify_all_included:
                raise RuntimeError(f'No section for {ex.image_path}')

    if not all(len(s) > 0 for s in sections):
        for name, s in zip(section_names, sections):
            print(f'{name}: {len(s)}')
        raise RuntimeError('Some sections are empty')
    return sections, section_sizes


if __name__ == '__main__':
    main()
