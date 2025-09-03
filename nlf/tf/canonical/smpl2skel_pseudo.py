# This code generates the smpl2huge8.npy file, which is a joint
# regressor that maps from SMPL vertices to the 555 joints of the
# PosePile dataset as of the MeTRAbs-ACAE paper. The regressor is
# learned from pseudo-GT predictions made by MeTRAbs-ACAE on the
# SURREAL dataset. The regressor is learned using an affine-combining
# regressor, which is like the ACAE, but it's just one layer that
# combines input 3D points to output 3D points in an affine manner with
# regularization. The regularization penalizes a large support for output
# points, that is, the incoming weights should be spatially compact as
# measured on the canonical template. That is, a joint should only depend
# on vertices that are near each other.

import fleras
import h5py
import more_itertools
import numpy as np
import simplepyutils as spu
import tensorflow as tf
from affine_combining_autoencoder import AffineCombiningRegressor, AffineCombiningRegressorTrainer
from posepile.ds.surreal import main as surreal

from nlf.paths import DATA_ROOT, PROJDIR
from nlf.tf import tfu, tfu3d


def main():
    ji_mesh = surreal.get_smpl_mesh_joint_info().select_joints(list(range(24, 6890 + 24)))
    ji = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/huge8_joint_info.pkl')
    perm_mesh = get_permutation(ji_mesh)
    perm_huge8 = get_permutation(ji)
    invperm_huge8 = list(tf.math.invert_permutation(perm_huge8).numpy())
    invperm_mesh = list(tf.math.invert_permutation(perm_mesh).numpy())

    # The HDF5 file contain predictions made by MeTRAbs-ACAE on the SURREAL dataset.
    h5path = 'pred_surreal.h5'
    batch_size = 64
    ds = hdf5_as_tf_dataset(
        h5path,
        ['coords3d_true_cam', 'coords3d_pred_cam', 'image_path'])
    ds = ds.filter(was_pose_estimation_successful)

    def transform(inp):
        return dict(
            pose3d_in=tf.gather(inp['coords3d_true_cam'][24:], perm_mesh, axis=0),
            pose3d_out=tf.gather(inp['coords3d_pred_cam'], perm_huge8, axis=0))

    train_ds = (
        ds.
        filter(
            lambda x: tf.strings.regex_full_match(x['image_path'], b'.+surreal/train.+')).
        map(transform).
        cache(f'{DATA_ROOT}/cache/surreal_for_regressor_finetuned'))
    val_ds = (
        ds.
        filter(lambda x: tf.strings.regex_full_match(x['image_path'], b'.+surreal/val.+')).
        map(transform).
        cache())

    n_train = more_itertools.ilen(train_ds)
    steps_per_epoch = n_train // batch_size
    n_val = more_itertools.ilen(val_ds)

    train_ds = train_ds.shuffle(10000).repeat().batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    model = AffineCombiningRegressor(3353 * 2, 184, 222 * 2, 111, chiral=True)

    canonical_verts = np.load(f'{PROJDIR}/canonical_vertices_smpl.npy')
    trainer = AffineCombiningRegressorTrainer(
        model, regul_lambda=0, supp_lambda=3e-1, random_seed=0,
        pose3d_template=canonical_verts[perm_mesh].copy() * 1000)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule()))

    hist = trainer.fit_epochless(
        train_ds, validation_data=val_ds, validation_steps=n_val // batch_size,
        validation_freq=150, steps=steps_per_epoch, verbose=1,
        callbacks=[fleras.callbacks.ProgbarLogger(), tf.keras.callbacks.History()])

    trainer.evaluate(val_ds, steps=n_val // batch_size)
    model.layer.threshold_for_sparsity(1e-3)
    trainer.evaluate(val_ds, steps=n_val // batch_size)

    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=30))
    hist2 = trainer.fit_epochless(
        train_ds, validation_data=val_ds, validation_steps=n_val // batch_size,
        validation_freq=150, steps=steps_per_epoch, verbose=1,
        callbacks=[fleras.callbacks.ProgbarLogger(), tf.keras.callbacks.History()])

    trainer.evaluate(val_ds, steps=n_val // batch_size)
    model.layer.threshold_for_sparsity(1e-3)
    trainer.evaluate(val_ds, steps=n_val // batch_size)

    hist3 = trainer.fit_epochless(
        train_ds, validation_data=val_ds, validation_steps=n_val // batch_size,
        validation_freq=150, steps=steps_per_epoch, verbose=1,
        callbacks=[fleras.callbacks.ProgbarLogger(), tf.keras.callbacks.History()])

    w = model.layer.get_w().numpy()
    w_backperm = w[invperm_mesh][:, invperm_huge8]
    np.save(f'{PROJDIR}/reprod_smpl2huge8.npy', w_backperm)


@fleras.optimizers.schedules.wrap(jit_compile=True)
def lr_schedule(step):
    n_total_steps = 37500
    if step < int(n_total_steps * 0.9):
        return 1e0
    else:
        return 1e-3


def was_pose_estimation_successful(ex):
    pose_smpl_true = ex['coords3d_true_cam'][:24]
    pose_smpl_pred = ex['coords3d_pred_cam'][:24]
    pose_smpl_pred_aligned = tfu3d.rigid_align(
        pose_smpl_pred[tf.newaxis], pose_smpl_true[tf.newaxis], scale_align=True)
    dist = tf.linalg.norm(pose_smpl_true - pose_smpl_pred_aligned, axis=-1)
    return tf.reduce_mean(tfu.auc(dist, 0, 150)) > 0.5


def hdf5_as_tf_dataset(filepath, hdf5_dataset_names=None):
    def generator():
        with h5py.File(filepath, 'r') as f:
            for arrays in zip(*[f[name] for name in hdf5_dataset_names]):
                yield dict(zip(hdf5_dataset_names, arrays))

    with h5py.File(filepath, 'r') as f:
        if hdf5_dataset_names is None:
            hdf5_dataset_names = list(f.keys())

        output_signature = {
            name: tf.TensorSpec(shape=f[name].shape[1:], dtype=f[name].dtype)
            for name in hdf5_dataset_names}

    return tf.data.Dataset.from_generator(
        generator, output_signature=output_signature)


def get_permutation(ji):
    left_ids = [i for i, name in enumerate(ji.names) if name[0] == 'l']
    right_ids = [ji.ids['r' + name[1:]] for i, name in enumerate(ji.names) if name[0] == 'l']
    center_ids = [i for i, name in enumerate(ji.names) if name[0] not in 'lr']
    permutation = left_ids + right_ids + center_ids
    return permutation
