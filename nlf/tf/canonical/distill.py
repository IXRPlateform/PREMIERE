import fleras
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from fleras.util.easydict import EasyDict

from nlf.paths import PROJDIR


def main():
    out_dim = 1024
    hidden_dim = 2048
    pos_enc_dim = 512
    n_steps = 200000
    batch_size = 1024

    # DATA
    nodes = np.load(f'{PROJDIR}/canonical_nodes3.npy')
    tet_eigve = np.load(f'{PROJDIR}/canonical_eigvec3.npy')
    x = nodes
    y = tet_eigve[:, 1:1025]
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1)

    data_train = tf.data.Dataset.from_tensor_slices(
        dict(embedding_true=y_train[:, :out_dim].astype(np.float32),
             query=x_train.astype(np.float32))).shuffle(
        reshuffle_each_iteration=True, buffer_size=len(x_train)).batch(batch_size).repeat()
    data_val = tf.data.Dataset.from_tensor_slices(
        dict(embedding_true=y_val[:, :out_dim].astype(np.float32),
             query=x_val.astype(np.float32))).batch(
        batch_size).repeat()

    # MODEL
    mini = np.min(x, axis=0)
    maxi = np.max(x, axis=0)
    center = (mini + maxi) / 2
    model = LearnableFourierNet(
        center=center, mini=mini, maxi=maxi, pos_enc_dim=pos_enc_dim, hidden_dim=hidden_dim,
        out_dim=out_dim)

    # TRAINING
    trainer = Trainer(model)
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=WarmupCosineDecay(1e-2, 0, n_steps, 0)))

    hist = trainer.fit_epochless(
        data_train, steps=n_steps, verbose=1, validation_data=data_val, validation_freq=1000,
        validation_steps=len(x_val) // batch_size,
        callbacks=[fleras.callbacks.ProgbarLogger(), tf.keras.callbacks.History()])

    # SAVE
    model.save(f'{PROJDIR}/lbo_mlp_{pos_enc_dim}fourier_{hidden_dim}gelu_{out_dim}')


class LearnableFourierNet(tf.keras.Model):
    def __init__(self, center, mini, maxi, pos_enc_dim=512, hidden_dim=2048, out_dim=1024):
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.factor = 1 / np.sqrt(np.float32(self.pos_enc_dim))
        self.W_r = tf.keras.layers.Dense(
            self.pos_enc_dim // 2, use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=12))
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.dense2 = tf.keras.layers.Dense(out_dim)

        self.center = center
        self.mini = mini
        self.maxi = maxi

    def call(self, inp, training):
        x = (inp - self.center) / (self.maxi - self.mini)
        x = self.W_r(x)
        x = tf.sin(tf.concat([x, x + np.pi / 2], axis=-1)) * self.factor
        x = self.dense1(x)
        return self.dense2(x)


class Trainer(fleras.ModelTrainer):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def forward_train(self, inps, training):
        pred = self.model(inps.query, training=training)
        return EasyDict(embedding_pred=pred)

    def compute_losses(self, inps, preds):
        losses = EasyDict()
        losses.loss_embedding = tf.reduce_mean(
            tf.square(preds.embedding_pred - inps.embedding_true))
        losses.loss = losses.loss_embedding
        return losses

    def compute_metrics(self, inps, preds, training):
        return EasyDict(
            l2=tf.sqrt(
                tf.reduce_mean(tf.square(preds.embedding_pred[:, :] - inps.embedding_true[:, :]))),
            l2_10=tf.sqrt(tf.reduce_mean(
                tf.square(preds.embedding_pred[:, :10] - inps.embedding_true[:, :10]))),
            l2_100=tf.sqrt(tf.reduce_mean(
                tf.square(preds.embedding_pred[:, :100] - inps.embedding_true[:, :100]))),
            l2_over128=tf.sqrt(tf.reduce_mean(
                tf.square(preds.embedding_pred[:, 128:256] - inps.embedding_true[:, 128:256]))),
            l2_over256=tf.sqrt(tf.reduce_mean(
                tf.square(preds.embedding_pred[:, 256:512] - inps.embedding_true[:, 256:512]))),
            l2_over512=tf.sqrt(tf.reduce_mean(
                tf.square(preds.embedding_pred[:, 512:] - inps.embedding_true[:, 512:]))))


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, target_lr, warmup_steps, total_steps, hold):
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.hold = tf.cast(hold, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.target_lr = tf.cast(target_lr, tf.float32)

    @tf.function
    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        if step < self.warmup_steps:
            return self.target_lr * (step / self.warmup_steps)

        if step <= self.warmup_steps + self.hold:
            return self.target_lr

        return 0.5 * self.target_lr * (1 + tf.cos(np.pi * (step - self.warmup_steps - self.hold) / (
                self.total_steps - self.warmup_steps - self.hold)))

if __name__ == '__main__':
    main()