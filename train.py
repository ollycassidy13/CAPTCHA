import os, random, numpy as np, tensorflow as tf, tensorflow_addons as tfa
from omegaconf import OmegaConf
from hydra import main, initialize, compose
from data import get_dataset, N_VOCAB
from model import build_crnn_ctc, enable_mixed_precision

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class CTCLoss(tf.keras.losses.Loss):
    """Wraps tf.nn.ctc_loss; expects padded label tensor filled with -1."""
    def __init__(self, blank_index):
        super().__init__(reduction="sum_over_batch_size", name="ctc_loss")
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        label_len = tf.math.count_nonzero(y_true + 1, axis=1, dtype=tf.int32)
        logit_len = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        sparse = tf.keras.backend.ctc_label_dense_to_sparse(
                    tf.cast(y_true, tf.int32), label_len)
        loss = tf.nn.ctc_loss(sparse,
                              y_pred,
                              label_len,
                              logit_len,
                              blank_index=self.blank_index,
                              logits_time_major=False)
        return loss

@main(version_base=None, config_path=".", config_name="config")
def train(cfg):
    set_seeds(cfg.seed)
    enable_mixed_precision(cfg)

    ds_train, ds_val, _ = get_dataset(cfg)

    model = build_crnn_ctc(cfg, N_VOCAB)
    loss_fn = CTCLoss(blank_index=N_VOCAB - 1)
    optimizer = tfa.optimizers.AdamW(
        weight_decay=cfg.train.weight_decay, 
        learning_rate=cfg.train.learning_rate)

    model.compile(optimizer=optimizer, loss=loss_fn)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=cfg.train.early_stop_patience, monitor="val_loss",
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=cfg.train.reduce_lr_patience, factor=0.3,
            monitor="val_loss", min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg.train.modeldir, "best.h5"),
            save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.TensorBoard(
            log_dir=cfg.train.logdir, write_graph=False)
    ]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.train.epochs,
        callbacks=callbacks
    )

    model.save(os.path.join(cfg.train.modeldir, "final.h5"))

if __name__ == "__main__":
    train()
