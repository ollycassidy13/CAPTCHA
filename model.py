import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from omegaconf import DictConfig

def build_crnn_ctc(cfg: DictConfig, n_vocab: int) -> tf.keras.Model:
    H, W = cfg.img_height, cfg.img_width
    img   = layers.Input(shape=(H, W, 1), name="img")     

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(img)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)           

    for filters in (128, 256):
        residual = layers.Conv2D(filters, 1, padding="same",
                                 use_bias=False)(x)
        x = layers.Conv2D(filters, 3, padding="same",
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same",
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)            

    x = layers.Permute((2, 1, 3))(x)                      
    T  = x.shape[1]                                       
    x = layers.Reshape((T, -1))(x)                        
    x = layers.Bidirectional(layers.LSTM(256,
                     return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(256,
                     return_sequences=True, dropout=0.25))(x)

    logits = layers.Dense(n_vocab, activation="linear", name="logits")(x)
    return models.Model(img, logits, name="crnn_ctc")

def enable_mixed_precision(cfg: DictConfig):
    if cfg.train.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
