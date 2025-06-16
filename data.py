import os, string, tensorflow as tf
from omegaconf import DictConfig

def build_lookup_tables(char_set: str, blank_char: str):
    chars = list(char_set + blank_char)
    ids   = tf.range(len(chars), dtype=tf.int32)

    to_id   = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(chars, ids),
        default_value=tf.constant(-1, tf.int32)
    )
    to_char = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(ids, chars),
        default_value=tf.constant("?", tf.string)
    )
    return to_id, to_char, len(chars)

_CHAR_TO_ID, _ID_TO_CHAR, N_VOCAB = build_lookup_tables(
    char_set=os.getenv("CHAR_SET", string.ascii_uppercase + string.digits),
    blank_char=os.getenv("BLANK_CHAR", "_")
)

def _load_image(path: tf.Tensor, cfg: DictConfig) -> tf.Tensor:
    """Decode PNG → grayscale → [0,1] float32"""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)            
    img = tf.image.rgb_to_grayscale(img)                
    img = tf.image.convert_image_dtype(img, tf.float32)   
    img = tf.image.resize(img, (cfg.img_height, cfg.img_width))
    return img

def _parse_path(path: tf.Tensor, cfg: DictConfig):
    filename = tf.strings.split(path, os.sep)[-1]
    label_str = tf.strings.split(filename, '_')[0]       
    chars = tf.strings.bytes_split(label_str)

    ids = _CHAR_TO_ID.lookup(chars)                      
    ids = tf.where(ids < 0, 0, ids)                      

    label_len = tf.shape(ids)[0]
    pad = tf.math.maximum(cfg.max_label_len - label_len, 0)
    ids = tf.pad(ids, paddings=[[0, pad]], constant_values=-1) 
    return ids, label_len

def _load_example(path: tf.Tensor, cfg: DictConfig):
    img = _load_image(path, cfg)
    ids, label_len = _parse_path(path, cfg)
    return img, ids

def get_dataset(cfg: DictConfig):
    all_files = tf.io.gfile.glob(os.path.join(cfg.data_dir, "*.png"))
    ds = tf.data.Dataset.from_tensor_slices(sorted(all_files))
    ds = ds.shuffle(len(all_files), seed=cfg.seed, reshuffle_each_iteration=False)

    n_total   = len(all_files)
    n_val     = int(cfg.val_split  * n_total)
    n_test    = int(cfg.test_split * n_total)
    n_train   = n_total - n_val - n_test

    splits = {
        "train": ds.take(n_train),
        "val"  : ds.skip(n_train).take(n_val),
        "test" : ds.skip(n_train + n_val)
    }

    def pip(ds, training=False):
        if training:
            ds = ds.shuffle(8192, seed=cfg.seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda p: _load_example(p, cfg),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(cfg.train.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return pip(splits["train"], True), pip(splits["val"]), pip(splits["test"])

def ids_to_text(ids):
    chars = _ID_TO_CHAR.lookup(ids)
    return tf.strings.reduce_join(chars, axis=-1)
