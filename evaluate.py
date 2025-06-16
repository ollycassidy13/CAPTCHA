import os, numpy as np, tensorflow as tf, tqdm
from omegaconf import OmegaConf
from data import get_dataset, ids_to_text, N_VOCAB

def dense_ctc_decode(logits, blank_index):
    if logits.dtype == tf.float16:
        logits = tf.cast(logits, tf.float32)
    
    T = logits.shape[1]
    batch_sz = tf.shape(logits)[0]

    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, [1, 0, 2]), 
        sequence_length=tf.fill([batch_sz], T),
        blank_index=blank_index
    )
    dense = tf.sparse.to_dense(decoded[0], default_value=-1)
    return tf.cast(dense, tf.int32)

def evaluate(cfg_path="config.yaml", weights_path="model/best.h5"):
    cfg = OmegaConf.load(cfg_path)
    _, _, ds_test = get_dataset(cfg)

    model = tf.keras.models.load_model(weights_path, compile=False)
    
    BLANK_INDEX = N_VOCAB - 1
    
    char_corr = char_tot = word_corr = word_tot = 0
    first_batch = True
    
    for images, true_ids in tqdm.tqdm(ds_test, desc="Evaluating"):
        logits = model(images, training=False)
        pred_ids = dense_ctc_decode(logits, BLANK_INDEX) 

        for i in range(tf.shape(images)[0]):
            true_seq = true_ids[i][true_ids[i] != -1]
            pred_seq = pred_ids[i][pred_ids[i] != -1]
            
            min_len = tf.minimum(tf.shape(true_seq)[0], tf.shape(pred_seq)[0])
            if min_len > 0:
                char_matches = tf.reduce_sum(tf.cast(
                    true_seq[:min_len] == pred_seq[:min_len], tf.int32))
                char_corr += char_matches
                char_tot += tf.shape(true_seq)[0] 

        max_len = tf.shape(true_ids)[1]
        pred_padded = tf.pad(pred_ids, 
                           [[0, 0], [0, max_len - tf.shape(pred_ids)[1]]], 
                           constant_values=-1)
        pred_padded = pred_padded[:, :max_len] 
        
        seq_matches = tf.reduce_all(pred_padded == true_ids, axis=1)
        word_corr += tf.reduce_sum(tf.cast(seq_matches, tf.int32))
        word_tot += tf.shape(images)[0]

        if first_batch:
            first_batch = False
            print("\nFirst 5 examples (after proper CTC decoding):")
            for i in range(min(5, tf.shape(images)[0])):
                true_seq = true_ids[i][true_ids[i] != -1]
                pred_seq = pred_ids[i][pred_ids[i] != -1]
                
                true_text = ids_to_text(tf.expand_dims(true_seq, 0))[0].numpy().decode('utf-8')
                pred_text = ids_to_text(tf.expand_dims(pred_seq, 0))[0].numpy().decode('utf-8')
                
                if hasattr(cfg, 'blank_char'):
                    true_text = true_text.replace(cfg.blank_char, "")
                    pred_text = pred_text.replace(cfg.blank_char, "")
                
                match = tf.reduce_all(true_seq == pred_seq).numpy()
                print(f"Sample {i+1}:")
                print(f"  Predicted: {pred_seq.numpy()} -> '{pred_text}'")
                print(f"  Actual:    {true_seq.numpy()} -> '{true_text}'")
                print(f"  Match:     {match}")

    char_accuracy = char_corr / char_tot if char_tot > 0 else 0
    word_accuracy = word_corr / word_tot if word_tot > 0 else 0
    
    print(f"\nCharacter-level accuracy: {char_accuracy:.4%}")
    print(f"Sequence-level accuracy:  {word_accuracy:.4%}")
    print(f"Total sequences evaluated: {word_tot}")

if __name__ == "__main__":
    evaluate()
