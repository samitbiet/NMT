from translator import infer
from translator import utils
from translator import models
from translator import config as cfg

import tensorflow as tf

# load input and terget language tokenizer
input_language_tokenizer = utils.load_tokenizer('/content/NMT/model/input_language_tokenizer.json')
target_language_tokenizer = utils.load_tokenizer('/content/NMT/model/target_language_tokenizer.json')

# init vocab size for input and terget language
vocab_inp_size = len(input_language_tokenizer.word_index)+1
vocab_tar_size = len(target_language_tokenizer.word_index)+1

# init encoder & decoder model
encoder = models.Encoder(vocab_inp_size, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)
decoder = models.Decoder(vocab_tar_size, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)

# restore model from checkpoints
checkpoint_dir = 'models/training_checkpoints'
checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


predictor = infer.Infer(
    input_language_tokenizer=input_language_tokenizer,
    target_language_tokenizer=target_language_tokenizer,
    max_length_input=cfg.MAX_INPUT_LANG_LEN,
    max_length_target=cfg.MAX_TARGET_LANG_LEN,
    encoder=encoder,
    decoder=decoder,
    units=cfg.UNITS
)


# Predict
translated_text = predictor.predict("ঘুম থেকে ওঠ")
print("Translated text: {}".format(translated_text))



print ('Tested ......... Samit')
