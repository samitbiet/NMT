from translator import utils
from translator import train
from translator import models
from translator import datasets
from translator import config as cfg

import os
import time
import tensorflow as tf
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

# Retrieve Data and Tokenizers
#------------------------------------------
path_to_dataset = '/content/NMT/data/ben.txt'
text_data = datasets.TatoebaDataset(path_to_dataset, cfg.NUM_DATA_TO_LOAD)

# retrive data and tokenizers
tensors, tokenizer =  text_data.load_data()
input_tensor, target_tensor = tensors 
inp_lang_tokenizer, targ_lang_tokenizer = tokenizer

# save tokenizer for further use
utils.save_tokenizer(
    tokenizer=inp_lang_tokenizer,
    save_at='/content/NMT/model',
    file_name='input_language_tokenizer.json')
utils.save_tokenizer(
    tokenizer=targ_lang_tokenizer,
    save_at='/content/NMT/model',
    file_name='target_language_tokenizer.json')

# Creating training and validation sets using an 80-20 split
#----------------------------------------------------------------
input_train, input_val, target_train, target_val = \
    train_test_split(input_tensor, target_tensor, test_size=0.2)

# set training params
buffer_size = len(input_train)
steps_per_epoch = len(input_train) // cfg.BATCH_SIZE
vocab_inp_size = len(inp_lang_tokenizer.word_index) + 1
vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1

# convert data to tf.data formate
dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train))
dataset = dataset.shuffle(buffer_size)
dataset = dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)


# init encoder & decoder
encoder = models.Encoder(
    vocab_inp_size, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)
decoder = models.Decoder(
    vocab_tar_size, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)


# init optimizer
optimizer = tf.keras.optimizers.Adam()

#logdir = os.path.join("/content/NMT/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# init checkpoint 
checkpoint_dir = '/content/NMT/models/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                encoder=encoder,
                                decoder=decoder)

if cfg.RESTORE_SAVED_CHECKPOINT:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


for epoch in range(cfg.EPOCHS):
    print("Epoch {} / {}".format(epoch, cfg.EPOCHS))
    pbar = tqdm(dataset.take(steps_per_epoch), ascii=True, total=steps_per_epoch)

    total_loss = 0
    enc_hidden = encoder.initialize_hidden_state()

    for step, data in enumerate(pbar):
        inp, targ = data
        
        batch_loss = train.train_step(
            inp, targ, targ_lang_tokenizer,
            enc_hidden, encoder, decoder, optimizer
        )

        total_loss += batch_loss

        pbar.set_description(
            "Step - {} / {} - batch loss - {:.4f} "
                .format(step+1, steps_per_epoch, batch_loss.numpy()))

    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    print('Epoch loss - {:.4f}'.format(total_loss / steps_per_epoch))



print ('Imported.......Samit')
