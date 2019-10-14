import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from operator import mul
from functools import reduce
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image

from . import data_utils
from .. import settings

# region Dataset Metadata
window_size = settings.WINDOW_SIZE
image_dims = [settings.CHUNK_SIZE, settings.N_MELS]
input_shape = [*image_dims, window_size]
# endregion

# region Model Hyperparameters
hidden_size = 32
latent_dims = settings.LATENT_DIMS
num_epochs = settings.EPOCHS
batch_size = settings.BATCH_SIZE


# endregion

# region Model Definition
@tf.function
def sampling(args):
    zm, zlv = args
    epsilon = K.random_normal(
        shape=(K.shape(zm)[0], K.int_shape(zm)[1]),
        mean=0.,
        stddev=1.
    )
    return zm + zlv * epsilon


def get_vae_loss_fn(zm, zlv):
    @tf.function
    def _loss_fn(y_true, y_pred):
        xent_loss = tf.keras.losses.mse(y_true, y_pred)
        kl_loss = -0.5 * K.mean(1 + zlv - K.square(zm) - K.exp(zlv))
        return xent_loss + kl_loss

    return _loss_fn


def get_model():
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    x = tf.keras.layers.Lambda(lambda e: K.squeeze(e, -1), name='squeeze')(x)
    x = tf.keras.layers.LSTM(hidden_size)(x)
    z_mean = tf.keras.layers.Dense(latent_dims, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dims, name='z_log_var')(x)
    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dims,), name='latent')([z_mean, z_log_var])
    enc = tf.keras.Model(inputs, z_mean, name='encoder')

    decoder_h = tf.keras.layers.LSTM(hidden_size, return_sequences=True, name='hidden_decoder')
    decoder_mean = tf.keras.layers.LSTM(input_shape[1], return_sequences=True, name='mean_decoder')

    decoder_input = tf.keras.layers.Input(shape=(latent_dims,))
    _h_decoded = tf.keras.layers.RepeatVector(input_shape[1])(decoder_input)
    _h_decoded = decoder_h(_h_decoded)
    _h_decoded = decoder_mean(_h_decoded)
    dec = tf.keras.Model(decoder_input, _h_decoded, name='decoder')

    h_decoded = tf.keras.layers.RepeatVector(input_shape[0])(z)
    h_decoded = decoder_h(h_decoded)
    h_decoded = decoder_mean(h_decoded)
    vae = tf.keras.Model(inputs, h_decoded, name='autoencoder')

    enc.summary()
    dec.summary()
    vae.summary()

    tf.keras.utils.plot_model(vae, show_shapes=True, to_file='autoencoder.png')

    vae.compile(
        # optimizer=tf.keras.optimizers.Adam(1e-6),
        loss=get_vae_loss_fn(z_mean, z_log_var),
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.1, patience=3, min_lr=1e-9, verbose=1
        )])
    return enc, dec, vae


# endregion

# train_dataset, test_dataset = data_utils.load_numpy_dataset(settings.TRAIN_DATA_DIR, return_tuples=True)
# train_dataset, _ = data_utils.image_to_number_dataset(image_dims, num_samples=500, batch_size=5)
train_dataset, _ = data_utils.image_to_number_dataset(image_dims, num_samples=100, batch_size=1, reconstruction=True)

encoder, decoder, autoencoder = get_model()
autoencoder.fit(train_dataset, epochs=20)
test = np.ones((1, *image_dims, 1))
print(test[0, 0, 0])
encoded = encoder.predict(test)
print(encoded)
print(decoder.predict(encoded))

# start = time.time()
# num_samples = 100
# with tqdm(train_dataset.take(num_samples), total=num_samples) as pbar:
#     for i, element in enumerate(pbar):
#         print(element)
#         # pbar.write(f"{i + 1}: {element[0].shape}")
#         pass
# print("----------------FINISHED----------------")
# print(time.time() - start)
