import os
import time
import glob
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from . import data_utils

tf.compat.v1.enable_eager_execution()
sns.set()


def show_spec(spec, **kwargs):
    sns.heatmap(spec, **kwargs).invert_yaxis()
    plt.show(block=False)


# INPUT
TRAIN_DATA_ROOT = "../../data/audio/pendulum/train"
TEST_DATA_ROOT = "../../data/audio/pendulum/test"

# HYPERPARAMETERS
WINDOW_SIZE = 1
BATCH_SIZE = 1

# OUTPUT
EXAMPLES_TO_GENERATE = 16
OUTPUT_DIR = "../../data/output/audio"

train_dataset = data_utils.load_dataset(TRAIN_DATA_ROOT)
test_dataset = data_utils.load_dataset(TEST_DATA_ROOT)


class ConvolutionalVariationalAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim, num_conv=2, input_shape=(512, 512, WINDOW_SIZE)):
        super().__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu',
                ),
                *[tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'
                ) for _ in range(num_conv)],
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2 * latent_dim)
            ], name="Encoder"
        )

        decoder_input_shape = (
            input_shape[0] // (2 ** (num_conv + 1)),
            input_shape[1] // (2 ** (num_conv + 1)),
            64
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(
                    decoder_input_shape[0] * decoder_input_shape[1] * decoder_input_shape[2],
                    activation=tf.nn.relu
                ),
                tf.keras.layers.Reshape(target_shape=decoder_input_shape),
                *[tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='SAME',
                    activation='relu'
                ) for _ in range(num_conv)],
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='SAME',
                    activation='relu'
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding='SAME'
                )
            ], name="Decoder"
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        inference = self.inference_net(x)
        mean, logvar = tf.split(inference, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


for e in train_dataset:
    print(e.shape)
