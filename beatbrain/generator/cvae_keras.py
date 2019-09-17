import os
import time
import numpy as np
import tensorflow as tf
from operator import mul
from functools import reduce
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image

from . import data_utils
from .. import settings

tf.compat.v1.enable_eager_execution()


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_apply_gradients(mdl, x, opt):
    with tf.GradientTape() as tape:
        loss = compute_loss(mdl, x)
    gradients = tape.gradient(loss, mdl.trainable_variables)
    opt.apply_gradients(zip(gradients, mdl.trainable_variables))


def visualize_model_outputs(mdl, epoch, test_input, output):
    output = Path(output)
    predictions = mdl.sample(eps=test_input)
    print(f"Saving Samples Images to {output}")
    for i, pred in enumerate(predictions):
        progress_dir = Path(settings.MODEL_WEIGHTS or datetime.now().strftime("%Y%m%d-%H%M%S")).resolve().stem
        out_dir = output.joinpath(progress_dir).joinpath(str(i + 1))
        os.makedirs(out_dir, exist_ok=True)
        image = Image.fromarray(pred[:, :, 0].numpy(), mode='F')
        image.save(os.path.join(out_dir, f"epoch_{epoch}.tiff"))


@tf.function
def reparameterize(args):
    mean, logvar = args
    batch = tf.keras.backend.shape(mean)[0]
    dim = tf.keras.backend.int_shape(mean)[1]
    eps = tf.random.normal(shape=(batch, dim))
    return eps * tf.keras.backend.exp(logvar * 0.5) + mean


def vae_loss(mean, logvar, img_dims):
    def loss_fn(y_pred, y_true):
        reconstruction_loss = tf.keras.losses.binary_crossentropy(y_pred, y_true)
        reconstruction_loss *= reduce(mul, img_dims)
        kl_loss = 1 + logvar - tf.keras.backend.square(mean) - tf.keras.backend.exp(logvar)
        kl_loss = -0.5 * tf.keras.backend.sum(kl_loss, axis=-1)
        return tf.keras.backend.mean(reconstruction_loss + kl_loss)

    return loss_fn


# region Model hyperparameters
window_size = settings.WINDOW_SIZE
image_dims = [settings.CHUNK_SIZE, settings.N_MELS]
input_shape = [*image_dims, window_size]
latent_dims = settings.LATENT_DIMS
num_conv = 2
num_filters = 32
max_filters = 64
kernel_size = 3
# endregion

# region Training hyperparameters
num_epochs = settings.EPOCHS
batch_size = settings.BATCH_SIZE
# endregion

# region Model definition
inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(num_conv):
    x = tf.keras.layers.Conv2D(
        filters=min(num_filters * (i + 1), max_filters),
        kernel_size=kernel_size,
        activation='relu',
        strides=2,
        padding='same',
        activity_regularizer=tf.keras.regularizers.l1(0.01)
    )(x)
latent_shape = x.shape
x = tf.keras.layers.Flatten()(x)
z_mean = tf.keras.layers.Dense(latent_dims, name='z_mean')(x)
z_log_var = tf.keras.layers.Dense(latent_dims, name='z_log_var')(x)
z = tf.keras.layers.Lambda(reparameterize, output_shape=[latent_dims], name='z')([z_mean, z_log_var])
encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

latent_inputs = tf.keras.layers.Input(shape=(latent_dims,), name='z_sampled')
x = tf.keras.layers.Dense(
    reduce(mul, latent_shape[1:]),
    activation='relu'
)(latent_inputs)
x = tf.keras.layers.Reshape(latent_shape[1:])(x)
for i in range(num_conv):
    x = tf.keras.layers.Conv2DTranspose(
        filters=min(num_filters * (num_conv - i), max_filters),
        kernel_size=kernel_size,
        strides=2,
        activation='relu',
        padding='same',
        activity_regularizer=tf.keras.regularizers.l1(0.01)
    )(x)
reconstructed = tf.keras.layers.Conv2DTranspose(
    filters=window_size,
    kernel_size=3,
    strides=1,
    padding='SAME',
    activation='sigmoid'
)(x)
decoder = tf.keras.Model(latent_inputs, reconstructed, name='decoder')
decoder.summary()
outputs = decoder(encoder(inputs)[2])
vae = tf.keras.Model(inputs, outputs, name='vae')
vae.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=vae_loss(z_mean, z_log_var, image_dims)
)
vae.summary()
# endregion

# region Train and evaluate
train_dataset, test_dataset = data_utils.load_numpy_dataset(settings.TRAIN_DATA_DIR, return_tuples=True)

# endregion

# class CVAE(tf.keras.Model):
#     def __init__(self, latent_dim=settings.LATENT_DIMS, num_conv=2, image_dims=(settings.CHUNK_SIZE, settings.N_MELS),
#                  window_size=settings.WINDOW_SIZE, num_filters=32, max_filters=64, kernel_size=3):
#         super(CVAE, self).__init__()
#         input_shape = [*image_dims, window_size]
#         self.window_size = window_size
#         self.image_dims = image_dims
#         self.latent_dims = latent_dim
#
#         inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
#         x = inputs
#         for i in range(num_conv):
#             x = tf.keras.layers.Conv2D(
#                 filters=min(num_filters * (i + 1), max_filters),
#                 kernel_size=kernel_size,
#                 activation='relu',
#                 strides=2,
#                 padding='same',
#                 activity_regularizer=tf.keras.regularizers.l1(0.01)
#             )(x)
#         latent_shape = x.shape
#         x = tf.keras.layers.Flatten()(x)
#         latent_vector = tf.keras.layers.Dense(latent_dim + latent_dim)(x)
#         self.encoder = tf.keras.Model(inputs, latent_vector, name='encoder')
#
#         latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
#         x = tf.keras.layers.Dense(
#             latent_shape[1] * latent_shape[2] * latent_shape[3],
#             activation='relu'
#         )(latent_inputs)
#         x = tf.keras.layers.Reshape(latent_shape[1:])(x)
#         for i in range(num_conv):
#             x = tf.keras.layers.Conv2DTranspose(
#                 filters=min(num_filters * (num_conv - i), max_filters),
#                 kernel_size=kernel_size,
#                 strides=2,
#                 activation='relu',
#                 padding='same',
#                 activity_regularizer=tf.keras.regularizers.l1(0.01)
#             )(x)
#         reconstructed = tf.keras.layers.Conv2DTranspose(
#             filters=self.window_size,
#             kernel_size=3,
#             strides=1,
#             padding='SAME'
#         )(x)
#         self.decoder = tf.keras.Model(latent_inputs, reconstructed, name='decoder')
#         self.encoder.summary()
#         self.decoder.summary()
#         tf.keras.utils.plot_model(self.encoder, to_file='encoder.png', show_shapes=True)
#         tf.keras.utils.plot_model(self.decoder, to_file='decoder.png', show_shapes=True)
#
#         # =====================================
#         # Use this to remove the `tf.split` op in .encode()
#         # self.latent_shape = x.shape
#         # self.z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
#         # self.z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
#         # z = self.reparameterize(self.z_mean, self.z_log_var)
#         # =====================================
#
#     def call(self, inputs, training=False):
#         if training:
#             latent = self.encode(inputs)
#
#         else:
#             return self.decode(inputs)
#
#     @staticmethod
#     def log_normal_pdf(sample, mean, logvar, raxis=1):
#         log2pi = tf.math.log(2. * np.pi)
#         return tf.reduce_sum(
#             -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#             axis=raxis)
#
#     @tf.function
#     def compute_loss(self, x):
#         mean, logvar = self.encode(x)
#         z = self.reparameterize(mean, logvar)
#         x_logit = self.decode(z)
#         cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
#         logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#         logpz = self.log_normal_pdf(z, 0., 0.)
#         logqz_x = self.log_normal_pdf(z, mean, logvar)
#         return -tf.reduce_mean(logpx_z + logpz - logqz_x)
#
#     @tf.function
#     def compute_apply_gradients(self, x, opt):
#         with tf.GradientTape() as tape:
#             loss = self.compute_loss(self, x)
#         gradients = tape.gradient(loss, self.trainable_variables)
#         opt.apply_gradients(zip(gradients, self.trainable_variables))
#
#     def decode(self, z, apply_sigmoid=False):
#         logits = self.decoder(z)
#         if apply_sigmoid:
#             return tf.sigmoid(logits)
#         return logits
#
#     @tf.function
#     def sample(self, eps=None):
#         if eps is None:
#             eps = tf.random.normal(shape=(100, self.latent_dims))
#         return self.decode(eps, apply_sigmoid=True)
#
#     def encode(self, x):
#         mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
#         return mean, logvar
#
#     @staticmethod
#     def reparameterize(mean, logvar):
#         eps = tf.random.normal(shape=mean.shape)
#         return eps * tf.exp(logvar * 0.5) + mean


# optimizer = tf.keras.optimizers.Adam(1e-4)
# model = CVAE(num_conv=4)
# model.compile(optimizer=optimizer)

# if os.path.exists(settings.MODEL_WEIGHTS):
#     print(f"Loading weights from '{settings.MODEL_WEIGHTS}'")
#     model.load_weights(settings.MODEL_WEIGHTS)

# num_train = num_test = 0
# generation_vector = tf.random.normal(shape=[settings.EXAMPLES_TO_GENERATE, model.latent_dims])
# visualiziation_output_dir = os.path.join(settings.OUTPUT_DIR, 'progress')
# visualize_model_outputs(model, 0, generation_vector, visualiziation_output_dir)
#
# for epoch in range(1, settings.EPOCHS + 1):
#     start = time.time()
#     print(f"Training | Epoch {epoch} / {settings.EPOCHS}...")
#     for train_x in tqdm(train_dataset, total=num_train or None):
#         compute_apply_gradients(model, train_x, optimizer)
#         if epoch == 1:
#             num_train += 1
#     print(f"Finished Train Step | Epoch {epoch} Train Step took {time.time() - start:.2f} seconds")
#
#     if epoch % 1 == 0:
#         # Evaluate Model
#         print(f"Evaluation | Epoch {epoch}...")
#         loss = tf.keras.metrics.Mean()
#         for test_x in tqdm(test_dataset, total=num_test):
#             loss(compute_loss(model, test_x))
#             if epoch == 1:
#                 num_test += 1
#         elbo = -loss.result()
#         print(f"Epoch {epoch} took {time.time() - start:.2f} seconds | Test Set ELBO: {elbo}")
#         # Save Model Weights
#         os.makedirs(os.path.dirname(settings.MODEL_WEIGHTS), exist_ok=True)  # Create dir if it doesn't exist
#         model.save_weights(settings.MODEL_WEIGHTS)
#         # Save Generated Samples
#         visualize_model_outputs(model, epoch, generation_vector, visualiziation_output_dir)

# start = time.time()
# for i, element in enumerate(tqdm(train_dataset.take(2000))):
#     # print(i, element)
#     tqdm.write(f"{i + 1}: {element.shape}")
#     pass
# print("----------------FINISHED----------------")
# print(time.time() - start)
