import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image

from . import data_utils
from .. import settings

tf.compat.v1.enable_eager_execution()


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def visualize_model_outputs(model, epoch, test_input, output):
    output = Path(output)
    predictions = model.sample(eps=test_input)
    print(f"Saving Samples Images to {output}")
    for i, pred in enumerate(predictions):
        progress_dir = (
            Path(settings.MODEL_WEIGHTS or datetime.now().strftime("%Y%m%d-%H%M%S"))
            .resolve()
            .stem
        )
        out_dir = output.joinpath(progress_dir).joinpath(str(i + 1))
        os.makedirs(out_dir, exist_ok=True)
        image = Image.fromarray(pred[:, :, 0].numpy(), mode="F")
        image.save(os.path.join(out_dir, f"epoch_{epoch}.tiff"))


class CVAE(tf.keras.Model):
    def __init__(
        self,
        latent_dim=settings.LATENT_DIMS,
        num_conv=2,
        image_dims=(settings.CHUNK_SIZE, settings.N_MELS),
        window_size=settings.WINDOW_SIZE,
        num_filters=32,
        max_filters=64,
        kernel_size=3,
    ):
        super(CVAE, self).__init__()
        input_shape = [*image_dims, window_size]
        self.window_size = window_size
        self.image_dims = image_dims
        self.latent_dims = latent_dim

        inputs = tf.keras.layers.Input(shape=input_shape, name="encoder_input")
        x = inputs
        for i in range(num_conv):
            x = tf.keras.layers.Conv2D(
                filters=min(num_filters * (i + 1), max_filters),
                kernel_size=kernel_size,
                activation="relu",
                strides=2,
                padding="same",
                activity_regularizer=tf.keras.regularizers.l1(0.01),
            )(x)
        latent_shape = x.shape
        x = tf.keras.layers.Flatten()(x)
        latent_vector = tf.keras.layers.Dense(latent_dim + latent_dim)(x)
        self.encoder = tf.keras.Model(inputs, latent_vector, name="encoder")

        latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(
            latent_shape[1] * latent_shape[2] * latent_shape[3], activation="relu"
        )(latent_inputs)
        x = tf.keras.layers.Reshape(latent_shape[1:])(x)
        for i in range(num_conv):
            x = tf.keras.layers.Conv2DTranspose(
                filters=min(num_filters * (num_conv - i), max_filters),
                kernel_size=kernel_size,
                strides=2,
                activation="relu",
                padding="same",
                activity_regularizer=tf.keras.regularizers.l1(0.01),
            )(x)
        reconstructed = tf.keras.layers.Conv2DTranspose(
            filters=self.window_size, kernel_size=3, strides=1, padding="SAME"
        )(x)
        self.decoder = tf.keras.Model(latent_inputs, reconstructed, name="decoder")
        self.encoder.summary()
        self.decoder.summary()
        tf.keras.utils.plot_model(self.encoder, to_file="encoder.png", show_shapes=True)
        tf.keras.utils.plot_model(self.decoder, to_file="decoder.png", show_shapes=True)

        # =====================================
        # Use this to remove the `tf.split` op in .encode()
        # self.latent_shape = x.shape
        # self.z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
        # self.z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
        # z = self.reparameterize(self.z_mean, self.z_log_var)
        # =====================================

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dims))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean


train_dataset, test_dataset = data_utils.load_numpy_dataset(settings.TRAIN_DATA_DIR)

optimizer = tf.keras.optimizers.Adam(1e-4)
model = CVAE(num_conv=4)
if os.path.exists(settings.MODEL_WEIGHTS):
    print(f"Loading weights from '{settings.MODEL_WEIGHTS}'")
    model.load_weights(settings.MODEL_WEIGHTS)

num_train = num_test = 0
generation_vector = tf.random.normal(
    shape=[settings.EXAMPLES_TO_GENERATE, model.latent_dims]
)
visualiziation_output_dir = os.path.join(settings.OUTPUT_DIR, "progress")
visualize_model_outputs(model, 0, generation_vector, visualiziation_output_dir)

for epoch in range(1, settings.EPOCHS + 1):
    start = time.time()
    print(f"Training | Epoch {epoch} / {settings.EPOCHS}...")
    for train_x in tqdm(train_dataset, total=num_train or None):
        compute_apply_gradients(model, train_x, optimizer)
        if epoch == 1:
            num_train += 1
    print(
        f"Finished Train Step | Epoch {epoch} Train Step took {time.time() - start:.2f} seconds"
    )

    if epoch % 1 == 0:
        # Evaluate Model
        print(f"Evaluation | Epoch {epoch}...")
        loss = tf.keras.metrics.Mean()
        for test_x in tqdm(test_dataset, total=num_test):
            loss(compute_loss(model, test_x))
            if epoch == 1:
                num_test += 1
        elbo = -loss.result()
        print(
            f"Epoch {epoch} took {time.time() - start:.2f} seconds | Test Set ELBO: {elbo}"
        )
        # Save Model Weights
        os.makedirs(
            os.path.dirname(settings.MODEL_WEIGHTS), exist_ok=True
        )  # Create dir if it doesn't exist
        model.save_weights(settings.MODEL_WEIGHTS)
        # Save Generated Samples
        visualize_model_outputs(
            model, epoch, generation_vector, visualiziation_output_dir
        )

# start = time.time()
# for i, element in enumerate(tqdm(train_dataset.take(2000))):
#     # print(i, element)
#     tqdm.write(f"{i + 1}: {element.shape}")
#     pass
# print("----------------FINISHED----------------")
# print(time.time() - start)
