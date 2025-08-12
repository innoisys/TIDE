import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_conv(input, filters, size, strides=1, normalization=False):
    input = layers.Conv2D(filters, size, activation='relu', strides=strides, padding="same")(input)
    # input = layers.ThresholdedReLU(6)(input)
    # x = layers.ThresholdedReLU(6)(x)
    # if normalization:
    #    x = layers.BatchNormalization()(x)
    return input


def create_tconv(input, filters, size, strides=1, normalization=False):
    input = layers.Conv2DTranspose(filters, size, activation='relu', strides=strides, padding="same")(input)
    # input = layers.ThresholdedReLU(6)(input)
    # if normalization:
    #    x = layers.BatchNormalization()(x)
    return input


def create_msb_conv_old(input, filters):
    s = create_conv(input, filters, (3, 3))
    m = create_conv(input, filters, (5, 5))
    l = create_conv(input, filters, (7, 7))
    c = layers.concatenate([s, m, l], -1)
    c = layers.Conv2D(filters, (1, 1), 1, activation=None, use_bias=False, padding='same')(c)  # * 3
    c = layers.ThresholdedReLU(6)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Conv2D(filters, (3, 3), 1, activation=None, padding='same')(c)
    c = layers.ThresholdedReLU(6)(c)
    out = layers.BatchNormalization()(c)
    return out


def create_msb_conv(input, filters):
    s = create_conv(input, filters, 3)
    m = create_conv(input, filters, 5)
    l = create_conv(input, filters, 7)
    c = layers.concatenate([s, m, l], -1)
    input = layers.Conv2D(filters * 3, 1, 1, activation='relu', use_bias=False, padding='same')(c)  # * 3
    input = layers.Conv2D(filters, 3, 1, activation='relu', padding='same')(input)
    return input


def create_peephole_old(input, output):
    output_filters = output.shape[-1]
    i = create_conv(input, filters=output_filters, size=(3, 3), strides=1)
    out = layers.Add()([i, output])
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(output_filters, (1, 1), 1, activation=None, use_bias=False, padding='same')(out)
    out = layers.ThresholdedReLU(6)(out)
    return out


def create_peephole(input, output):
    output_filters = output.shape[-1]
    i = create_conv(input, filters=output_filters, size=(3, 3), strides=1)
    out = layers.Add()([i, output])
    out = layers.Conv2D(output_filters, (1, 1), 1, activation='relu', use_bias=False, padding='same')(out)
    return out


def create_encoder(latent_dim=2, input_shape=(96, 96, 3)):
    encoder_inputs = keras.Input(shape=input_shape)
    x_in = create_conv(encoder_inputs, 16, 3, 1)
    x = create_msb_conv(x_in, 32)
    x = create_peephole(x_in, x)
    x = create_conv(x, 32, 3, 2)
    x_in = create_conv(x, 64, 3, 2)
    x = create_msb_conv(x_in, 64)
    x_in = create_peephole(x_in, x)
    x = create_msb_conv(x_in, 128)
    x_in = create_peephole(x_in, x)
    x = create_msb_conv(x_in, 256)
    x = create_peephole(x_in, x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def create_decoder(latent_dim=2):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(12 * 12 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((12, 12, 256))(x)
    x_in = create_tconv(x, 256, 3, 2)
    x = create_msb_conv(x_in, 256)
    x_in = create_peephole(x_in, x)
    x = create_msb_conv(x_in, 128)
    x = create_peephole(x_in, x)
    x_in = create_tconv(x, 64, 3, 2)
    x = create_msb_conv(x_in, 64)
    x = create_peephole(x_in, x)
    x_in = create_tconv(x, 32, 3, 2)
    x = create_msb_conv(x_in, 32)
    x = create_peephole(x_in, x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


def create_decoder_old(latent_dim=2):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(12 * 12 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((12, 12, 256))(x)
    x = create_tconv(x, 256, 3, 2)
    x = create_tconv(x, 128, 3, 1)
    x = create_tconv(x, 64, 3, 2)
    x = create_tconv(x, 64, 3, 1)
    x = create_tconv(x, 32, 3, 2)
    x = create_tconv(x, 32, 3, 1)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function()
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def full_summary(self):
        for layer in self.layers:
            print(layer.summary())

    @tf.function()
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            if tf.math.is_nan(kl_loss) or tf.math.is_inf(kl_loss):
                kl_loss = tf.float32.max
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
