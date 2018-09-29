"""
Keras based implementation of Autoencoding
beyond pixels using a learned similarity metric.

References:
- https://arxiv.org/abs/1512.09300
- https://arxiv.org/abs/1511.06434
"""

import numpy as np
import tensorflow as tf
from utils import load, save
from tensorflow.python.keras import initializers, backend as K
from tensorflow.python.keras.layers import (Input, Dense, Reshape, Activation,
                                            Conv2D, Conv2DTranspose, LeakyReLU,
                                            Flatten, BatchNormalization as BN)
from tensorflow.python.keras.models import Sequential, Model

learning_rate = .0002
beta1 = .5
z_dim = 512


def cleanup(data):
    X = data
    X = X/127.5 - 1.
    Z = np.random.normal(0, 1, (X.shape[0], z_dim))
    return Z, X

def generator(batch_size, gf_dim, ch, rows, cols):
    """
    Generator model.

    Generates image from compressed image representation (vector Z).

    Input: Z (compressed image representation)
    Output: generated image
    """
    model = Sequential()

    # Input layer.
    model.add(Dense(gf_dim*8*rows[0]*cols[0],
                    name="g_h0_lin",
                    batch_input_shape=(batch_size, z_dim),
                    kernel_initializer=initializers.random_normal(
                        stddev=.02)))
    model.add(Reshape((rows[0], cols[0], gf_dim*8)))
    model.add(BN(axis=3, name="g_bn0", epsilon=1e-5,
                 gamma_initializer=initializers.random_normal(
                     mean=1., stddev=.02)))
    model.add(Activation("relu"))

    # Layer 1.
    model.add(Conv2DTranspose(gf_dim*4, kernel_size=(5, 5), strides=(2, 2),
                              padding="same", name="g_h1",
                              kernel_initializer=initializers.random_normal(
                                  stddev=.02)))
    model.add(BN(axis=3, name="g_bn1", epsilon=1e-5,
                 gamma_initializer=initializers.random_normal(
                     mean=1., stddev=.02)))
    model.add(Activation("relu"))

    # Layer 2.
    model.add(Conv2DTranspose(gf_dim*2, kernel_size=(5, 5), strides=(2, 2),
                              padding="same", name="g_h2",
                              kernel_initializer=initializers.random_normal(
                                  stddev=.02)))
    model.add(BN(axis=3, name="g_bn2", epsilon=1e-5,
                 gamma_initializer=initializers.random_normal(
                     mean=1., stddev=.02)))
    model.add(Activation("relu"))

    # Layer 3.
    model.add(Conv2DTranspose(gf_dim, kernel_size=(5, 5), strides=(2, 2),
                              padding="same", name="g_h3",
                              kernel_initializer=initializers.random_normal(
                                  stddev=.02)))
    model.add(BN(axis=3, name="g_bn3", epsilon=1e-5,
                 gamma_initializer=initializers.random_normal(
                     mean=1., stddev=.02)))
    model.add(Activation("relu"))

    # Output layer.
    model.add(Conv2DTranspose(ch, kernel_size=(5, 5), strides=(2, 2),
                              padding="same", name="g_h4",
                              kernel_initializer=initializers.random_normal(
                                  stddev=.02)))
    model.add(Activation("tanh"))

    return model


def encoder(batch_size, df_dim, ch, rows, cols):
    """
    Encoder model.

    Encodes image into compressed image representation (vector Z).

    Input: image
    Output: Z (compressed image representation)
    """
    model = Sequential()

    # Input layer.
    X = Input(batch_shape=(batch_size, rows[-1], cols[-1], ch))

    # Layer 1.
    model = Conv2D(df_dim, kernel_size=(5, 5), strides=(2, 2),
                   padding="same", name="e_h0_conv",
                   kernel_initializer=initializers.random_normal(
                       stddev=.02))(X)
    model = LeakyReLU(.2)(model)

    # Layer 2.
    model = Conv2D(df_dim*2, kernel_size=(5, 5), strides=(2, 2),
                   padding="same", name="e_h1_conv")(model)
    model = BN(axis=3, name="e_bn1", epsilon=1e-5,
               gamma_initializer=initializers.random_normal(
                   mean=1., stddev=.02))(model)
    model = LeakyReLU(.2)(model)

    # Layer 3.
    model = Conv2D(df_dim*4, kernel_size=(5, 5), strides=(2, 2),
                   name="e_h2_conv", padding="same",
                   kernel_initializer=initializers.random_normal(
                       stddev=.02))(model)
    model = BN(axis=3, name="e_bn2", epsilon=1e-5,
               gamma_initializer=initializers.random_normal(
                   mean=1., stddev=.02))(model)
    model = LeakyReLU(.2)(model)

    # Layer 4.
    model = Conv2D(df_dim*8, kernel_size=(5, 5), strides=(2, 2),
                   padding="same", name="e_h3_conv",
                   kernel_initializer=initializers.random_normal(
                       stddev=.02))(model)
    model = BN(axis=3, name="e_bn3", epsilon=1e-5,
               gamma_initializer=initializers.random_normal(
                   mean=1., stddev=.02))(model)
    model = LeakyReLU(.2)(model)
    model = Flatten()(model)

    # Output layer 1.
    mean = Dense(z_dim, name="e_h3_lin",
                 kernel_initializer=initializers.random_normal(
                     stddev=.02))(model)

    # Output layer 2.
    logsigma = Dense(z_dim, name="e_h4_lin",
                     activation="tanh",
                     kernel_initializer=initializers.random_normal(
                         stddev=.02))(model)

    return Model([X], [mean, logsigma])


def discriminator(batch_size, df_dim, ch, rows, cols):
    """
    Discriminator model.

    Similar to encoder, but main purpose to discinguish
    between fake (generated) and legit (original) images.

    Input: image
    Output: scalar (fake/legit)
    """
    # Input layer.
    X = Input(batch_shape=(batch_size, rows[-1], cols[-1], ch))

    # Layer 1.
    model = Conv2D(df_dim, kernel_size=(5, 5), strides=(2, 2),
                   padding="same", name="d_h0_conv",
                   batch_input_shape=(batch_size, rows[-1], cols[-1], ch),
                   kernel_initializer=initializers.random_normal(
                       stddev=.02))(X)
    model = LeakyReLU(.2)(model)

    # Layer 2.
    model = Conv2D(df_dim*2, kernel_size=(5, 5), strides=(2, 2),
                   padding="same", name="d_h1_conv",
                   kernel_initializer=initializers.random_normal(
                       stddev=.02))(model)
    model = BN(axis=3, name="d_bn1", epsilon=1e-5,
               gamma_initializer=initializers.random_normal(
                   mean=1., stddev=.02))(model)
    model = LeakyReLU(.2)(model)

    # Layer 3.
    model = Conv2D(df_dim*4, kernel_size=(5, 5), strides=(2, 2),
                   padding="same", name="d_h2_conv",
                   kernel_initializer=initializers.random_normal(
                       stddev=.02))(model)
    model = BN(axis=3, name="d_bn2", epsilon=1e-5,
               gamma_initializer=initializers.random_normal(
                   mean=1., stddev=.02))(model)
    model = LeakyReLU(.2)(model)

    # Layer 4. Also outputs convolution layer.
    model = Conv2D(df_dim*8, kernel_size=(5, 5), strides=(2, 2),
                   padding="same", name="d_h3_conv",
                   kernel_initializer=initializers.random_normal(
                       stddev=.02))(model)
    dec = BN(axis=3, name="d_bn3", epsilon=1e-5,
             gamma_initializer=initializers.random_normal(
                 mean=1., stddev=.02))(model)
    dec = LeakyReLU(.2)(dec)
    dec = Flatten()(dec)

    # Output layer.
    dec = Dense(1, name="d_h3_lin",
                kernel_initializer=initializers.random_normal(
                    stddev=.02))(dec)

    return Model([X], [dec, model])


def get_model(sess, image_shape=(80, 160, 3), gf_dim=64, df_dim=64,
              batch_size=64, name="autoencoder", gpu=0):
    """
    Compiles and outputs models and functions for training and running models.
    """
    K.set_session(sess)
    checkpoint_dir = './outputs/results_' + name
    with tf.variable_scope(name), tf.device("/gpu:{}".format(gpu)):
        # sizes
        ch = image_shape[2]
        rows = [int(image_shape[0]/i) for i in [16, 8, 4, 2, 1]]
        cols = [int(image_shape[1]/i) for i in [16, 8, 4, 2, 1]]

        # nets
        G = generator(batch_size, gf_dim, ch, rows, cols)
        G.compile("sgd", "mse")
        g_vars = G.trainable_weights
        print("G.shape: ", G.output_shape)

        E = encoder(batch_size, df_dim, ch, rows, cols)
        E.compile("sgd", "mse")
        e_vars = E.trainable_weights
        print("E.shape: ", E.output_shape)

        D = discriminator(batch_size, df_dim, ch, rows, cols)
        D.compile("sgd", "mse")
        d_vars = D.trainable_weights
        print("D.shape: ", D.output_shape)

        Z2 = Input(batch_shape=(batch_size, z_dim), name='more_noise')
        Z = G.input
        Img = D.input
        G_train = G(Z)
        E_mean, E_logsigma = E(Img)
        G_dec = G(E_mean + Z2 * E_logsigma)
        D_fake, F_fake = D(G_train)
        D_dec_fake, F_dec_fake = D(G_dec)
        D_legit, F_legit = D(Img)

        # Loss function:
        #
        #   L = L prior + L Disllike + L GAN
        #
        # costs
        recon_vs_gan = 1e-6

        # L Disllike
        like_loss = tf.reduce_mean(tf.square(F_legit - F_dec_fake)) / 2.

        # L prior (VAE)
        kl_loss = tf.reduce_mean(-E_logsigma + .5 * (-1 + tf.exp(2. * E_logsigma) + tf.square(E_mean)))

        # L GAN
        d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_legit), logits=D_legit))
        d_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_fake), logits=D_fake))
        d_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_dec_fake), logits=D_dec_fake))
        d_loss_fake = d_loss_fake1 + d_loss_fake2

        g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_fake), logits=D_fake))
        g_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_dec_fake), logits=D_dec_fake))

        # Final loss functions.
        d_loss = d_loss_legit + d_loss_fake
        g_loss = g_loss1 + g_loss2 + recon_vs_gan * like_loss
        e_loss = kl_loss + like_loss

        # Optimizers.
        print("Generator variables:")
        for v in g_vars:
            print(v.name)
        print("Discriminator variables:")
        for v in d_vars:
            print(v.name)
        print("Encoder variables:")
        for v in e_vars:
            print(v.name)

        e_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(e_loss, var_list=e_vars)
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        tf.global_variables_initializer().run()

    # summaries
    sum_d_loss_legit = tf.summary.scalar("d_loss_legit", d_loss_legit)
    sum_d_loss_fake = tf.summary.scalar("d_loss_fake", d_loss_fake)
    sum_d_loss = tf.summary.scalar("d_loss", d_loss)
    sum_g_loss = tf.summary.scalar("g_loss", g_loss)
    sum_e_loss = tf.summary.scalar("e_loss", e_loss)
    sum_e_mean = tf.summary.histogram("e_mean", E_mean)
    sum_e_sigma = tf.summary.histogram("e_sigma", tf.exp(E_logsigma))
    sum_Z = tf.summary.histogram("Z", Z)
    sum_gen = tf.summary.image("G", G_train)
    sum_dec = tf.summary.image("E", G_dec)

    # saver
    saver = tf.train.Saver()
    g_sum = tf.summary.merge([sum_Z, sum_gen, sum_d_loss_fake, sum_g_loss])
    e_sum = tf.summary.merge([sum_dec, sum_e_loss, sum_e_mean, sum_e_sigma])
    d_sum = tf.summary.merge([sum_d_loss_legit, sum_d_loss])
    writer = tf.summary.FileWriter("/tmp/logs/"+name, sess.graph)

    # functions
    def train_d(images, z, counter, sess=sess):
        z2 = np.random.normal(0., 1., z.shape)
        outputs = [d_loss, d_loss_fake, d_loss_legit, d_sum, d_optim]
        with tf.control_dependencies(outputs):
          updates = []
          for update in D.updates:
            if isinstance(update, tuple):
              p, new_p = update
              updates.append(tf.assign(p, new_p))
            else:
              # assumed already an op
              updates.append(update)
        outs = sess.run(outputs + updates, feed_dict={
            Img: images, Z: z, Z2: z2, K.learning_phase(): 1
            })
        dl, dlf, dll, sums = outs[:4]
        writer.add_summary(sums, counter)
        return dl, dlf, dll

    def train_g(images, z, counter, sess=sess):
        # generator
        z2 = np.random.normal(0., 1., z.shape)
        outputs = [g_loss, G_train, g_sum, g_optim]
        with tf.control_dependencies(outputs):
          updates = []
          for update in G.updates:
            if isinstance(update, tuple):
              p, new_p = update
              updates.append(tf.assign(p, new_p))
            else:
              # assumed already an op
              updates.append(update)
        outs = sess.run(outputs + updates, feed_dict={
            Img: images, Z: z, Z2: z2, K.learning_phase(): 1
            })
        gl, samples, sums = outs[:3]
        writer.add_summary(sums, counter)
        # encoder
        outputs = [e_loss, G_dec, e_sum, e_optim]
        with tf.control_dependencies(outputs):
          updates = []
          for update in G.updates:
            if isinstance(update, tuple):
              p, new_p = update
              updates.append(tf.assign(p, new_p))
            else:
              # assumed already an op
              updates.append(update)
        outs = sess.run(outputs + updates, feed_dict={
            Img: images, Z: z, Z2: z2, K.learning_phase(): 1
            })
        gl, samples, sums = outs[:3]
        writer.add_summary(sums, counter)

        return gl, samples, images

    def f_load():
        try:
            return load(sess, saver, checkpoint_dir, name)
        except Exception:
            print("Loading weights via Keras")
            G.load_weights(checkpoint_dir+"/G_weights.keras")
            D.load_weights(checkpoint_dir+"/D_weights.keras")
            E.load_weights(checkpoint_dir+"/E_weights.keras")

    def f_save(step):
        save(sess, saver, checkpoint_dir, step, name)
        G.save_weights(checkpoint_dir+"/G_weights.keras", True)
        D.save_weights(checkpoint_dir+"/D_weights.keras", True)
        E.save_weights(checkpoint_dir+"/E_weights.keras", True)

    def sampler(z, x):
        code = E.predict(x, batch_size=batch_size)[0]
        out = G.predict(code, batch_size=batch_size)
        return out, x

    return train_g, train_d, sampler, f_save, f_load, [G, D, E]
