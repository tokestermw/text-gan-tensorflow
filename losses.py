""" Loss functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# TODO: check to see if we can separate generator
class GanTypes:
    jsd = "jsd"  # Jensen-Shannon Divergence aka Maximum Likelihood Estimate.
    emd = "emd"  # Earth Mover's Distance aka Wasserstein Distance.
    ls = "ls"  # Least Squares Distance.


def gan_loss(d_logits_real, d_logits_fake, gan_type="jsd"):
    if gan_type == "jsd":
        d_probs_real = tf.sigmoid(d_logits_real)
        d_probs_fake = tf.sigmoid(d_logits_fake)

        d_loss_real = tf.log(d_probs_real)
        d_loss_fake = tf.log(1. - d_probs_fake)

        d_loss = -tf.reduce_mean(d_loss_real + d_loss_fake)
        g_loss = -tf.reduce_mean(d_loss_fake)

    elif gan_type == "emd":
        # batch size should be first dim
        d_mean_real = tf.reduce_mean(d_logits_real, axis=-1)
        d_mean_fake = tf.reduce_mean(d_logits_fake, axis=-1)

        d_loss = tf.reduce_mean(d_mean_real - d_mean_fake)
        g_loss = -tf.reduce_mean(d_mean_fake)

    elif gan_type == "ls":
        d_sq_real = (d_logits_real - 1.) ** 2.
        d_sq_fake = d_logits_fake ** 2.

        d_loss = 0.5 * (tf.reduce_mean(d_sq_real) + tf.reduce_mean(d_sq_fake))
        g_loss = 0.5 * tf.reduce_mean((d_logits_fake - 1.) ** 2.)

    else:
        raise ValueError("Wrong gan_type.")

    return d_loss, g_loss

