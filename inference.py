""" Inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# -- local imports
from utils import set_logging_verbosity
from data_loader import DATA_PATH, tokenize, vectorize
from layers import _phase_train, _phase_infer
from model import Model
from search import reverse_decode, greedy_argmax


def restore_model(sess, model_dir):
    ckpt_path = tf.train.latest_checkpoint(model_dir)

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)


def generate_sample(sess, seed_text, probs, input_ph, word2idx, idx2word):
    # seed_text = "how are you"
    vector = vectorize(seed_text, word2idx)
    out = greedy_argmax(vector[:-1], lambda x: sess.run(probs, {input_ph: [x]}))
    text = reverse_decode(out, idx2word)
    return text


if __name__ == "__main__":
    from train import opts, FLAGS

    corpus = DATA_PATH[FLAGS.corpus_name]
    # TODO: move to flags
    model = Model(corpus, **opts)

    g_probs = model.g_tensors_pretrain_valid.probs 

    with tf.Session() as sess:
        restore_model(sess, FLAGS.model_dir)

        sess.run(_phase_infer)

        while True:
            seed_text = input("Enter seed text: ")
            generated_text = generate_sample(sess, seed_text, g_probs, model.input_ph, model.word2idx, model.idx2word)
            print(generated_text)
