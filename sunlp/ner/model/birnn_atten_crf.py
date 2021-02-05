#!/usr/bin/env python
# encoding: utf-8
"""
    @File: birnn_atten_crf.py
    @Desc: RNN/GRU/LSTM/Self ATTENTION/CRF
    @Reference:
    @Author: letter
    @Contact: 5403517@qq.com
"""
import json

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

# 设置GPU内存可增长
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class BiRnnAttentionCrfModel(keras.Model):

    def __init__(self, vocab_size, num_tags, d_model=128, rnn_type='gru', rnn_layers=1, max_seq_len=128):
        super(BiRnnAttentionCrfModel, self).__init__()

        rnn_type = rnn_type.lower()
        assert rnn_type in ('rnn', 'gru', 'lstm'), "Check rnn_type in ('rnn', 'gru', 'lstm') !!!"
        rnn_layer = {'rnn': layers.RNN, 'gru': layers.GRU, 'lstm': layers.LSTM}

        self.embed = layers.Embedding(vocab_size, d_model)
        self.rnn_net = [layers.Bidirectional(rnn_layer[rnn_type](units=d_model, return_sequences=True))
                        for _ in range(rnn_layers)]
        self.linear_q, self.linear_k, self.linear_v = (layers.Dense(d_model) for _ in range(3))
        self.linear_tag = layers.Dense(num_tags)

        # init transition_params
        initializer = tf.keras.initializers.GlorotUniform()
        self.transition_params = tf.Variable(initializer([num_tags, num_tags]), "transitions")
        self.config = {
            'vocab_size': vocab_size,
            'num_tags': num_tags,
            'd_model': d_model,
            'rnn_type': rnn_type,
            'rnn_layers': rnn_layers,
            'max_seq_len': max_seq_len,
        }

    def call(self, inputs, training=None, mask=None):
        """
        Args:

            inputs:
                Training or Evaluation
                (x, y, seq_len)
                    - x  (batch,seq_len)
                    - y  (batch,seq_len)
                    - seq_len  (batch)
                Inference
                (x, seq_len)
                    - x  (batch,seq_len)
                    - seq_len  (batch)
            mask:
            training:
        Returns:
            batch_pred_sequence:
                (batch, seq_len): label id
            log_likelihood:
                (batch)  crf loss logic
        """
        # === Encoder ===
        x, seq_len = inputs['x'], inputs['seq_len']
        x = self.embed(x)
        for rnn in self.rnn_net:
            x = rnn(x)

        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        # === Self Attention ===
        w = tf.nn.softmax(tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])), axis=-1)
        x = tf.matmul(w, v)

        # === CRF ===
        x = self.linear_tag(x)
        batch_pred_sequence, batch_viterbi_score = tfa.text.crf_decode(x, self.transition_params, seq_len)
        if training:
            log_likelihood, self.transition_params = tfa.text.crf_log_likelihood(x,
                                                                                 inputs['y'],
                                                                                 seq_len,
                                                                                 self.transition_params,
                                                                                 )
            return batch_pred_sequence, log_likelihood

        else:
            return batch_pred_sequence

    @staticmethod
    def default_config():
        return {
            'vocab_size': None,
            'num_tags': None,
            'd_model': 128,
            'rnn_type': 'gru',
            'rnn_layers': 1,
            'max_seq_len': 128,
        }

    def save_config(self, file_path):
        with open(file_path, 'w', encoding='utf8', ) as f:
            f.write(json.dumps(self.config, indent=4))
            f.close()
    def model_config(self):
        pass



