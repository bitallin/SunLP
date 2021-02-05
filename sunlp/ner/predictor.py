#!/usr/bin/env python
# encoding: utf-8
"""
    @File: predictor.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
import json

import tensorflow as tf

from sunlp import logger
from sunlp.base.predictor import BasePredictor
from sunlp.ner.model import MODELS


class Predictor(BasePredictor):
    def __init__(self, model_name='birnn_atten_crf', model_ckpt_dir='saved',
                 vocab_file='vocab.json', model_conf_file='model_conf.json'):
        self.model_name = model_name
        self.model_ckpt_dir = model_ckpt_dir
        self.vocab_file = vocab_file
        self.model_conf_file = model_conf_file
        self._initialize()

    def _initialize(self):
        self._load_vocab()
        self._load_model()

    def _load_vocab(self):
        with open("{}/{}".format(self.model_ckpt_dir, self.vocab_file), 'r', encoding='utf8') as f:
            vocab_json = json.loads(f.read())
        self.char2id = vocab_json['char2id']
        self.label2id = vocab_json['label2id']

        self.id2char = {int(idx): char for char, idx in self.char2id.items()}
        self.id2label = {int(idx): label for label, idx in self.label2id.items()}

    def _load_model(self):
        with open("{}/{}".format(self.model_ckpt_dir, self.model_conf_file), 'r', encoding='utf8', ) as f:
            model_config = json.loads(f.read())
        self._max_seq_len = model_config['max_seq_len']
        self._model = MODELS[self.model_name](**model_config)
        ckpt = tf.train.Checkpoint(model=self._model)
        ckpt.restore(tf.train.latest_checkpoint(self.model_ckpt_dir))
        logger.debug('Model loaded successfully!!')

    def _text_to_tensor(self, texts):
        _texts = [list(text) for text in texts]
        _texts = [list(map(lambda char: self.char2id.get(char, '<UNK>'), text)) for text in _texts]
        _seq_len = [len(text) if len(text) <= self._max_seq_len else self._max_seq_len for text in _texts]
        _texts = tf.keras.preprocessing.sequence.pad_sequences(_texts, maxlen=self._max_seq_len, padding='post',
                                                               truncating='post', value=self.char2id['PAD'])
        dataset = tf.data.Dataset.from_tensor_slices({'x': _texts, 'seq_len': _seq_len})
        return dataset

    def predict(self, texts, batch_size=8):
        preds = []
        dataset = self._text_to_tensor(texts).batch(batch_size)
        for batch_data in dataset:
            batch_pred = self._model(batch_data, training=False).numpy()

            batch_pred = [list(map(lambda label: self.id2label.get(label, '<UNK>'), pred))[:seq_len] for pred, seq_len
                          in zip(batch_pred, batch_data['seq_len'])]

            preds.extend(batch_pred)
        texts = [list(text) for text in texts]
        return list(zip(texts, preds))
