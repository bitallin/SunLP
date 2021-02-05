#!/usr/bin/env python
# encoding: utf-8
"""
    @File: dataset.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
import json
from typing import List

import tensorflow as tf
import tensorflow.keras as keras

from sunlp import logger


class Dataset:
    def __init__(self, max_seq_len, pad_idx=0, unk_idx=1):

        self.max_seq_len = max_seq_len
        self._PAD_IDX = pad_idx
        self._UNK_IDX = unk_idx
        self.char2id = {'<PAD>': pad_idx, '<UNK>': unk_idx}
        self.label2id = {'<UNK>': unk_idx}

    def fit_vocab(self, data_file) -> (dict, dict):
        """
            build/expand vocab by dataset file
        Args:
            sents:
            sents_labels:

        Returns:
        """
        sents, sents_labels = self._read_file_data(data_file)
        [[self.char2id.setdefault(char, len(self.char2id)) for char in sent] for sent in sents]
        [[self.label2id.setdefault(label, len(self.label2id)) for label in sent] for sent in sents_labels]
        logger.debug("Vocab has been fitted")

    def load_vocab_file(self, vocab_file):
        """
            load vocab from vocab file
        """
        with open("{}".format(vocab_file), 'r', encoding='utf8') as f:
            json_data = json.loads(f.read(), encoding='utf8')
            self.char2id = json_data['char2id']
            self.label2id = json_data['label2id']
            logger.debug('Vocab loaded from {}'.format(vocab_file))

    def save_vocab(self, file_path):
        vocab_json = {
            "char2id": self.char2id,
            "label2id": self.label2id,
        }
        with open(file_path, 'w', encoding='utf8') as f:
            f.write(json.dumps(vocab_json, ensure_ascii=False, indent=4))
            f.close()
            logger.debug('Vocab has been saved at {}'.format(file_path))

    @staticmethod
    def _read_file_data(data_fp) -> (List[list], List[list]):
        sents, sents_labels = [], []
        sent = []
        label = []
        for line in open(data_fp, encoding='utf8'):
            if line == '\n':
                sents.append(sent), sents_labels.append(label)
                sent, label = [], []
                continue
            _char, _label = line.strip().split()
            sent.append(_char), label.append(_label)
        return sents, sents_labels

    def build_dataset(self, data_file) -> tf.data.Dataset:
        # to id
        sents, sents_labels = self._read_file_data(data_file)
        sents = list(
            map(lambda sent: list(map(lambda char: self.char2id.get(char, self.char2id['<UNK>']), sent)), sents))
        seq_len = list(map(lambda sent: len(sent) if len(sent) <= self.max_seq_len else self.max_seq_len, sents))
        sents_labels = list(
            map(lambda sent: list(map(lambda label: self.label2id.get(label, self.label2id['<UNK>']), sent)),
                sents_labels))

        # padding
        sents = keras.preprocessing.sequence.pad_sequences(sents, maxlen=self.max_seq_len,
                                                           padding='post', truncating='post',
                                                           value=self._PAD_IDX)
        sents_labels = keras.preprocessing.sequence.pad_sequences(sents_labels, maxlen=self.max_seq_len,
                                                                  padding='post', truncating='post',
                                                                  value=self._PAD_IDX)
        # to tensor
        dataset = tf.data.Dataset.from_tensor_slices({'x': sents, 'y': sents_labels, 'seq_len': seq_len})
        logger.debug('Dataset has been built')
        return dataset
