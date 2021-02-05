#!/usr/bin/env python
# encoding: utf-8
"""
    @File: trainer.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
# !/usr/bin/env python
# encoding: utf-8

import os

import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score

from sunlp import logger
from sunlp.base.trainer import BaseTrainer
from sunlp.ner.data import Dataset
from sunlp.ner.model import MODELS


class SequenceTaggingTrainer(BaseTrainer):
    """
        ner任务的训练器，可选择model中任一模型训练，数据需保持数据规范，更换模型时，需从模型从获取到参数来调参
    """

    def __init__(self,
                 model_name,
                 model_config: dict,
                 saved_dir='saved', data_dir='data',
                 train_file='train.txt', dev_file='dev.txt',
                 vocab_file='vocab.json', model_conf_file='model_conf.json',
                 model_ckpt_name='model.ckpt',
                 rebuild_vocab=True,
                 rebuild_model=True,
                 lr=1e-3,
                 max_seq_len=128, pad_idx=0, unk_idx=1):

        self.model_name = model_name
        self.model_config = model_config
        self.save_dir = saved_dir
        self.data_dir = data_dir

        self.train_file = train_file
        self.dev_file = dev_file
        self.vocab_file = vocab_file
        self.model_conf_file = model_conf_file
        self.model_ckpt_name = model_ckpt_name

        self.lr = lr
        self.max_seq_len = max_seq_len

        self._PAD_IDX = pad_idx
        self._UNK_IDX = unk_idx
        self.dataset_help = None
        self.rebuild_model = rebuild_model
        self.rebuild_vocab = rebuild_vocab
        self._initialize()

    def _build_model(self):

        # vocab_size = len(self.dataset_help.char2id),
        # num_tags = len(self.dataset_help.label2id),
        # d_model = 128, rnn_type = 'gru', rnn_layers = 1,
        # max_seq_len = self.max_seq_len

        self._model = MODELS[self.model_name](**self.model_config)
        ckpt = tf.train.Checkpoint(model=self._model)
        if not self.rebuild_model:
            ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir=self.save_dir))
            logger.debug('=== Model restored ===')
        self._ckpt_manager = tf.train.CheckpointManager(ckpt, directory=self.save_dir,
                                                        checkpoint_name=self.model_ckpt_name,
                                                        max_to_keep=3)

        model_config_fp = "{}/{}".format(self.save_dir, self.model_conf_file)
        self._model.save_config(model_config_fp)
        logger.debug('Model Config has been saved at {}'.format(model_config_fp))
        logger.debug('Model loaded successfully!')

    def _initialize(self):

        train_ds_fp = "{}/{}".format(self.data_dir, self.train_file)

        # 1. build dataset
        self.dataset_help = Dataset(max_seq_len=self.max_seq_len, pad_idx=self._PAD_IDX, unk_idx=self._UNK_IDX)
        # 1.1. build vocab
        if self.rebuild_vocab:
            self.dataset_help.fit_vocab(data_file=train_ds_fp)
        else:
            self.dataset_help.load_vocab_file(vocab_file="{}/{}".format(self.save_dir, self.vocab_file))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        vocab_save_fp = "{}/{}".format(self.save_dir, self.vocab_file)

        self.dataset_help.save_vocab(file_path=vocab_save_fp)

        # 1.2. build dataset
        self._train_ds = self.dataset_help.build_dataset(data_file=train_ds_fp)
        if self.dev_file:
            self._dev_ds = self.dataset_help.build_dataset(data_file="{}/{}".format(self.data_dir, self.dev_file))

        # 2. build model
        # 模型中需要包含vocab_size、vocab_size两个参数名, 构建完词典会更新这两个参数
        self.model_config['vocab_size'] = len(self.dataset_help.char2id)
        self.model_config['num_tags'] = len(self.dataset_help.label2id)
        self._build_model()

        # 3. build criterion crf自带loss
        # self._criterion = keras.losses.CategoricalCrossentropy()
        # 4. build optimizer
        self._optimizer = keras.optimizers.Adam(learning_rate=self.lr)

    def train_step(self, batch_ds):
        # train
        with tf.GradientTape() as tape:
            batch_pred_sequence, log_likelihood = self._model(batch_ds, training=True)
            loss = -tf.reduce_sum(log_likelihood)
        grads = tape.gradient(loss, self._model.trainable_variables)

        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        acc_num, label_sum = 0, 0
        for pred_y, true_y, length in zip(batch_pred_sequence, batch_ds['y'], batch_ds['seq_len']):
            acc_num += accuracy_score(true_y[:length], pred_y[:length], normalize=False)
            label_sum += length
        acc = acc_num / label_sum * 100
        return loss.numpy(), acc, acc_num, label_sum

    def train(self, epochs, batch_size):

        max_eval_acc = 0
        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_acc_num, epoch_sum = 0, 0, 0
            logger.debug('====== Epoch Start ======')
            for step, batch_data in enumerate(self._train_ds.batch(batch_size)):
                loss, acc, acc_num, label_sum = self.train_step(batch_data)
                epoch_loss += loss
                epoch_acc_num += acc_num
                epoch_sum += label_sum
                if step % 10 == 0:
                    logger.debug(
                        '[Training] Epoch:{}, Step:{}, Step loss:{}, Step Accuracy:{:2f}%'.format(epoch, step, loss,
                                                                                                  acc))
            logger.debug('=' * 6, 'Epoch End', '=' * 6, )
            logger.debug('[Training] Epoch:{}, Epoch loss:{}, Epoch Accuracy:{:2f}%'.format(epoch,
                                                                                            epoch_loss,
                                                                                            epoch_acc_num / epoch_sum * 100))
            eval_loss, eval_acc = self.evaluate(epoch, batch_size)
            if eval_acc > max_eval_acc:
                max_eval_acc = eval_acc
                if epoch > 5:
                    self._ckpt_manager.save()
                    logger.debug('Best model has been Saved..')

    def evaluate(self, epoch, batch_size):
        loss = 0
        acc_num, label_sum = 0, 0
        for step, batch_ds in enumerate(self._dev_ds.batch(batch_size)):
            batch_pred_sequence, log_likelihood = self._model(batch_ds, training=True)
            loss += -tf.reduce_sum(log_likelihood).numpy()
            for pred_y, true_y, length in zip(batch_pred_sequence, batch_ds['y'], batch_ds['seq_len']):
                acc_num += accuracy_score(true_y[:length], pred_y[:length], normalize=False)
                label_sum += length
        acc = acc_num / label_sum * 100
        logger.debug('[Validation] Epoch:{}, Epoch Loss:{}, Epoch Acc:{:2f}%'.format(epoch, loss, acc))
        return loss, acc
#
#
# trainer = SequenceTaggingTrainer(rebuild_vocab=True,
#                                  rebuild_model=True, )
# trainer.train(epochs=10, batch_size=128)
