#!/usr/bin/env python
# encoding: utf-8
"""
    @File: trainer.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
import abc


class BaseTrainer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
