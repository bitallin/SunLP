#!/usr/bin/env python
# encoding: utf-8
"""
    @File: model.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
import abc
import abc


class BaseModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass