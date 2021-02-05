#!/usr/bin/env python
# encoding: utf-8
"""
    @File: predictor.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
import abc


class BasePredictor(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass


