#!/usr/bin/env python
# encoding: utf-8
"""
    @File: runner.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
import abc


class BaseRunner(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass
