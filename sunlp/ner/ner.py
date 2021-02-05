#!/usr/bin/env python
# encoding: utf-8
"""
    @File: ner.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
from sunlp.ner.predictor import Predictor
from sunlp.base.runner import BaseRunner


class NER(BaseRunner):
    def __init__(self, model_name='birnn_atten_crf'):
        model_name = model_name.lower()
        self._predictor = Predictor(model_name=model_name)

    def run(self, texts):
        res = self._predictor.predict(texts)
        return res
