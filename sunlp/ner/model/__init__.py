#!/usr/bin/env python
# encoding: utf-8
"""
    @File: __init__.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
from sunlp.ner.model.birnn_atten_crf import BiRnnAttentionCrfModel

MODELS = {
    'birnn_atten_crf': BiRnnAttentionCrfModel
}
