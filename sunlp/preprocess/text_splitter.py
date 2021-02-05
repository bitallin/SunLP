#!/usr/bin/env python
# encoding: utf-8
"""
    @File: text_splitter.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
import re


class TextSplitter:
    def __init__(self, stops='[，。,.？?！!；;：\n ]'):
        self.stops = stops
        self.re_stop = re.compile(self.stops)

    def split_text(self, text):
        return [x for x in self.re_stop.split(text) if x]
