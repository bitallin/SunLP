#!/usr/bin/env python
# encoding: utf-8
"""
    @File: text_clear.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
import re

from sunlp.preprocess.data.langconv import *


class TextClear:
    def __init__(self):
        # 加载正则编译器
        """
            如果不使用re.S参数，则只在每一行内进行匹配，如果一行没有，就换下一行重新开始。
            而使用re.S参数以后，正则表达式会将这个字符串作为一个整体，在整体中进行匹配。
        """
        self.re_que_punc = re.compile(r"[?？]+")
        self.re_cn = re.compile(r'[\u4e00-\u9fa5]')
        self.re_time = re.compile(r"\d+-\d+-\d+\s+\d+:\d+")
        self.re_punc = re.compile(r"[!'""()-_=+%【】😁😂·！、⑤③。:：“”《》（）*~|丨@#￥…&—，；‘,.[]]+")
        self.re_eng = re.compile(r'[a-zA-Z]+')
        self.re_float = re.compile(r'\d+.\d+')
        self.re_digit = re.compile(r'\d+')
        self.re_http = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.re_forward = re.compile("@.*?: ", re.S)
        self.re_emoji = re.compile(r"\[.*?]", re.S)
        self.re_at = re.compile("@.*? ", re.S)
        self.re_yuanweibo = re.compile(r"【原微博】", re.S)

    def news_title_clear(self, sentence: str):
        """

        Args:
            sentence: str
        Returns:
            str
        """
        sentence = self.tradition_to_simple(sentence)  # 繁体 => 简体
        sentence = self.delete_time(sentence)  # 删除时间
        sentence = self.replace_eng(sentence)  # 英文 => <eng>
        sentence = self.replace_que_punc(sentence)  # ？ => <que>
        sentence = self.replace_dig_float(sentence)  # 数字 => <digit>, 浮点数 => <float>
        sentence = self.replace_punc(sentence)  # 标点 => <punc>
        return sentence

    def clear_for_weibo(self, sentence):
        sentence = self.tradition_to_simple(sentence)  # 繁体 => 简体
        sentence = self.delete_none(sentence)
        sentence = self.delete_yuanweibo(sentence)
        sentence = self.delete_zhuanfa(sentence)  # 删除转发
        sentence = self.delete_time(sentence)  # 删除时间
        sentence = self.delete_forward(sentence)
        sentence = self.delete_at(sentence)
        sentence = self.delete_http(sentence)  # del http
        sentence = self.delete_emoji(sentence)
        sentence = self.replace_que_punc(sentence)  # ？ => <que>
        sentence = self.replace_dig_float(sentence)  # 数字 => <digit>, 浮点数 => <float>
        sentence = self.replace_punc(sentence)  # 标点 => <punc>
        return sentence

    def extract_cn(self, sentence):
        res = re.findall(self.re_cn, sentence)
        return ''.join(res)

    @staticmethod
    def delete_none(sentence: str):
        """删除空白,空格"""
        return sentence.replace(' ', '').replace('</br>', '').replace('\u200b', '')

    def replace_punc(self, sentence):
        sentence = re.sub(self.re_punc, '<punc>', sentence)
        return sentence

    def replace_dig_float(self, sentence):
        sentence = re.sub(self.re_float, '<float>', sentence)
        sentence = re.sub(self.re_digit, '<digit>', sentence)
        return sentence

    def replace_eng(self, sentence):
        sentence = re.sub(self.re_eng, '<eng>', sentence)
        return sentence

    def delete_time(self, sentence):
        sentence = re.sub(self.re_time, "", sentence)
        return sentence

    @staticmethod
    def delete_bracket(sentence):
        sentence = re.sub("</br>", "", sentence)
        return sentence

    @staticmethod
    def delete_zhuanfa(sentence):
        sentence = re.sub(r"转发微博", "", sentence)
        return sentence

    def delete_emoji(self, sentence):
        sentence = re.sub(self.re_emoji, "", sentence)
        return sentence

    def delete_at(self, sentence):
        sentence = re.sub(self.re_at, "", sentence)
        return sentence

    def delete_yuanweibo(self, sentence):
        sentence = re.sub(self.re_yuanweibo, "", sentence)
        return sentence

    def delete_forward(self, sentence):
        sentence = re.sub(self.re_forward, "", sentence)
        return sentence

    def delete_http(self, sentence):
        sentence = re.sub(self.re_http, "", sentence)
        # sentence = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', sentence)
        return sentence

    def replace_que_punc(self, sentence):
        sentence = re.sub(self.re_que_punc, "<que>", sentence)
        return sentence

    @staticmethod
    def remove_stopwords(sentences, stop_word_file: str):
        """
        Args:
            sentences: [ [w1, w2,...], [w1, w2,...]]
            stop_word_file: file path
        Returns:
            list: [ [w1, w2,...], [w1, w2,...]]
        """
        _sentences = []
        stop_words = [word.strip() for word in open(stop_word_file, 'r', encoding='utf8').readlines()]
        for sentence in sentences:
            _sentence = [word for word in sentence if word not in stop_words]
            _sentences.append(_sentence)
        return _sentences

    @staticmethod
    def tradition_to_simple(text: str):
        """繁体转简体"""
        text = Converter('zh-hans').convert(text)
        return text

    @staticmethod
    def words_deduplication(words: list):
        """
            词去重,取长度最长的
        Args:
            words: ['中华民族伟大', '民族伟大']
        Returns:
            list: ['中华民族伟大']
        """

        words.sort(key=lambda x: len(x), reverse=True)
        rtn_word_list = [words[0]]
        for word in words[1:]:
            word_del_flag = False
            for _word in rtn_word_list:
                if word in _word:
                    word_del_flag = True
                    break
            if not word_del_flag:
                rtn_word_list.append(word)
        return rtn_word_list
