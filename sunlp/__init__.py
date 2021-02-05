#!/usr/bin/env python
# encoding: utf-8
"""
    @File: __init__.py
    @Desc:
    @Reference: 
    @Author: letter
    @Contact: 5403517@qq.com
"""
import logging, sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
