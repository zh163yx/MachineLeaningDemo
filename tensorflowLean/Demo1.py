#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/10 13:03
# @Author  : zh
# @Site    : 
# @File    : Demo1.py
# @Software: PyCharm
from tensorflow.contrib import learn

def main():
    iris = learn.datasets.load_dataset('iris')