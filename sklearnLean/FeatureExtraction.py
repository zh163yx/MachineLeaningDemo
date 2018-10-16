#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/11 13:45
# @Author  : zh
# @Infor   : 特征抽取
# @File    : FeatureExtraction.py
# @Software: PyCharm
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba


def countvec():
    '''
    数量特征
    :return:
    '''
    cv = CountVectorizer()
    data = cv.fit_transform(["你好啊", "sss lo lk,sdf ji."])
    print(data)
    print("*"*20)
    print(cv.get_feature_names())
    print(data.toarray())


def dicvec():
    '''
    字典特征抽取
    :return:
    '''
    cv = DictVectorizer(sparse= False)
    data = cv.fit_transform([{'city': '上海', 'a': 11},{'city': '北京', 'a': 12}])
    print(cv.get_feature_names())
    print(data)
    print("*"*20)


def tfvec():
    '''
    TFIDF 文本特征重要度
    :return:
    '''
    c1 =jieba.cut("我们将收到输出，包含有关下载内容和要安装的软件包的信息，然后提示您继续执行y或n")
    c2 = jieba.cut("如果您不再处理特​​定项目，并且不再需要相关环境，则可以将其删除。为此，请键入以下内容")
    st1= ' '.join(list(c1))
    st2= ' '.join(list(c2))
    cv = TfidfVectorizer()
    data = cv.fit_transform([st1,st2])
    print(cv.get_feature_names())
    print(data.toarray())


def hanzvec():
    '''
    文本特征抽取,不统计单个字母,若是对中文进行抽取,需要先分词
    :return:
    '''
    c1 =jieba.cut("我们将收到输出，包含有关下载内容和要安装的软件包的信息，然后提示您继续执行y或n")
    c2 = jieba.cut("如果您不再处理特​​定项目，并且不再需要相关环境，则可以将其删除。为此，请键入以下内容")
    st1= ' '.join(list(c1))
    st2= ' '.join(list(c2))
    cv = CountVectorizer()
    data = cv.fit_transform([st1,st2])
    print(cv.get_feature_names())
    print(data.toarray())


if __name__ == "__main__":
    # countvec()
    # dicvec()
    # hanzvec()
    tfvec()
