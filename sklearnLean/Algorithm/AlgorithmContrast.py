#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 10:31
# @Author  : zh
# @info     :各类算法比较演示
# @File    : AlgorithmContrast.py
# @Software: PyCharm
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
# 导入新闻数据集
from sklearn.datasets import fetch_20newsgroups
# 导入标准化特征工程
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def knncls():
    """
    K-近邻算法
    :return:
    """
    # 读取数据文件
    data = pd.read_csv('train.csv')
    # print(data.head(50))
    # 处理数据
    # 1.缩小数据根据x,y
    data = data.query("x>1.0 & x < 1.5 & y > 2.5 & y < 3.")
    # 处理时间 将time列以秒为单位转换为datetime类型
    time_value = pd.to_datetime(arg=data['time'], unit='s')
    # print(time_value)
    # 转换为字典格式
    time_dic = pd.DatetimeIndex(time_value)
    # print(time_dic)
    # 添加一些特征
    data['day'] = time_dic.day
    data['hour'] = time_dic.hour
    data['weekday'] = time_dic.weekday
    # 剔除一些特征 axis =1 按列剔除
    data = data.drop(['time'], axis=1)
    # 把签到少于n个的剔除
    # groupby().count()之后除了place_id所有列的值都变成了count的数量
    place_count = data.groupby('place_id').count()
    # print(place_count)
    # reset_index还原索引其实就是在前面加上整型索引
    tf = place_count[place_count.row_id > 3].reset_index()
    # print(tf)
    data = data[data['place_id'].isin(tf.place_id)]
    y = data['place_id']
    x = data.drop(['place_id', 'row_id'], axis=1)
    # 划分测试集与训练集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程(进行标准化)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 使用K-近邻算法
    knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # print(knn.score(x_test, y_test))

    # 网格搜索与交叉验证
    # 构造参数
    para = {'n_neighbors': [3, 5, 7, 10]}
    gc = GridSearchCV(knn, param_grid=para, cv=2)
    gc.fit(x_train, y_train)
    print("在测试集上准确率：", gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果：", gc.best_score_)
    print("选择最好的模型是：", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果：", gc.cv_results_)


def naviedayes():
    """
    使用朴素贝叶斯对文本进行分类
    :return: None
    """

    # 从sklean.data中导入训练集
    news = fetch_20newsgroups(subset='all')

    x_train, x_test, y_train, y_test= train_test_split(news.data, news.target,test_size=0.25)
    # 特征工程(特征抽取TF*IDF)
    tf = TfidfTransformer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)

    # 朴素贝叶斯算法alpha拉普拉斯平滑防止概率为0
    mlt = MultinomialNB(alpha=1)
    mlt.fit(x_train, y_train)
    # 预测
    y_predict = mlt.predict(x_test)
    # 得出准确率
    print("准确率为：", mlt.score(x_test, y_test))
    print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))
    pass


if __name__ == "__main__":
    knncls()
