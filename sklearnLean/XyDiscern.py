#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/29 9:42
# @Author  : zh
# @Site    : 
# @File    : One.py
# @Software: PyCharm
import sqlite3
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier
import pickle
def train():
    features=[]
    lables = []
    conn = sqlite3.connect("sqlite.db")#链接sqlite3数据库
    cur = conn.cursor()
    rows =cur.execute("select X,Y,flg from Coordinates")
    #将数据读入特征与标签列表
    for row in rows:
        features.append([row[0], row[1]])
        lables.append(row[2])
    #随机划分数据
    x_train,x_text,y_train,y_test = sklearn.model_selection.train_test_split(features,lables,test_size=0.2)
    #使用K值邻近算法
    clf = KNeighborsClassifier()
    #训练
    clf.fit(x_train,y_train)
    #存储模型
    fw = open("xy","wb")
    pickle.dump(clf,fw)
    fw.close()
    # 测试
    pre = clf.predict(x_text)
    # 输出测试结果准确率
    print(sklearn.metrics.accuracy_score(y_test, pre, normalize=True))


def load():
    # 加载模型
    fr = open("xy", "rb")
    flt = pickle.load(fr)
    fr.close()
    while 1:
        instr = input("请输入坐标以逗号隔开(q退出)\n")
        if instr == 'q':
            return
        strss = instr.split(',')
        if len(strss) != 2:
            print("请输入正确数据")
            continue
        arr = [[int(i) for i in strss]]
        # 预测
        if flt.predict(arr)[0] == 0:
            print("空测数据")
        if flt.predict(arr)[0] == 1:
            print("安全数据")
        if flt.predict(arr)[0] == -1:
            print("危险数据")
        print(flt.predict(arr))


if __name__ == "__main__":
    load()
    #train()
