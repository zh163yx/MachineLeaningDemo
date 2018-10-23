#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 9:50
# @Author  : zh
# @info    : 线性回归练习
# @File    : regression.py
# @Software: PyCharm
import tensorflow as tf
import os
import random

def myregression():
    """
    线性回归训练
    :return:None
    """
    # 准备数据 特征值
    x = tf.random_normal([100, 1], mean=3.75, stddev=0.5, name="data")
    y_true = tf.matmul(x, [[0.7]])+0.8

    # 建立模型 建立权重与偏置(模型中的参数因为需要不断调整所以使用变量表示)
    w = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, name="w"))
    b = tf.Variable(0.0, name="b")

    # 进行预测 使用均方误差求出损失
    y_pre = tf.matmul(x, w)+b
    loss = tf.reduce_mean(tf.square(y_true-y_pre))

    # 梯度下降 进行优化w与b
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # 记录变量
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("w", w)
    merge = tf.summary.merge_all()
    # 定义的变量需要初始化Op
    init_op = tf.global_variables_initializer()

    # 定义保存模型
    saver = tf.train.Saver()

    # 使用会话运行
    with tf.Session() as sess:
        # 初始化变量OP
        sess.run(init_op)
        # 建立事件文件
        filewriter = tf.summary.FileWriter('summary', graph=sess.graph)

        # 判断是否存在保存的模型若有加载模型
        if os.path.exists("ckpt/checkpoint"):
            saver.restore(sess, "ckpt/reg")
        print("随机权重:%f,偏置:%f" % (w.eval(), b.eval()))
        # 执行训练
        for i in range(500):
            sess.run(train_op)
            print("第%d次权重:%f,偏置:%f" % (i, w.eval(), b.eval()))
            summary = sess.run(merge)
            filewriter.add_summary(summary, i)
        saver.save(sess, 'ckpt/reg')


if __name__ == "__main__":
    myregression()
