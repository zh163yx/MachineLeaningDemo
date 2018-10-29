#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 9:11
# @Author  : zh
# @info    : 手写图片训练与预测
# @File    : Digitpredict.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def full_connect():
    """
    建立全连接层预测手写数字
    :return:
    """
    flg = False
    # 1.加载从minist下载的手写数据
    minist_data = input_data.read_data_sets("data/minist", one_hot=True)

    # 2. 建立占位符 在session中运行时可以动态赋值
    # 作用域用于可视化时清晰观看
    with tf.variable_scope("data"):
        # None占的是样本集的个数,表示有未知个32*32 的样本
        x = tf.placeholder(tf.float32, shape=[None, 784])
        # 有未知个标签,每个标签有十个数据,使用one_hot编码表示0-9
        y_true = tf.placeholder(tf.int32, shape=[None, 10])
    # 3.建立全连接模型->使用线性回归方式,权重加偏置
    with tf.variable_scope("model"):
        # 第一次随机权重与偏置,注意权重与偏置需要通过梯度下降来进行调整,所以应为tf中变量类型
        weigth = tf.Variable(tf.random_normal(shape=[784, 10], mean=0.0, stddev=1.0), name="w")
        bias = tf.Variable(tf.constant(0.0), name="b")
        # 预测值=特征*权重+偏置 使用矩阵运算
        y_pre = tf.matmul(x, weigth) + bias
    # 4.建立模型进行预测后->求出损失 损失使用softmax求出损失值后在通过reduce_mean求出平均值
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pre))
    # 5. 使用梯度下降方式进行优化
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss=loss)
    # 6.计算准确率
    with tf.variable_scope("acc"):
        # tf.equal([0,1],[0,0]) 将对应的元素比较是否相等,return[true, false]
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pre, 1))
        # 先将true, false 转换为float32 转换为之后是[0,1]的集合,求出平均值刚好是1的概率
        acc = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    # 7.收集变量进行显示
    # 收集单个变量
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("acc", acc)
    # 收集多维变量
    tf.summary.histogram("weigthes", weigth)
    tf.summary.histogram("bias", bias)
    # 创建初始化变量op
    init_op = tf.global_variables_initializer()
    # 合并收集变量
    merge = tf.summary.merge_all()

    # 创建保存server
    saver = tf.train.Saver()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        # 创建可视化文件
        writer = tf.summary.FileWriter("summary", sess.graph)
        if flg:
            # 循环去训练
            for i in range(2000):
                # 有placeholder占位符需要传入feed_dict
                data_x, data_y = minist_data.train.next_batch(50)
                sess.run(train_op, feed_dict={x: data_x, y_true: data_y})

                # 写入训练数据
                summary = sess.run(merge, feed_dict={x: data_x, y_true: data_y})
                writer.add_summary(summary, i)

                # 打印每次的准确率
                print("训练第%d 步,准确率为:%f" % (i, sess.run(acc, feed_dict={x: data_x, y_true: data_y})))
            # 保存模型
            saver.save(sess, "ckpt/full_con")
        else:
            # 加载模型
            saver.restore(sess, "ckpt/full_con")
            # 进行预测
            for i in range(10):
                pre_x, pre_y = minist_data.train.next_batch(1)
                print("测试第%d张图片,真实为%d,预测为%d" % (i, tf.argmax(pre_y, 1).eval(), tf.argmax(sess.run(
                    y_pre, feed_dict={x: pre_x, y_true: pre_y}
                ), 1).eval()))


def model():
    """
    返回卷积模型
    :return:
    """
    # 数据占位符
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.int32, [None, 10])

    # 第一层卷积
    with tf.variable_scope("cv1"):
        # 随机权重与偏置
        # filter 5*5*1 32个
        weigth1 = tf.Variable(tf.random_normal([5, 5, 1, 32], mean=0.0, stddev=1.0))
        bias1 = tf.Variable(tf.constant(0.0, shape=[32]))

        # 卷积
        # 卷积需要对输入改变形状[None, 784] -> [-1, 28, 28, 1]
        x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1])
        # 经过卷积后[-1,28,28,1] -> [-1,28,28,32]
        cv1 = tf.nn.conv2d(x_reshape, weigth1, strides=[1, 1, 1, 1], padding="SAME")+bias1
        # 激活
        relu1 = tf.nn.relu(cv1)
        # 池化 [-1,28,28,32] -> [-1,14,14,32]
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("cv2"):
        # 随机权重[5,5,32,64]与偏置
        weight2 = tf.Variable(tf.random_normal([5, 5, 32, 64], mean=0.0, stddev=1.0))
        bias2 = tf.Variable(tf.constant(0.0, shape=[64]))
        # 卷积 卷积之后[-1,14,14,32] -> [-1,14,14,64]
        cv2 = tf.nn.conv2d(pool1, weight2, strides=[1, 1, 1, 1], padding="SAME") + bias2
        # 激活
        relu2 = tf.nn.relu(cv2)
        # 池化[-1,14,14,64] -> [-1,7,7,64]
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 全连接层
    with tf.variable_scope("full"):
        weight_full = tf.Variable(tf.random_normal([7*7*64, 10], mean=0.0, stddev=1.0))
        bias2_full = tf.Variable(tf.constant(0.0, shape=[10]))
        # 修改卷积后的形状
        x_full = tf.reshape(pool2, shape=[-1, 7*7*64])

        # 进行预测
        y_pre_full = tf.matmul(x_full, weight_full) + bias2_full
    return x, y, y_pre_full


def con_full():
    """
    卷积
    :return:
    """
    # 加载数据
    minist_data = input_data.read_data_sets("data/minist", one_hot=True)
    # 构建模型
    x, y_true, y_pre = model()
    # 交叉熵损失
    with tf.variable_scope("loss"):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pre)
        loss_mean = tf.reduce_mean(loss)

    # 梯度下降
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mean)
    # 计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pre, 1))
        # equal_list  None个样本   [1, 0, 1, 0, 1, 1,..........]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 变量初始化
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(1000):
            x_mi, y_mi = minist_data.train.next_batch(50)
            print(x_mi)
            print(y_mi)
            sess.run(train_op, feed_dict={x: x_mi, y_true: y_mi})
            print("第%d次训练:准确率:%f" % (
                i, sess.run(accuracy, feed_dict={x: x_mi, y_true: y_mi})
            ))
        saver.save(sess, "ckpt/con_full")


if __name__ == "__main__":
    # full_connect()
    con_full()
