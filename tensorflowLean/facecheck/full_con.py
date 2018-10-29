#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 21:46
# @Author  : zh
# @info     :
# @File    : full_con.py
# @Software: PyCharm
import tensorflow as tf
import os


# 定义一个初始化权重的函数
def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 定义一个初始化偏置的函数
def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def read_and_decode():
    """
    读取验证码数据API
    :return: image_batch, label_batch
    """
    file_array = os.listdir("img/deal")
    file_list = [os.path.join("img/deal", file) for file in file_array]
    # 1、构建文件队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2、构建阅读器，读取文件内容，默认一个样本
    reader = tf.WholeFileReader()

    # 读取内容
    key, value = reader.read(file_queue)
    # 解码内容，字符串内容
    # 1、先解析图片的特征值
    image = tf.image.decode_jpeg(value)
    # 2. 先解析图片的目标值
    label = key
    # print(key)

    # print(image, label)

    # 改变形状

    image_reshape = tf.image.resize_images(image, size=[20, 80])
    image_reshape.set_shape([20, 80, 1])
    label_reshape = tf.reshape(label, [1])

    # print(image_reshape, label_reshape)

    # 进行批处理,每批次读取的样本数 100, 也就是每次训练时候的样本
    image_batch, label_btach = tf.train.batch([image_reshape, label_reshape], batch_size=20, num_threads=2, capacity=20)

    # print(image_batch, label_btach)
    return image_batch, label_btach


def fc_model(image):
    """
    进行预测结果
    :param image: 100图片特征值[100, 20, 80, 3]
    :return: y_predict预测值[100, 4 * 26]
    """
    with tf.variable_scope("model"):
        # 将图片数据形状转换成二维的形状
        image_reshape = tf.reshape(image, [-1, 20 * 80 * 1])

        # 1、随机初始化权重偏置
        # matrix[100, 20 * 80 * 3] * [20 * 80 * 3, 4 * 26] + [104] = [100, 4 * 26]
        weights = weight_variables([20 * 80 * 1, 10])
        bias = bias_variables([10])

        # 进行全连接层计算[100, 4 * 26]
        y_predict = tf.matmul(tf.cast(image_reshape, tf.float32), weights) + bias
    return y_predict


def predict_to_onehot(label):
    """
    将读取文件当中的目标值转换成one-hot编码
    :param label: [100, 4]      [[13, 25, 15, 15], [19, 23, 20, 16]......]
    :return: one-hot
    """

    # 进行one_hot编码转换，提供给交叉熵损失计算，准确率计算[100, 4, 26]
    label_onehot = tf.one_hot(label, depth=10, on_value=1.0, axis=1)
    return label_onehot


def captcharec():
    """
    验证码识别程序
    :return:
    """
    # 1、读取验证码的数据文件 label_btch [100 ,4]
    image_batch, label_batch = read_and_decode()
    # print(":"*90)
    # 2、通过输入图片特征数据，建立模型，得出预测结果
    # 一层，全连接神经网络进行预测
    # matrix [100, 20 * 80 * 3] * [20 * 80 * 3, 4 * 26] + [104] = [100, 4 * 26]
    y_predict = fc_model(image_batch)

    #  [100, 4 * 26]
    # print(y_predict)

    # 3、先把目标值转换成one-hot编码 [100, 4, 26] 使用占位符
    # y_true = predict_to_onehot(label_batch)
    y_true = tf.placeholder(tf.float32, [None, 10])

    # 4、softmax计算, 交叉熵损失计算
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失 ,y_true [100, 4, 26]--->[100, 4*26]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(y_true, [-1, 10]),
            logits=y_predict))
    # 5、梯度下降优化损失
    with tf.variable_scope("optimizer"):

        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 6、求出样本的每批次预测的准确率是多少 三维比较
    with tf.variable_scope("acc"):

        # 比较每个预测值和目标值是否位置(4)一样    y_predict [100, 4 * 26]---->[100, 4, 26]
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(tf.reshape(y_predict, [20, 10]), 1))

        # equal_list  100个样本   [1, 0, 1, 0, 1, 1,..........]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 开启会话训练
    with tf.Session() as sess:
        sess.run(init_op)
        # labe_on_hot = predict_to_onehot(label_batch.eval())
        # 定义线程协调器和开启线程（有数据在文件当中读取提供给模型）
        coord = tf.train.Coordinator()

        # 开启线程去运行读取文件操作
        threads = tf.train.start_queue_runners(sess, coord=coord)
        label_array = [int(str(label_one).split(r"\\")[1].split("_")[0]) for label_one in label_batch.eval()]
        label_onehot = predict_to_onehot(label_array)
        # 训练识别程序
        for i in range(400):

            sess.run(train_op, feed_dict={y_true: label_onehot.eval()})

            print("第%d批次的准确率为：%f" % (i, sess.run(accuracy, feed_dict={y_true: label_onehot.eval()})))

        # 回收线程
        coord.request_stop()

        coord.join(threads)

    return None


if __name__ == '__main__':
    captcharec()