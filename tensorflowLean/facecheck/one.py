#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/25 16:46
# @Author  : zh
# @info     :
# @File    : one.py
# @Software: PyCharm

import requests
import json
import cv2
import os
import numpy as np
import tensorflow as tf


def getpic(index):
    """
    从百度图片上获取照片
    :return:
    """
    url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=re" \
          "sult&queryWord=%E5%88%98%E4%BA%A6%E8%8F%B2%E5%86%99%E7%9C%9F&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpi" \
          "cid=&st=-1&z=&ic=0&word=%E5%88%98%E4%BA%A6%E8%8F%B2%E5%86%99%E7%9C%9F&s=&se=&tab=&width=&height" \
          "=&face=0&istype=2&qc=&nc=1&fr=&expermode=&cg=star&pn={0}&rn=60&gsm=78&1540459800443=".format(index)
    response = requests.get(url)
    jsondata = json.loads(response.text)
    imgurl = []
    for ix in jsondata["data"]:
        if "middleURL" in ix:
            imgurl.append(ix["middleURL"])
    for iy in range(len(imgurl)):
        # print(imgurl[iy])
        try:
            img_res = requests.get(imgurl[iy])
            imgdata = img_res.content
            print(iy+index)
            with open("img/lyf/{0}.jpg".format(iy+index), "wb") as f:
                f.write(imgdata)
        except:
            print("error")


def getdealpic(imgfile):
    face_cascade = cv2.CascadeClassifier('D:\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
    frame = cv2.imread(imgfile)
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
    faces = face_cascade.detectMultiScale(
        frameGray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(2, 2),
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = frameGray[y:h + y, x:x + w]
        return img
    else:
        return None


def dealpic():
    """
    识别图片中的人脸并将其裁剪出来保存
    :return:
    """
    i = 0
    filelist = os.listdir("img/lyf")
    for file in filelist:
        try:
            print(i)
            imgfile = os.path.join("img/lyf", file)
            img = getdealpic(imgfile)
            cv2.imwrite("img/lyf/deal/" + file, img)
            i += 1
        except:
            print("error")
        # return


def model(img):
    """
    返回卷积模型
    :return:
    """
    # 数据占位符
    with tf.variable_scope("data"):
        x = tf.reshape(img, [-1, 28*28*1])
        print(x)
        # x = tf.placeholder(tf.float32, [None, 784])
        # y = tf.placeholder(tf.int32, [None, 2])

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
        print(x_full)
        # 进行预测
        y_pre_full = tf.matmul(x_full, weight_full) + bias2_full
    return y_pre_full


def read_img():
    """
    读取img
    :return:img_batch, label_batch
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

    image_reshape = tf.image.resize_images(image, size=[28, 28])
    image_reshape.set_shape([28, 28, 1])
    label_reshape = tf.reshape(label, [1])
    print(image_reshape)
    # print(image_reshape, label_reshape)

    # 进行批处理,每批次读取的样本数 100, 也就是每次训练时候的样本
    image_batch, label_btach = tf.train.batch([image_reshape, label_reshape], batch_size=20, num_threads=2, capacity=20)

    # print(image_batch, label_btach)
    return image_batch, label_btach


def predict_to_onehot(label):
    """
    将读取文件当中的目标值转换成one-hot编码
    :param label: [100, 4]      [[13, 25, 15, 15], [19, 23, 20, 16]......]
    :return: one-hot
    """

    # 进行one_hot编码转换，提供给交叉熵损失计算，准确率计算[100, 4, 26]
    label_onehot = tf.one_hot(label, depth=10, on_value=1.0, axis=1)
    return label_onehot


def con_full():
    """
    卷积
    :return:
    """
    # 加载数据
    img_batch, label_batch = read_img()
    y_true = tf.placeholder(tf.float32, [None, 10])
    # 构建模型
    print(img_batch)
    y_pre = model(img_batch)
    # 交叉熵损失
    with tf.variable_scope("loss"):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels= y_true, logits=y_pre)
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
        coor = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coor)
        label_array = [int(str(label_one).split(r"\\")[1].split("_")[0]) for label_one in label_batch.eval()]
        label_onehot = predict_to_onehot(label_array)
        for i in range(1010):
            # print(sess.run(key0))
            sess.run(train_op, feed_dict={y_true: label_onehot.eval()})
            print("第%d次训练:准确率:%f" % (
                i, sess.run(accuracy, feed_dict={y_true: label_onehot.eval()})
            ))
        saver.save(sess, "ckpt/mx")
        coor.request_stop()
        coor.join(thread)



def pre():
    """
    预测
    :return:
    """
    img = getdealpic("G://0.jpg")
    if img is None:
        return
    # cv2.imshow("i", img)
    # cv2.waitKey()
    img = tf.reshape(img, shape=[len(img), len(img[0]), 1])
    img_reshape = tf.image.resize_images(img, size=[28, 28])
    print(img_reshape)
    y_pre = model(img_reshape)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "ckpt/mx")
        print(tf.argmax(sess.run(y_pre), 1).eval())


if __name__ == "__main__":
    # con_full()
    pre()
    # getpic(0)
