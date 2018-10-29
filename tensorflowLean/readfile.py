#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 12:49
# @Author  : zh
# @info     :
# @File    : readfile.py
# @Software: PyCharm

import tensorflow as tf
import os


def readcsv(filelist):
    """
    读取csv文件
    :param filelist:
    :return:
    """
    # 创建csv文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 构造阅读器读取csv文件默认一行
    reader = tf.TextLineReader()
    # key 文件名, value一行数据
    key, value = reader.read(file_queue)

    # 对读取的内容进行解码
    # record_defaults 指定数据类型以及默认数据
    record = [["None"], ["None"]]
    exmple, lable = tf.decode_csv(value, record_defaults=record)

    # 批处理 读取数据量主要取决于batch_size
    exmple_batch, lable_batch = tf.train.batch([exmple, lable], batch_size=6, num_threads=1, capacity=6)

    return exmple_batch, lable_batch


def picread(filelist):
    """
    图片读取
    :param filelist:
    :return:
    """
    # 建立文件队列
    file_queue = tf.train.string_input_producer(filelist)
    # 创建图片阅读器
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    print(value)
    # 图片解码
    image = tf.image.decode_jpeg(value)
    print(image)
    # 图片处理改变大小同意格式
    image_resize = tf.image.resize_images(image, size=[200, 200])
    print(image_resize)
    # 三维补齐
    image_resize.set_shape([200, 200, 1])
    # 进行批处理
    image_batch, lable_batch = tf.train.batch([image_resize, key], batch_size=20, num_threads=1, capacity=20)
    return image_batch, lable_batch


def byteread(filelist):
    # 文件队列
    flie_queue = tf.train.string_input_producer(filelist)

    # 文件阅读器
    reader = tf.FixedLengthRecordReader(3073)
    key, value = reader.read(flie_queue)
    # 二进制解码
    img = tf.decode_raw(value, tf.uint8)

    # 切割划分特征与标签
    lable = tf.slice(img, [0], [1])
    image = tf.slice(img, [1], 3073)

    # 特征改变
    image_reshape = tf.reshape(image, [32, 32, 3])

    # 批处理
    image_batch, lable_batch = tf.train.batch([image_reshape, lable], batch_size=20, num_threads=1, capacity=20)

    return image_batch, lable_batch


if __name__ == "__main__":
    filelist = os.listdir("facecheck/img/deal")
    filelist = [os.path.join('facecheck/img/deal', file) for file in filelist]

    # exmple_batch, lable_batch = readcsv(filelist=filelist)
    image_batch, lable_batch = picread(filelist=filelist)
    # 开启会话
    with tf.Session() as sess:
        # 线程协调器
        coord = tf.train.Coordinator()
        # 开启线程进行读取
        thread = tf.train.start_queue_runners(sess, coord=coord)
        # print(sess.run([exmple_batch, lable_batch]))
        print(sess.run(lable_batch))

        # 回收子线程
        coord.request_stop()
        coord.join(thread)
