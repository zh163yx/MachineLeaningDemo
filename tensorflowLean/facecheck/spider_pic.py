#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 16:10
# @Author  : zh
# @info     : 从百度图片上下载图片
# @File    : spider_pic.py
# @Software: PyCharm
import requests
import json
import random
import cv2
from threading import Thread
import os
import time


def spider_start(uri, path):
    """
    开始爬取
    :return:
    """
    ua_list = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Mozilla/5.0 (Windows NT 6.1; rv2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
        "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome"
        "/17.0.963.56 Safari/535.11"
    ]
    index = 0
    while index < 600:
        try:
            url = uri.format(index)
            headers = {"User-Agent": ua_list[random.randint(0, 4)]}
            response = requests.get(url, headers=headers)
            jsondata = json.loads(response.text)
            imgurl = []
            for ix in jsondata["data"]:
                if "middleURL" in ix:
                    imgurl.append(ix["middleURL"])
            for iy in range(len(imgurl)):
                # print(imgurl[iy])
                headers = {"User-Agent": ua_list[random.randint(0, 4)],
                               "Cookie": "winWH=%5E6_1920x938; BDIMGISLOGIN=0; BDqhfp=%E6%98%8E%E6%98%9F%E5%9B%BE%E7%"
                               "89%87%E5%A4%A7%E5%85%A8%E8%B5%B5%E4%B8%BD%E9%A2%96%26%260-10-1undefined%26%26"
                               "6622%26%2615; BAIDUID=6C8A8D69B0A7C629E1A69D511E0745E6:FG=1; BIDUPSID=6C8A8D"
                               "69B0A7C629E1A69D511E0745E6; PSTM=1537597682; BDUSS=hFRW5OZThNdXZzaFhKU2pFQjc"
                               "zSHcxRnVBfmh2RWR5czJkTEl4VE13VnlSZVpiQVFBQUFBJCQAAAAAAAAAAAEAAACEwhRSYnWwr"
                               "rPUsugAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                               "AAAAAHK4vltyuL5bZU; Hm_lvt_737dbb498415dd39d8abf5bc2404b290=1540455935; BDRC"
                               "VFR[feWj1Vr5u3D]=I67x6TjHwwYf0; delPer=0; PSINO=2; BDRCVFR[dG2JNJb_ajR]=mk3SLV"
                               "N4HKm; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; indexPageSugList=%5B%22%E6%98%8E%E6%"
                               "98%9F%E5%9B%BE%E7%89%87%E5%A4%A7%E5%85%A8%E8%B5%B5%E4%B8%BD%E9%A2%96%22%2C%22%E6"
                               "%9D%A8%E9%A2%96%22%2C%22%E6%98%8E%E6%98%9F%22%2C%22%E4%BD%95%E7%82%85%E5%86%99%E"
                               "7%9C%9F%22%2C%22%E5%88%98%E4%BA%A6%E8%8F%B2%E5%86%99%E7%9C%9F%22%2C%22%E8%8C%83%E"
                               "5%86%B0%E5%86%B0%E5%86%99%E7%9C%9F%22%2C%22%E8%8C%83%E5%86%B0%E5%86%B0%E8%87%AA%E"
                               "6%8B%8D%22%5D; cleanHistoryStatus=0; H_PS_PSSID=1424_27213_21107"
                                  }
                img_res = requests.get(imgurl[iy], headers=headers)
                imgdata = img_res.content
                print(iy + index)
                with open("img/{0}/{1}.jpg".format(path,iy + index), "wb") as f:
                    f.write(imgdata)
        except Exception as e:
            print(e)
        index += 30


def deal_pic(path):
    """
    处理图片
    :param path:
    :return:
    """
    i = 0
    filelist = os.listdir("img/"+path)
    os.mkdir("img/{0}/deal".format(path))
    face_cascade = cv2.CascadeClassifier(r'D:\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
    for file in filelist:
        try:
            time.sleep(.3)
            print(i)
            imgfile = os.path.join("img/"+path, file)
            frameGray = cv2.imread(imgfile, 0)
            # frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
            faces = face_cascade.detectMultiScale(
                frameGray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(2, 2),
            )
            if len(faces) > 0:
                x, y, w, h = faces[0]
                img = frameGray[y:h + y, x:x + w]
                cv2.imwrite("img/{0}/deal/{1}".format(path, file), img)
            i += 1
        except Exception as e:
            print(e)


def re_name(path,i):
    count = 1
    for file in os.listdir("img/{0}/deal".format(path)):
        os.rename(os.path.join("img/{0}/deal".format(path), file), os.path.join("img/{0}/deal".format(path), i+str(count)+".jpg"))
        count += 1


if __name__ == "__main__":
    # t = Thread(target=spider_start, args=(
    #    "http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord+=&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&word=%E7%8E%8B%E5%8A%9B%E5%AE%8F&z=&ic=0&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&step_word=%E7%8E%8B%E5%8A%9B%E5%AE%8F&pn={0}&rn=30&gsm=1e&1540546651296=",
    #    "wlh"
    # ))
    # t.start()
    # tt = Thread(target=spider_start, args=(
    #    "http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%E6%9D%A8%E7%B4%AB&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word=%E6%9D%A8%E7%B4%AB&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&pn={0}&rn=30&gsm=3000000001e&1540546728168=",
    #    "yz"
    # ))
    # tt.start()
    # zjl = Thread(target=spider_start, args=(
    #    "http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%E8%BF%AA%E4%B8%BD%E7%83%AD%E5%B7%B4&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word=%E8%BF%AA%E4%B8%BD%E7%83%AD%E5%B7%B4&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&pn={0}&rn=30&gsm=5a&1540545040382=",
    #    "dlrb"
    # ))
    # zjl.start()
    # t.join()
    # tt.join()
    # zjl.join()
    # dlrb = Thread(target=deal_pic, args=("wlh",))
    # dlrb.start()
    # gxt = Thread(target=deal_pic, args=("yz",))
    # gxt.start()
    # zj = Thread(target=deal_pic, args=("zj",))
    # zj.start()
    # zjl = Thread(target=deal_pic, args=("zjl",))
    # zjl.start()
    # zly = Thread(target=deal_pic, args=("zly",))
    # zly.start()
    # zs = Thread(target=deal_pic, args=("zs",))
    # zs.start()
    # dlrb.join()
    # gxt.join()
    # zj.join()
    # zjl.join()
    # zly.join()
    # zs.join()
    re_name("zs", "9_")
