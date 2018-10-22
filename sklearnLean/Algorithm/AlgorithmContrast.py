#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 10:31
# @Author  : zh
# @info     :各类算法比较演示
# @File    : AlgorithmContrast.py
# @Software: PyCharm
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import jieba
# 导入标准化特征工程
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
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
    # 导入训练集
    # news = fetch_20newsgroups(subset='all')
    news = pd.read_csv(r"D:\WorkSpace\Python\SpiderDemo\newsData.csv")
    dataO = news.iloc[:, 0]
    target = news.iloc[:, 1]
    data = []
    # 分词
    for i in range(len(dataO)):
        print("*"*80)
        s = dataO[i]
        s = s.replace(" ", "").replace('\r', '').replace('\u3000', '').replace('\u200b', '')
        if s.find("window.__SINAFLASHURL__") >= 0:
            target.pop(i)
            continue
        if s.find("conten") >= 0:
            target.pop(i)
            continue
        if s.find("不知道第几波的Evo") >= 0:
            target.pop(i)
            continue
        s_array = jieba.cut(s)
        s = " ".join(list(s_array))
        data.append(s)
    print(len(data))
    print(len(target))
    # 划分测试集与训练集
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
    # 特征工程(特征抽取TF*IDF)
    # 使用TF*IDF
    tf = TfidfVectorizer()
    print(x_train)
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)

    # 朴素贝叶斯算法alpha拉普拉斯平滑防止概率为0
    mlt = MultinomialNB(alpha=1)
    mlt.fit(x_train, y_train)
    # 预测
    y_predict = mlt.predict(x_test)
    # 模型保存
    joblib.dump(mlt, 'model.pkl')
    # 保存矢量器
    joblib.dump(tf, 'tf.pkl')
    # 得出准确率
    print("准确率为：", mlt.score(x_test, y_test))


def naviePre():
    model = joblib.load('model.pkl')
    s ="上港回应土超豪门求购武磊：他自己都不知道这2018.10.18 09:38:07《足球》报 本周末联赛重启，接下来的三周时间" \
       "，联赛再无间歇期，冠军与降级的悬念将会在此时间段全部解开。上港与恒大关于中超冠军奖杯的争夺战，到了最后的冲" \
       "刺阶段。 “要把最好的状态带到最后5场比赛” 记者陈伟报道 对于上港而言，接下来的3场比赛极为重要，因为对手相对较" \
       "强，他们要先后碰到苏宁、鲁能、恒大，前两支都是前5球队，而恒大更是争冠的主要对手。 本周日，上港客场对阵苏宁。本" \
       "赛季首回合交手，上港主场击败苏宁，此次客战，他们当然希望再次击败苏宁，取得关键的3分。但上港面临不少困难，主" \
       "要来自于人员——主帅佩雷拉、核心浩克、防线大将贺惯均不能出战。这还不止，于海在国家队比赛里受伤，也会缺席。 " \
       "国字号抽调人员影响大 在这个国家队比赛日，上港被各级国字号抽掉的球员有10位，武磊、颜骏凌、于海在国家队，" \
       "陈威、李申圆、陈彬彬和俞豪在荷兰参加国奥队的集训，雷文杰和林创益在国家集训队，外援艾哈迈多夫代表乌" \
       "兹别克斯坦国家队比赛。为了让训练可以正常进行，教练组临时抽调了几个年轻球员加入平时的训练。 打完贵" \
       "州的比赛，上港放了三天假。11日，球队集中，备战最后五轮联赛。上港最后五轮安排是客场对阵苏宁，" \
       "主场对阵鲁能，客场对阵恒大，主场对阵人和，客场对阵权健。 联赛重启之战，对于上港能否保住积分" \
       "领先的优势，以更好的心态迎接未来的比赛，至关重要，因此，上港极为重视。“回来之后的备战前" \
       "两天是恢复性训练，这是让我们开始回归到正常的备战节奏，随后教练组开始在训练中强调对抗，因为后面几场比赛之间衔" \
       "接得比较紧，教练组强调我们需要在体能储备上要有准备。”蔡慧康说。 最近一段时间，上港的状态不错，上一" \
       "轮拿下贵州后，取得了主场8连胜，不过球队也没有因此满足，“现在队里气氛挺好，我们都向往着最好的结果，要把最" \
       "好的状态带到最后5场比赛。”队员们都希望能够一直延续胜利的势头。 伤停不少是难题 主场对阵贵州队的比赛，" \
       "上港虽然取得了一场大胜，但是付出的代价也是相当大。主教练佩雷拉被罚上了看台，浩克和贺惯在比赛中都吃到了" \
       "黄牌，对阵苏宁的比赛都因为累积四张黄牌停赛无缘参加。 屋漏偏逢连夜雨，于海在国家队和叙利亚的比赛受伤下场，" \
       "比赛第32分钟，于海与对方球员在防守中相撞，随后被换下场，并被送往医院进行检查，直到晚上快12点的时候，进行了" \
       "核磁共振检查的于海才从医院返回。 初步检查的结果是膝盖处有积液，初步诊断为左膝前交叉韧带有三分之一的地方断裂" \
       "，同时内侧副韧带有损伤。为了进一步确诊，于海在当晚就返回了上海。 于海伤势需要一段时间的休息，联赛余下的赛事" \
       "将会缺席，并可能影响其代表国家队参加亚洲杯。 17日，上港俱乐部发表公告，对于于海受伤进行了情况说明，表示球员" \
       "能够为国出战是无限光荣，俱乐部为于海的拼搏精神称赞，深受感动。将会时刻跟进并关注其伤情，为其提供最好的治疗" \
       "和恢复条件。 对阵苏宁，上港在进攻线少了浩克，防守端更是少了贺惯与于海两位绝对主力，佩雷拉该如何调整？ 从目前" \
       "的情况来看，对阵苏宁的比赛，防线“万金油”王燊超将会内收和石柯搭档中后卫，傅欢和张卫将会出任两边边后卫。浩克" \
       "无法上场，艾哈迈多夫、奥斯卡和埃尔克森将会成为首发三外援。尽管无法参加比赛，但是浩克在这个间歇期的投入度依" \
       "然相当高，按照他的说法，“要以最好的状态来帮助球队在下一轮击败鲁能”。 在上港，浩克是毫无疑问的“带头大哥”" \
       "，在夺冠的关键时刻，他以身作则：“我还是会在场外继续帮助队友，比如在训练场上我会提醒一下大家，在场下和大家" \
       "说一说。其实不光是我，我们每一个首发球员，每一个替补球员，包括没有进名单的球员，都会用自己的方法去帮助球队，" \
       "大家团结一致踢好比赛。” 至于主教练不能现场指挥的影响，浩克表示：“影响会有，但我们经过一个赛季的磨合，每个" \
       "人都已经知道该怎么去踢属于自己风格的比赛，每个人在自己的位置上都知道自己要做什么事情。我们很有信心在下一场" \
       "比赛中取得胜利。” 另外关于上港还有些传闻，比如队内的队员被国外球队看中。最新的传闻是土媒的报道，声称土超" \
       "豪门加拉塔萨雷有意引进武磊，希望打开中国的市场。但此前已经在不少转会传闻中与国外球队“产生过联系”的武磊已经" \
       "适应这样的节奏，按照上港俱乐部人士的说法是，“他自己都不知道这件事。”而在此冲冠关键时刻，俱乐部也不会在这" \
       "个阶段考虑任何人员流动的事宜。点击进入专题： 中超联赛"
    s_array = jieba.cut(s)
    s = " ".join(list(s_array))
    # 分词后进行特征抽取
    df = joblib.load('tf.pkl')
    ss = df.transform([s])
    pre = model.predict(ss)
    print(pre)


def logstic():
    """
    使用逻辑回归二分法
    :return: None
    """
    # 构造列标签名
    col_name = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
                "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
                "Mitoses", "Class"]
    # 读取数据
    data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/"
                       "breast-cancer-wisconsin.data", names=col_name)
    # 缺失值处理
    data = data.replace(to_replace='?', value=np.nan)
    # 删除缺失值
    data = data.dropna()

    x_train, x_test, y_train, y_test = train_test_split(data[col_name[1:10]], data[col_name[10]], test_size=0.25)
    # 进行标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 进行训练
    lr = LogisticRegression(C=1.0)
    lr.fit(x_train, y_train)
    print(lr.coef_)
    print(lr.score(x_test, y_test))
    # 召回率
    y_pre = lr.predict(x_test)
    print("召回率", classification_report(y_test, y_pre, labels=[2, 4], target_names=["良性", "恶性"]))

    return None


if __name__ == "__main__":
    # knncls()
    # naviedayes()
    # naviePre()
    logstic()
