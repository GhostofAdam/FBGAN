import pandas as pd
import numpy as np
import random
import math
import joblib
import matplotlib.pyplot as plt

# 导入回归模型
# 线性回归模型
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
# 非线性模型
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
# 集成模型
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor

# 导入分类模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# 计算相关系数
def calc_corr(a, b):
    '''
    :param a:
    :param b:
    :return: Pearson correlation coefficient
    '''
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
    corr_factor = cov_ab / sq
    return corr_factor


def exchange(x, y):
    '''
    :param x: x
    :param y: y
    :return: x(=y), y(=x)
    '''
    temp = np.zeros([len(x)], dtype=float)
    for i in range(len(x)):
        temp[i] = x[i]
    for i in range(len(x)):
        x[i] = y[i]
        y[i] = temp[i]
    return x, y


# 分层抽样 随机选取各温度200项=1000项作为训练集100和测试集900
def group_sample(data, label, nums, Is_regre):
    '''
    :param data: features / X
    :param label: label / y
    :param nums: the number of different temperatures
    :param Is_regre: Is or not regression
    :return: Train and Test Dataset
    '''
    label_p = np.zeros([len(label), 1], dtype=int)
    for i in range(len(label)):
        label_p[i] = label[i]
    data_p = np.concatenate((data, label_p), axis=1)
    # 打乱顺序
    for i in reversed(range(1, len(data_p))):
        j = random.sample(range(0, i), 1)
        data_p[i], data_p[j[0]] = exchange(data_p[i], data_p[j[0]])
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    flag = np.zeros([5], dtype=float)
    if Is_regre:
        for i in range(len(data_p)):
            if data_p[i][-1] < 30:
                flag[0] += 1
            elif data_p[i][-1] < 50:
                flag[1] += 1
            elif data_p[i][-1] < 65:
                flag[2] += 1
            elif data_p[i][-1] < 85:
                flag[3] += 1
            else:
                flag[4] += 1
            if (0 <= data_p[i][-1] < 30 and flag[0] <= 20) or \
               (30 <= data_p[i][-1] < 50 and flag[1] <= 20) or \
               (50 <= data_p[i][-1] < 65 and flag[2] <= 20) or \
               (65 <= data_p[i][-1] < 85 and flag[3] <= 20) or \
               (data_p[i][-1] >= 85 and flag[4] <= 20):
                X_test.append(data_p[i][:-1])
                y_test.append(data_p[i][-1])
            elif (0 <= data_p[i][-1] < 30 and flag[0] <= nums[0]) or \
                 (30 <= data_p[i][-1] < 50 and flag[1] <= nums[1]) or \
                 (50 <= data_p[i][-1] < 65 and flag[2] <= nums[2]) or \
                 (65 <= data_p[i][-1] < 85 and flag[3] <= nums[3]) or \
                 (data_p[i][-1] >= 85 and flag[4] <= nums[4]):
                X_train.append(data_p[i][:-1])
                y_train.append(data_p[i][-1])
            else:
                continue
    else:
        for i in range(len(data_p)):
            if data_p[i][-1] == 0:
                flag[0] += 1
            elif data_p[i][-1] == 1:
                flag[1] += 1
            elif data_p[i][-1] == 2:
                flag[2] += 1
            elif data_p[i][-1] == 3:
                flag[3] += 1
            else:
                flag[4] += 1
            if (data_p[i][-1] == 0 and flag[0] <= 5) or (data_p[i][-1] == 1 and flag[1] <= 5) or \
               (data_p[i][-1] == 2 and flag[2] <= 0) or (data_p[i][-1] == 3 and flag[3] <= 0) or \
               (data_p[i][-1] == 4 and flag[4] <= 0):
                X_test.append(data_p[i][:-1])
                y_test.append(data_p[i][-1])
            elif (data_p[i][-1] == 0 and flag[0] <= nums[0]) or (data_p[i][-1] == 1 and flag[1] <= nums[1]) or \
                 (data_p[i][-1] == 2 and flag[2] <= nums[2]) or (data_p[i][-1] == 3 and flag[3] <= nums[3]) or \
                 (data_p[i][-1] == 4 and flag[4] <= nums[4]):
                  X_train.append(data_p[i][:-1])
                  y_train.append(data_p[i][-1])
            else:
                continue
    # print(X_train[0], X_test[0])
    # print(len(X_train), len(X_test[0]))
    return X_train, X_test, y_train, y_test


def train():
    # 读取数据
    df = pd.read_excel("Dataset_features_3.xlsx", sheet_name='descriptors')
    df_li = df.values.tolist()
    data_thermo = []
    data_features = []
    for s_li in df_li:
        if s_li[-1] != 5:
            data_thermo.append(s_li[-2])
            data_features.append(s_li[1:-2])
    # print(data_features[0])
    nums = np.zeros([5], dtype=int)
    label = np.zeros([len(data_thermo)], dtype=int)
    for i in range(len(data_thermo)):
        if data_thermo[i] < 30:
            nums[0] += 1
            label[i] = 0
        elif data_thermo[i] < 50:
            nums[1] += 1
            label[i] = 1
        elif data_thermo[i] < 65:
            nums[2] += 1
            label[i] = 2
        elif data_thermo[i] < 85:
            nums[3] += 1
            label[i] = 3
        else:
            nums[4] += 1
            label[i] = 4
    # print(nums)
    # 建立预测模型
    final_r = 0
    for i in range(0, 1):
        # 回归预测
        Is_regre = True
        X_train, X_test, y_train, y_test = group_sample(data_features, data_thermo, nums, Is_regre)
        reg1 = GradientBoostingRegressor(random_state=1, n_estimators=100)
        reg2 = RandomForestRegressor(random_state=1, n_estimators=100)
        reg3 = KNeighborsRegressor()
        model = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
        model.fit(X_train, y_train)
        y_pre = model.predict(X_test)
        joblib.dump(model, 'Mymodel_3.pkl')
        # acc = 0.0
        # for j in range(len(y_pre)):
        #     if (y_pre[j] > 50 and y_test[j] > 50) or (y_pre[j] <= 50 and y_test[j] <= 50):
        #         acc += 1
        # acc /= len(y_pre)
        r = calc_corr(y_pre, y_test)
        print(r)
        final_r += r
        xx = [30, 40, 50, 60, 70, 80, 90]
        yy = [30, 40, 50, 60, 70, 80, 90]
        if r >= 0.8:
            plt.scatter(y_pre, y_test)
            plt.scatter(xx, yy)
            plt.show()
        # 分类预测
        # Is_regre = False
        # X_train, X_test, y_train, y_test = group_sample(counts, label, nums, Is_regre)
        # model = ExtraTreesClassifier()
        # model.fit(X_train, y_train)
        # r = model.score(X_test, y_test)
        # print(r)
        # final_r += r
    print("The final r is ", final_r/1)


def test():
    df = pd.read_excel("Dataset_test_3.xlsx", sheet_name='descriptors')
    df_li = df.values.tolist()
    X = []
    for s_li in df_li:
        X.append(s_li[1:])
    model = joblib.load('Mymodel_3.pkl')
    y_pre = model.predict(X)
    print(y_pre)


if __name__ == '__main__':
    # train()
    test()
