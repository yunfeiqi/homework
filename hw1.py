#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/11 09:57:28
@Author  :   sam.qi
@Version :   1.0
@Desc    :   HomeWork one: Regression:
             目标： 根据前9个小时 18 个Feature 预测第10个小时的PM2.5值
'''

import pandas as pd

data_path = ""

# 加载数据集
data = pd.read_csv(data_path, encoding='big5')

# 数据预处理，提出有用数据，并把一些 Null 设置成 0
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()


def feature_extract_1():
    pass


def feature_extract_2():
    pass


def normalize():
    pass


def training():
    pass


def testing():
    pass


def predict():
    pass


def save_predict_2_csv():
    pass


if __name__ == "__main__":
    pass
