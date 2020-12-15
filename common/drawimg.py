#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/15 17:13:18
@Author  :   sam.qi
@Version :   1.0
@Desc    :   绘图
'''

import matplotlib.pyplot as plt


def draw_plot(x, y, title):
    x_axis = x
    plt.plot(x_axis, y)
    plt.xlabel("epoch number")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(title)
    plt.close()
