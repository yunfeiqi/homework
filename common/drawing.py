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
    # plt.plot(x_axis, y2)
    plt.xlabel("epoch number")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(title)
    plt.close()


def draw_dubble_y_2d(x, y1, y2, y1_label="label:y1", y2_label="label:y2", title='双坐标2维图'):
    '''
        绘制二维，双坐标图形
    '''
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, label=y1_label, color="red")
    ax2 = ax1.twinx()
    ax2.plot(x, y2, label=y2_label, color='green')
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(y1_label)
    ax2.set_ylabel(y2_label)

    plt.savefig(title)
    plt.close()
