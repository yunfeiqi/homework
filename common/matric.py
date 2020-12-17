#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/15 18:54:37
@Author  :   sam.qi
@Version :   1.0
@Desc    :   指标
'''

import torch


def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1  # 大於等於0.5為有惡意
    outputs[outputs < 0.5] = 0  # 小於0.5為無惡意
    correct = torch.mean(torch.eq(outputs, labels)).item()
    return correct
