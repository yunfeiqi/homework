#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/14 14:17:37
@Author  :   sam.qi
@Version :   1.0
@Desc    :   训练工具
'''

import torch
from torch.nn import Module
from torch.utils.data import dataloader
from common.drawing import draw_plot, draw_dubble_y_2d
from common.matric import evaluation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Train(object):
    def __init__(self, model: Module, criterion, optimizer, num_epoch, device="cpu") -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.device = device

    def start(self, data_train: dataloader.DataLoader):
        self.model.train()

        # model to device
        self.model = self.model.to(self.device)

        loss_items = []
        acc_items = []
        steps = []
        for epoch in range(self.num_epoch):
            print("Start Epoch:{}".format(epoch))
            for step, data in enumerate(data_train):
                X = data[0]
                y = data[1]
                inputs = X.to(self.device)
                labels = y.to(self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                # output: batch * 1 * 1
                outputs = self.model(inputs)
                # output: batch * 1
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if step > 0 and step % 100 == 0:
                    _loss = loss.item()
                    print(_loss)
                    loss_items.append(_loss)
                    steps.append(len(steps))
                    _acc = evaluation(outputs, labels)
                    acc_items.append(_acc)
                    draw_dubble_y_2d(steps, loss_items,
                                     acc_items, "训练损失-精度.png")
