#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/14 14:17:37
@Author  :   sam.qi
@Version :   1.0
@Desc    :   训练工具
'''

import torch
from torch._C import dtype
from torch.nn import Module
from torch.utils.data import dataloader


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

        for epoch in range(self.num_epoch):
            for step, data in enumerate(data_train):
                X = data[0]
                y = data[1]
                inputs = X.to(self.device)
                labels = y.to(self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                # output: batch * 1 * 1
                outputs = self.model(inputs)
                # output: batch * 1
                outputs = outputs.squeeze(dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if step % 10 == 0:
                    print(loss.item())
