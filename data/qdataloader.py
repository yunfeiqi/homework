#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/14 12:41:47
@Author  :   sam.qi
@Version :   1.0
@Desc    :   自定义DataLoader，将给定的数据集转化成DataLoader
'''
import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset

from common.file import check_exist, read_all_lins


class BaseDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        super().__init__()
        if not isinstance(X, torch.LongTensor):
            X = torch.LongTensor(X)

        if y is not None and not isinstance(y, torch.LongTensor):
            y = torch.LongTensor(y)
        self.data = X
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]


class DataAccess(object):
    '''
        数据接入
    '''

    def __init__(self) -> None:
        super().__init__()
        self.dataset = None

    def get_dataloader(self, batch_size, shuffle=True):
        if self.dataset is None:
            raise RuntimeError("dataset 未能构建构成功")

        return data.dataloader.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle)


class MemoryDataAccess(DataAccess):
    '''
        文件加载类
    '''

    def __init__(self, x, y=None) -> None:
        super().__init__()
        self.dataset = BaseDataset(x, y)
