#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/22 13:00:08
@Author  :   sam.qi
@Version :   1.0
@Desc    :   Homeword 3 基于CNN的图片分类
'''


from numpy.core.defchararray import mod
from numpy.lib.function_base import _percentile_dispatcher
from torch import optim
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool1d, MaxPool2d
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import re
from numpy.lib.twodim_base import mask_indices
import cv2
import os
import numpy as np


train_size = 500
val_size = 200
test_size = 100

# 读取数据集


def readfile(path, label=None):
    images_dir = sorted(os.listdir(path))
    d_size = train_size+test_size+val_size
    x = np.zeros((d_size, 128, 128, 3), dtype=np.uint8)
    if label is None:
        y = None
    else:
        y = np.zeros((d_size), dtype=np.uint8)
    for idx in range(d_size):
        file = images_dir[idx]
        if len(re.findall(r"\\.gif$", file)) > 0:
            continue
        p_file = os.path.join(path, file)
        p_file = p_file.replace('\\', '/')
        print(p_file)
        img = cv2.imread(p_file)
        x[idx, :, :] = cv2.resize(img, (128, 128))
        if y is not None:
            y[idx] = label

    if y is None:
        return x
    else:
        return x, y
# -------------------------------构造 Dataset class---------------------------------


# training 时做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 图片水平旋转
    transforms.RandomRotation(15),  # 随机旋转图片
    transforms.ToTensor()  # 图片转Tensor，并惊醒归一化
])

# testing 时不需要要做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# 构造DataSet


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None) -> None:
        super().__init__()
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


# --------------------------------定义变量--------------------------------
batch_size = 128

# 分别将 Training Data testting Data 用readfile 读进来
pokemon_path = "C:/data/images/pokemon/"
x_p, y_p = readfile(pokemon_path, 1)

digimon_path = "C:/data/images/digimon/"
x_d, y_d = readfile(digimon_path, 0)


train_x = np.append(x_p[:train_size], x_d[:train_size], axis=0)
train_y = np.append(y_p[:train_size], y_d[:train_size], axis=0)


val_x = np.append(x_p[train_size:val_size+train_size],
                  x_d[train_size:val_size+train_size], axis=0)
val_y = np.append(y_p[train_size:val_size+train_size],
                  y_d[train_size:val_size+train_size], axis=0)

train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# --------------------------------定义 模型--------------------------------


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 64 * 128 * 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 64 * 64 * 64

            nn.Conv2d(64, 128, 3, 1, 1),  # 128 * 64 * 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),   # 128 * 32 * 32

            nn.Conv2d(128, 256, 3, 1, 1),  # 256,32,32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),    # 256, 16,16

            nn.Conv2d(256, 512, 3, 1, 1),  # 512,8,8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),    # 512,8,8

            nn.Conv2d(512, 512, 3, 1, 1),  # 512,8,8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)  # 512, 4,4
        )
        self.fn = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fn(out)


# -------------------------------- 基础模型训练 --------------------------------

device = "cpu"
model = Classifier()
model = model.to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 30


def train():
    for epoch in range(num_epoch):
        train_acc = 0
        train_loss = 0
        val_acc = 0
        val_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            X = data[0]
            X = X.to(device)
            Y_hat = model(X)
            Y = data[1].to(device)

            batch_loss = loss(Y_hat, Y)
            batch_loss.backward()
            optimizer.step()

            # 计算 acc and loss
            train_acc += np.sum(np.argmax(Y_hat.cpu().data.numpy(),
                                          axis=1) == Y.numpy())
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X = data[0].to(device)
                Y = data[1]
                Y_hat = model(X)
                batch_loss = loss(Y_hat, Y.to(device))

                val_acc += np.sum(torch.argmax(Y_hat,
                                               dim=1).cpu() == Y.numpy())
                val_loss += batch_loss.item()

        # print result
        print("{}-{}, Train Acc:{},Train Loss:{}, Val Acc:{}, Val Loss:{}".format(epoch + 1, num_epoch, train_acc / train_size,
                                                                                  train_loss / train_size, val_acc/val_size, val_loss/val_size))


# -------------------------------- 将 Train 和 Val 合并训练更好模型 基础模型训练 --------------------------------
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(
    train_val_set, batch_size=batch_size, shuffle=True)

best_model = Classifier()
best_model = best_model.to(device)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=0.001)
num_epoch = 30

for epoch in range(num_epoch):
    train_acc = 0
    train_loss = 0
    best_model.train()

    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        X = data[0].to(device)
        Y = data[1]
        Y_hat = best_model(X)
        batch_loss = loss(Y_hat, Y.to(device))
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(torch.argmax(Y_hat, dim=1).cpu() == Y.numpy())
        train_loss += batch_loss.item()

    print("{}-{}, Train Acc:{},Loss:{}".format(epoch+1, num_epoch, train_acc /
                                               train_val_set.__len__(), train_loss/train_val_set.__len__()))

# -------------------------------- 利用模型进行预测 --------------------------------
test_x = x_p[-1*test_size:]
test_y = y_p[-1*test_size:]
test_x.extend(x_d[-1*test_size:])
test_y.extend(y_d[-1*test_size:])
test_set = ImgDataset(test_x, test_y, test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

best_model.eval()
test_acc = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        X = data[0]
        Y = data[1]
        Y_hat = best_model(X.to(device))
        Y_hat_label = torch.argmax(Y_hat, dim=1)
        test_acc += np.sum(Y_hat_label == Y.numpy())

print("TestAcc:{}".format(test_acc/test_size))
