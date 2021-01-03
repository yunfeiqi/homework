#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/29 21:09:38
@Author  :   sam.qi
@Version :   1.0
@Desc    :   Homework 8 Seq2Seq
'''

from numpy.core.defchararray import mod
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import nltk
import random
import torch.nn as nn
import re
import torch
from torch.utils.data import Dataset

import os
import json
import numpy as np
from torch.utils.data.dataloader import DataLoader

# ------------------------------- 基础配置 ---------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------构造 Dataset class---------------------------------


class LabelTransform(object):
    def __init__(self, size, pad) -> None:
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(
            label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label


class EN2CnDataset(Dataset):
    '''
        定义英文转中文Dataset
    '''

    def __init__(self, root, max_output_len, set_name) -> None:
        super().__init__()
        self.root = root

        # 加载分词字典
        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')

        # 加载数据集
        self.data = []
        with open(os.path.join(self.root, f'{set_name}.txt'), 'r') as f:
            for line in f:
                self.data.append(line)

        print(f'{set_name} dataset size: {len(self.data)}')

        # 统计
        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)

        # 将所有输入和输出对齐
        self.transform = LabelTransform(
            max_output_len, self.word2int_en['<PAD>'])

    def get_dictionary(self, lang):
        '''
            根据不同的语言，加载不同的Vocab Map
        '''
        with open(os.path.join(self.root, f'word2int_{lang}.json'), 'r') as f:
            word2int = json.load(f)

        with open(os.path.join(self.root, f"int2word_{lang}.json"), 'r') as f:
            int2word = json.load(f)

        return word2int, int2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentences = self.data[index]
        # 拆分中英文
        sentences = re.split('[\t]', sentences)
        sentences = list(filter(None, sentences))
        assert len(sentences) == 2

        # 特殊子元
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']

        # 在开头添加 BOS 结尾添加EOS，OOV 使用 UNK
        en, cn = [BOS], [BOS]
        sentence = re.split(" ", sentences[0])
        sentence = list(filter(None, sentence))
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        sentence = re.split(" ", sentence[1])
        sentence = list(filter(None, sentence))
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)

        en, cn = np.array(en), np.array(cn)
        # 将句子补齐
        en, cn = self.transform(en), self.transform(cn)
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)

        return en, cn


# ------------------------------- 模型部分 Module --------------------------------

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout=0.02):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # 双向RNN
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers,
                          dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input: batch * sequence_len

        # emb: batch * seq_len * emb_dim
        emb = self.embedding(input)

        # outputs: batch * sequence *  (hid_dim * bidir)
        # hidden:  (n_layer * direction) * batch * hid_dim
        outputs, hidden = self.rnn(self.dropout(emb))

        return outputs, hidden


class RNNDecoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
        self.isatt = isatt

        self.input_dim = emb_dim
        self.rnn = nn.GRU(self.input_dim, self.hid_dim,
                          self.n_layers, dropout=dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim*2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim*2, self.hid_dim*4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim*4, self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_output):
        # input: [batch, vocab_size]
        # hidden: [batch ,layer * direction,hid_dim]

        input = input.unsqueeze(1)
        # [batch , 1, emb_dim]
        embeded = self.dropout(self.embedding(input))

        # output: [batch ,1, hid_dim ]
        # hidden :[n_layer,batch size,hid_dim]
        output, hidden = self.rnn(embeded, hidden)
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        return prediction, hidden


class Seq2Seq(nn.Module):
    '''
        由Encoder 和Decoder组成
        接受输入传给Encoder
        将Encoder 的输出传给Decoder
        不断的将Decoder 输出回传给Decoder，进行解码
        解码完成，将Decoder 输出回传
    '''

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, target, teacher_forcing_ratio):
        # input: [batch,sequence]
        # target: [batch,target len]
        # teacher_foring_ratio: 有多少几率使用Target
        batch_size = target.size(0)
        target_len = target.size(1)
        vocab_size = self.decoder.cn_vocab_size

        # 存储结果
        outputs = torch.zeros(batch_size, target_len,
                              vocab_size).to(self.device)

        # ** 进行Encoder操作**
        encoder_output, hidden = self.encoder(input)
        # encoder_output 主要用于 Attension
        # encoder_hidden 用来初始化 Decoder的 Hidden
        # 因为Encoder是双向的，其维度是：[n_layer * direction,Batsh,Hid_dim] 所以需要进行转化
        # 转化方法就是将 direction 拼接在一起
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

        # 取 BOS
        input = target[:, 0]
        preds = []

        # ** 开始 Decoder **
        for step in range(target_len):
            decoder_output, hidden = self.decoder(
                input, hidden, encoder_output)
            outputs[:, step] = decoder_output
            teacher_force = random.random() <= teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            input = target[:,
                           step] if teacher_force and step < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, input, target):
        # TODO Use Beam Search

        batch_size = input.size(0)
        input_len = input.size(1)
        vocab_size = self.decoder.cn_vocab_size
        outputs = torch.zeros(batch_size, input_len,
                              vocab_size).to(self.device)

        # 开始 Encoder
        encoder_output, hidden = self.encoder(input)
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

        # 开始 Decoder
        input = target[:, 0]
        preds = []
        for step in range(input_len):
            decoder_output, hidden = self.decoder(
                input, hidden, encoder_output)
            outputs[:, step] = decoder_output
            top1 = decoder_output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds


# ------------------------------- 支持方法 --------------------------------

def load_model(model, path):
    '''
        加载模型参数
    '''
    print("加载模型参数：{}".format(path))
    model.load_state_dict(torch.load(path))
    return model


def build_model(config, en_vocab_size, cn_vocab_size):
    # 构建模型实例
    encoder = RNNEncoder(en_vocab_size, config.emb_dim,
                         config.hid_dim, config.n_layers, config.dropout)
    decoder = RNNDecoder(cn_vocab_size, config.emb_dim, config.hid_dim,
                         config.n_layers, config.dropout, config.isatt)
    model = Seq2Seq(encoder, decoder, config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.load_model_path is not None:
        model = load_model(model, config.load_model_path)
    model = model.to(config.device)
    return model, optimizer


def token2sentence(output, int2word):
    ''' 
        将数字转化为字
    '''
    sentences = []
    for tokens in output:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == "<EOS>":
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences


def infinite_iter(data_loader):
    ''' 
        无限迭代器
    '''

    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)


def schedule_sampling():
    '''
        Schedule Sampling 策略
    '''

    return 1


def computebleu(sentences, targets):
    # score = 0
    # assert (len(sentences) == len(targets))

    # def cut_token(sentence):
    #     tmp = []
    #     for token in sentence:
    #         if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
    #             tmp.append(token)
    #         else:
    #             tmp += [word for word in token]
    #     return tmp

    # for sentence, target in zip(sentences, targets):
    #     sentence = cut_token(sentence)
    #     target = cut_token(target)
    #     score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))

    # return score
    return 0


# ------------------------------- 模型训练 Training & testing --------------------------------


def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps, train_dataset):
    '''
        训练模型
    '''
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0

    for step in range(summary_steps):
        optimizer.zero_grad()

        sources, target = next(train_iter)
        sources, target = sources.to(device), target.to(device)
        outputs, preds = model(sources, target, schedule_sampling())

        # 忽略 Target 的第一个Token，因为它是BOS
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        target = target[:, 1:].reshape(-1)
        loss = loss_function(outputs, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()
        if (step+1) % 5 == 0:
            loss_sum = loss_sum / 5
            print("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}      ".format(
                total_steps + step + 1, loss_sum, np.exp(loss_sum)), end=" ")
            losses.append(loss_sum)
            loss_sum = 0
    return model, optimizer, losses


# ------------------------------- 模型预测 Predict --------------------------------

def testing(model, dataloader, loss_function):
    model.eval()
    loss_sum = 0
    bleu_score = 0
    n = 0

    result = []
    for source, target in dataloader:
        source, target = source.to(device), target.to(device)
        batch_size = source.size(0)
        output, preds = model.inference(source, target)
        output = output[:, 1:].reshape(-1, output.size(2))
        target = target[:, 1:].reshape(-1)

        loss = loss_function(output, target)
        loss_sum += loss.item()

        # 将预测结果转为文字
        targets = target.view(source.size(0), -1)
        preds = token2sentence(preds, dataloader.dataset.int2word_cn)
        sources = token2sentence(
            source.to("cpu").numpy(), dataloader.dataset.int2word_en)
        targets = token2sentence(
            targets.to("cpu").numpy(), dataloader.dataset.int2word_cn)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))

        # 计算Blue Score
        bleu_score += computebleu(preds, targets)
        n += batch_size
    return loss_sum/len(dataloader), bleu_score/n, result


# ------------------------------- 配置类 --------------------------------
class configurations(object):
    def __init__(self):
        self.batch_size = 60
        self.emb_dim = 256
        self.hid_dim = 512
        self.n_layers = 3
        self.dropout = 0.5
        self.lr = 0.00005
        self.max_output_len = 50              # 最後輸出句子的最大長度
        self.num_steps = 350  # 12000                # 總訓練次數
        self.store_steps = 300                # 訓練多少次後須儲存模型
        self.summary_steps = 50  # 300              # 訓練多少次後須檢驗是否有overfitting
        self.load_model = False               # 是否需載入模型
        self.store_model_path = "./ckpt"      # 儲存模型的位置
        # 載入模型的位置 e.g. "./ckpt/model_{step}"
        self.load_model_path = None
        self.data_path = "./data/cmn-eng"          # 資料存放的位置
        self.isatt = False                # 是否使用 Attention Mechanism
        self.device = device


# ------------------------------- 训练&测试模型入口 --------------------------------

def train_process(config):
    # 准备训练数据
    train_dataset = EN2CnDataset(
        config.data_path, config.max_output_len, 'training')
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    # 准备验证数据
    val_datset = EN2CnDataset(
        config.data_path, config.max_output_len, 'validation')
    val_loader = DataLoader(val_datset, batch_size=1)

    # 构建模型实例
    model, optimizer = build_model(
        config, train_dataset.en_vocab_size, val_datset.cn_vocab_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while total_steps < config.num_steps:
        # 训练模型
        model, optimizer, loss = train(
            model, optimizer, train_iter, loss_function, total_steps, config.summary_steps, train_dataset)
        train_losses += loss

        # 验证模型
        val_loss, blue_score, result = testing(
            model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(blue_score)

        total_steps += config.summary_steps
        print("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}".format(
            total_steps, val_loss, np.exp(val_loss), blue_score))


def test_process(config):
    test_dataset = EN2CnDataset(
        config.data_path, config.max_output_len, 'testing')
    test_loader = DataLoader(test_dataset, batch_size=1)
    model, optimizer = build_model(
        config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
    print("Finish build model")
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    test_loss, blue_score, result = testing(model, test_loader, loss_function)
    # 保存结果
    with open("./test_output.txt", 'w') as f:
        for line in result:
            print(line, file=f)

    return test_loss, blue_score


# ------------------------------- 开始训练模型 --------------------------------
config = configurations()
print('config:\n', vars(config))
train_losses, val_losses, bleu_scores = train_process(config)

# ------------------------------- 开始测试模型 --------------------------------
# config = configurations()
# print('config:\n', vars(config))
# test_loss, bleu_score = test_process(config)
# print(f'test loss: {test_loss}, bleu_score: {bleu_score}')
