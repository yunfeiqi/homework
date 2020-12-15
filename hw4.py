from numpy.lib.ufunclike import fix
from common.training import Train
from data.qdataloader import BaseDataset
from common.file import read_all_lins
from common.file import check_exist
from data.qdataloader import DataAccess
import torch.optim as optim
import torch
from torch.nn import parameter
from torch.nn.modules import batchnorm
from torch.utils import data
import os
import torch
from torch import nn

from gensim.models import ldamodel, word2vec


def evaluation(outputs, labels):
    outputs[outputs >= 0.5] = 1  # 正
    outputs[outputs < 0.5] = 0  # 负
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


def train_word2vec(x):
    model = word2vec.Word2Vec(x, sim=250, window=5,
                              min_count=5, workers=12, iter=10, sg=1)
    return model


# print("loading training data ...")
# train_x, y = load_training_data("training_label.txt")
# train_x_no_label = load_training_data("training_nolabel.txt")

# print("load testing data ...")
# test_x = load_testing_data("testing_data.txt")
# model = train_word2vec(train_x + train_x_no_label + test_x)
# print("saveing model ")
# model.save(os.path.join("./", "w2v_all.model"))


class LSTM_Net(nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(600000, hidden_dim)
            print(600000, hidden_dim)
        else:
            print(embedding.size(0), embedding.size(1))
            self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = self.embedding.weight.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim,
                            num_layers=num_layers, batch_first=True)

        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, 1),
                                        nn.Sigmoid())

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        # x : batch * seq_len * hidden_size
        x, _ = self.lstm(inputs)

        # LSTM 最后一个hidden state
        # x: batch * 1 * hidden_size
        x = x[:, -1, :]
        # batch * 1 * 1
        x = self.classifier(x)
        return x


def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameter() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(
        total, trainable))

    model.train()
    criterion = nn.BCELoss()
    t_batch = len(train)
    v_batch = len(valid)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_Loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        for i, (input, labels) in enumerate(train):
            inputs = inputs.to(device, device=torch.long)
            labels = labels.to(device, device=torch.float)

            optimizer.zero_grad()
            outputs = model(inputs)
            outpus = outputs.sequeeze()
            loss = criterion(outputs, labels)
            loss.backword()
            optimizer.step()
            correct = evaluation(outputs, labels)
            total_acc += (correct/batch_size)
            total_loss = loss.item()


class FileDataAccess(DataAccess):
    '''
        文件加载类
    '''

    def __init__(self, x_path, max_sentence=60, vocab=None) -> None:
        super().__init__()
        self.x_path = x_path
        self.max_sentence = max_sentence
        self.dataset = None
        self.vocab = vocab
        self.data_load()

    def data_load(self):
        """
        load data from file
        """
        x_data = []
        labels = None

        if self.x_path is None or not check_exist(self.x_path):
            raise RuntimeError("指定文件不存在")
        lines = read_all_lins(self.x_path)
        lines = [line.strip("\n").split(" ") for line in lines]

        if "training_label" in self.x_path:
            labels = [int(line[0]) for line in lines]
            lines = [line[2:] for line in lines]
        # split

        word2idx = {}
        idx2word = []
        ids = []
        for line in lines:
            id_line = []
            for word in line:
                word = word.lower().rstrip()
                if self.vocab is None:
                    id = word2idx.get(word, None)
                    if id is None:
                        id = len(word2idx)
                        word2idx[word] = id
                else:
                    if word in self.vocab:
                        id = self.vocab[word]
                    else:
                        id = self.vocab['<UNK>']
                idx2word.append(word)
                id_line.append(id)

            if len(id_line) > self.max_sentence:
                id_line = id_line[: self.max_sentence]
            else:
                pad_len = self.max_sentence - len(id_line)
                for i in range(pad_len):
                    id_line.append(0)

            ids.append(id_line)

        self.data = ids
        self.labels = labels
        self.dataset = BaseDataset(self.data, self.labels)


def make_embedding():
    embedding_matrix = []
    print("Get embedding ...")
    embedding = word2vec.Word2Vec.load('w2v_all.model')

    embedding_dim = embedding.vector_size
    word2id = {}
    for i, word in enumerate(embedding.wv.vocab):
        print('get words #{}'.format(i+1), end='\r')
        word2id[word] = len(word2id)

        embedding_matrix.append(embedding[word])
    print('')
    embedding_matrix = torch.tensor(embedding_matrix)

    # add token
    vectors = torch.empty(2, embedding_dim)
    word2id["<SPE>"] = len(word2id)
    word2id["<UNK>"] = len(word2id)
    embedding_matrix = torch.cat([embedding_matrix, vectors], dim=0)
    print("total words: {}".format(len(embedding_matrix)))
    return embedding_matrix, word2id


def entry():
    embedding, vocab = make_embedding()
    model = LSTM_Net(embedding, 250, 1, fix_embedding=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Train(model, criterion, optimizer, 5, device="cpu")

    fds = FileDataAccess("training_label.txt",
                         max_sentence=10, vocab=vocab)
    trainer.start(fds.get_dataloader(batch_size=5))


if __name__ == "__main__":
    entry()
