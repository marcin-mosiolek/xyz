import numpy as np
import random
import torch
from torch.autograd import Variable


class DataLoader(object):
    def __init__(self, path, batch_size=128):
        x, y = np.load(path, mmap_mode='r+')

        x = x.reshape(-1, 1, 300, 400)
        y = y.reshape(-1, 1, 300, 400)

        x = self.normalize(x)
        y = self.normalize(y)

        self.pivot = int(len(x) * 0.75)
        self.train_x = x[:self.pivot]
        self.train_y = y[:self.pivot]
        self.valid_x = x[self.pivot:]
        self.valid_y = x[self.pivot:]
        self.batch_size = batch_size

    def normalize(self, data):
        data[data > 0] = 1.0
        return data

    def shuffle(self):
        zipped = list(zip(self.train_x, self.train_y))
        random.shuffle(zipped)
        self.train_x, self.train_y = zip(*zipped)

    def __iter__(self):
        return self.next()

    def __next__(self):
        return self.next()

    def next(self):
        for i in range(0, self.pivot, self.batch_size):
            yield self.train_x[i: i + self.batch_size], self.train_y[i : i + self.batch_size]

    def len(self):
        return int(len(self.train_x)/self.batch_size)

    def valid_data(self):
        return self.valid_x, self.valid_y


def make_var(var):
    return Variable(torch.from_numpy(var)).float().cuda()


