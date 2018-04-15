import numpy as np
import random
import torch
from torch.autograd import Variable


class DataLoader(object):
    def __init__(self, path):
        self.x, y = np.load(path, mmap_mode='r+')

        x = self.x.reshape(-1, 1, 300, 400)
        y = y.reshape(-1, 1, 300, 400)

        x = self.normalize(x)
        y = self.normalize(y)

        self.pivot = int(len(x) * 0.75)
        self.train_x = x[:self.pivot]
        self.train_y = y[:self.pivot]
        self.valid_x = x[self.pivot:]
        self.valid_y = y[self.pivot:]

    def normalize(self, data):
        data[data > 0] = 1.0
        return data

    def shuffle(self):
        zipped = list(zip(self.train_x, self.train_y))
        random.shuffle(zipped)
        self.train_x, self.train_y = zip(*zipped)
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)


def make_gpu(var):
    return Variable(torch.from_numpy(var)).float().cuda()


def make_cpu(var):
    return Variable(torch.from_numpy(var)).float().cpu()


def make_numpy(var):
    return var.data.numpy()

