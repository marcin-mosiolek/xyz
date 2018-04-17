import numpy as np
import random
import torch
from torch.autograd import Variable


class DataLoader(object):
    def __init__(self, path, keep_original=False):
        x, y = np.load(path)
        x = x.reshape(-1, 1, 300, 400)
        y = y.reshape(-1, 1, 300, 400)

        if keep_original:
            self.x = x.copy()
            self.y = y.copy()

        self.pivot = int(len(x) * 0.75)
        self.train_x = x[:self.pivot]
        self.train_y = y[:self.pivot]
        self.valid_x = x[self.pivot:]
        self.valid_y = y[self.pivot:]

        self.train_x = self.normalize(self.train_x)
        self.train_y = self.normalize(self.train_y)
        self.valid_x = self.normalize(self.valid_x)
        self.valid_y = self.normalize(self.valid_y)

    def normalize(self, data):
        data[data > 0] = 1.0
        return data

    def shuffle(self):
        zipped = list(zip(self.train_x, self.train_y))
        random.shuffle(zipped)
        x, y = zip(*zipped)
        self.train_x = np.array(x)
        self.train_y = np.array(y)


def make_gpu(var):
    return Variable(torch.from_numpy(var)).float().cuda()


def make_cpu(var):
    return Variable(torch.from_numpy(var)).float().cpu()


def make_numpy(var):
    return var.data.numpy()

