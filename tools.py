import numpy as np
import random


class DataLoader(object):
    def __init__(self, path, batch_size=128):
        x, y = np.load(path, mmap_mode='r+')

        x = x.reshape(-1, 1, 300, 400)
        y = y.reshape(-1, 1, 300, 400)

        self.pivot = int(len(x) * 0.75)
        self.train_x = x[:self.pivot]
        self.train_y = y[:self.pivot]
        self.valid_x = x[self.pivot:]
        self.valid_y = x[self.pivot:]
        self.batch_size = batch_size

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


