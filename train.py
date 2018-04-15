import torch
import numpy as np
from torch.autograd import Variable
from torch import nn

from tools import DataLoader
from model import AutoEncoder
from tools import make_var

from progress.bar import Bar

import time


def validate(model, criterion, valid_x, valid_y, batch_size=128):
    losses = []
    times = []
    data_len = len(valid_x)

    progress = Bar("Validation", max=int(data_len / batch_size))

    for i in range(0, data_len, batch_size):
        # process batches
        x = valid_x[i: i + batch_size]
        y = valid_y[i: i + batch_size]
        # make cuda variables
        x = make_gpu(x)
        y = make_gpu(y)
        # make predictions
        start_time = time.time()
        predicted_y = model(x)
        end_time = time.time()
        loss = criterion(predicted_y, y)
        losses.append(loss.data[0])
        times.append(end_time - start_time)
        # monitor progress
        progress.next()
    progress.finish()

    return np.mean(losses), np.mean(times)


def train_step(model, criterion, optimizer, train_x, train_y, batch_size=128):
    losses = []
    data_len = len(train_x)

    progress = Bar("Training", max=int(data_len / batch_size))

    for i in range(0, data_len, batch_size):
        # process batches
        x = train_x[i: i + batch_size]
        y = train_y[i: i + batch_size]
        # make cuda variables
        x = make_gpu(x)
        y = make_gpu(y)
        # make predictions
        predicted_y = model(x)
        loss = criterion(predicted_y, y)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])
        # monitor progress
        progress.next()
    progress.finish()

    return np.mean(losses)


def main(num_epochs = 100, batch_size = 64, learning_rate = 1e-3, early_stopping=5, shuffle=True):
    # load data
    data = DataLoader("../autencoder/convex_hulls.npy", batch_size=batch_size)

    # load the model and parameters
    model = AutoEncoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train the stuff
    last_valid_loss = 1000
    stop_iter = early_stopping

    for epoch in range(num_epochs):
        print("\n======== Epoch [{}/{}] ========".format(epoch + 1, num_epochs))
        if shuffle:
            data.shuffle()
        train_loss = train_step(model, criterion, optimizer, data.train_x, data.train_y, batch_size)
        valid_loss, exe_time = validate(model, criterion, data.valid_x, data.valid_y, batch_size)
        print('Train loss: {:.4f}\nValid loss:{:.4f}'.format(train_loss, valid_loss))
        print('Average execution time {:.5f}'.format(exe_time / batch_size))

        # Early stopping
        if last_valid_loss <= valid_loss:
            stop_iter -= 1
        else:
            stop_iter = early_stopping
            last_valid_loss = valid_loss

        if not stop_iter:
            print("> Early stopping")
            break

    torch.save(model.state_dict(), './conv_autoencoder.pth')

if __name__ == "__main__":
    main()