import torch
import numpy as np
from torch.autograd import Variable
from torch import nn

from tools import DataLoader
from model import VAE
from tools import make_gpu

from progress.bar import Bar
from torch.nn import functional as F

import time

criterion = nn.MSELoss()

def validate(model, valid_x, valid_y, batch_size=128):
    model.training = False
    
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
        predicted_y, mu, logvar = model(x)
        end_time = time.time()

        loss = loss_function(predicted_y, y, mu, logvar)

        losses.append(loss.data[0])
        times.append(end_time - start_time)
        # monitor progress
        progress.next()
    progress.finish()

    return np.mean(losses), np.mean(times)


def loss_function(recon_x, y, mu, logvar):
    BCE = criterion(recon_x, y)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_step(model, optimizer, train_x, train_y, batch_size=128):
    model.training = True
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
        predicted_y, mu, logvar = model(x)
        print("Result: {}".format(predicted_y))
        loss = loss_function(predicted_y, y, mu, logvar)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])
        # monitor progress
        progress.next()
    progress.finish()

    return np.mean(losses)


def main(num_epochs = 1000, batch_size = 64, learning_rate = 1e-3, early_stopping=5, shuffle=False):
    # load parameters
    print("Loading model...")
    model = VAE().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())

    print("...completed")
    # load data
    print("Loading data...")
    data = DataLoader("/mnt/moria/voyage_clustering/convex_hulls2.npy")
    data.shuffle()
    print("...completed")
    print("Training set size: {}".format(data.train_x.shape))
    print("Validation set size: {}".format(data.valid_x.shape))

    # train the stuff
    best_valid_loss = 1000
    stop_iter = early_stopping
    
    for epoch in range(num_epochs):
        print("\n======== Epoch [{}/{}] ========".format(epoch + 1, num_epochs))
        if shuffle:
            data.shuffle()
        train_loss = train_step(model, optimizer, data.train_x, data.train_y, batch_size)
        valid_loss, exe_time = validate(model, data.valid_x, data.valid_y, batch_size)
        print('Train loss: {:.6f}\nValid loss: {:.6f}'.format(train_loss, valid_loss))
        print('Average execution time {:.6f}'.format(exe_time / batch_size))

        # Early stopping
        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            stop_iter = early_stopping
            print("New best model. Saving")
            torch.save(model.state_dict(), './conv_autoencoder.pth')
        else:
            stop_iter -= 1

        if not stop_iter:
            print("> Early stopping")
            break



if __name__ == "__main__":
    main()
