import torch
import numpy as np
from torch.autograd import Variable
from torch import nn

from tools import DataLoader
from model import AutoEncoder
from tools import make_var

from progress.bar import Bar


def validate(model, criterion, valid_x, valid_y, batch_size=128):
    losses = []
    data_len = len(valid_x)

    progress = Bar("Validation", max=int(data_len / batch_size))

    for i in range(0, data_len, batch_size):
        # process batches
        x = valid_x[i: i + batch_size]
        y = valid_y[i: i + batch_size]
        # make cuda variables
        x = make_var(x)
        y = make_var(y)
        # make predictions
        predicted_y = model(x)
        loss = criterion(predicted_y, y)
        losses.append(loss.data[0])
        # monitor progress
        progress.next()
    progress.finish()

    return np.mean(losses)


def train_step(model, criterion, optimizer, train_x, train_y, batch_size=128):
    losses = []
    data_len = len(train_x)

    progress = Bar("Training", max=int(data_len / batch_size))

    for i in range(0, data_len, batch_size):
        # process batches
        x = train_x[i: i + batch_size]
        y = train_y[i: i + batch_size]
        # make cuda variables
        x = make_var(x)
        y = make_var(y)
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


def main(num_epochs = 100, batch_size = 64, learning_rate = 1e-3):
    # load data
    data = DataLoader("../autencoder/convex_hulls.npy", batch_size=batch_size)

    # load the model and parameters
    model = AutoEncoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # train the stuff
    for epoch in range(num_epochs):
        print("======== Epoch [{}/{} ========".format(epoch + 1, num_epochs))
        valid_loss = validate(model, criterion, data.valid_x, data.valid_y, batch_size)
        train_loss = train_step(model, criterion, optimizer, data.train_x, data.train_y, batch_size)

        print('Train loss: {:.4f}\nValid loss:{:.4f}'.format(train_loss, valid_loss))

    torch.save(model.state_dict(), './conv_autoencoder.pth')

if __name__ == "__main__":
    main()