import torch
from torch.autograd import Variable
from torch import nn

from tools import DataLoader
from model import AutoEncoder
from tools import make_var

from progress.bar import Bar


def main(num_epochs = 100, batch_size = 128, learning_rate = 1e-3):
    # load data
    data = DataLoader("../autencoder/convex_hulls.npy", batch_size=batch_size)

    # load the model and parameters
    model = AutoEncoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    # train the stuff
    for epoch in range(num_epochs):
        progress = Bar("Training", max=data.len())
        for x, y in data:
            progress.next()
            #print(x.shape)
            x = make_var(x)
            y = make_var(y)
            # ===================forward=====================
            predicted_y = model(x)
            loss = criterion(predicted_y, y)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        valid_x, valid_y = data.valid_data()
        valid_x = make_var(valid_x)
        valid_y = make_var(valid_y)
        predicted_y = model(valid_x)
        valid_loss = criterion(predicted_y, valid_y)

        print('epoch [{}/{}], train loss:{:.4f}, valid loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0], valid_loss.data[0]))

        progress.finish()
    torch.save(model.state_dict(), './conv_autoencoder.pth')

if __name__ == "__main__":
    main()