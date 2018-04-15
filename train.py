import torch
from torch.autograd import Variable
from torch import nn

from tools import DataLoader
from model import AutoEncoder

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
            #print(x.shape)
            x = Variable(torch.from_numpy(x)).float().cuda()
            y = Variable(torch.from_numpy(y)).float().cuda()
            # ===================forward=====================
            predicted_y = model(x)
            loss = criterion(predicted_y, y)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_autoencoder.pth')

if __name__ == "__main__":
    main()