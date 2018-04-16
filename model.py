import torch
from torch.autograd import Variable
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 8, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(8, 16, 6, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 6, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, 8, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 8, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(8, 16, 6, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1).view(-1, 16 * 286 * 386),
            nn.Linear(16 * 286 * 386, 512),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 6, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, 8, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=1, padding=0),
            nn.Tanh()
        )

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        print(x.size())
        x = self.encoder(x)
        return self.fc1(x), self.fc2(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        print(z.size())
        z = self.decoder(z)
        return self.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar