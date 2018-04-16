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

        self.enc1 = nn.Linear(300 * 400, 1200)
        self.enc2 = nn.Linear(1200, 600)
        self.enc3 = nn.Linear(600, 400)

        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        self.dec3 = nn.Linear(30, 400)
        self.dec2 = nn.Linear(400, 600)
        self.dec1 = nn.linear(600, 1200)
        self.dec0 = nn.Linear(1200, 300 * 400)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))

        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.relu(self.dec3(z))
        z = self.relu(self.dec2(z))
        z = self.relu(self.dec1(z))
        z = self.relu(self.dec0(z))
        return self.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 300 * 400))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar