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
            nn.Conv2d(1, 24, 11, stride=4, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.Conv2d(24, 16, 5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        self.size = 16 * 69 * 94
        self.latent_size = 64

        self.post_encoder = nn.Linear(self.size, self.latent_size)
        self.post_encoder_bn = nn.BatchNorm1d(self.latent_size)

        self.fc1 = nn.Linear(self.latent_size, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, self.latent_size)

        self.pre_decoder = nn.Linear(self.latent_size, self.size)
        self.pre_decoder_bn = nn.BatchNorm2d(self.size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 6, stride=5),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, 8, stride=5, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=1, padding=0),
            nn.Tanh()
        )

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.encoder(x)
        print("Post encoder shape: {}".format(x.size()))
        x = self.relu(self.post_encoder_bn(self.post_encoder(x.view(-1, self.size))))
        return self.fc1(x), self.fc2(x)

    def reparameterize(self, mu, logvar):
        return mu
        #if self.training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
        #else:
        #return mu

    def decode(self, z):
        z = self.relu(self.pre_decoder_bn(self.pre_decoder(z))).view(-1, 16, 69, 94)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
