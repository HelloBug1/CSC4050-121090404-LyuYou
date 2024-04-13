import torch
import torch.nn as nn
import torch.nn.functional as F


class Exp1VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20) -> None:
        super().__init__()
        
        # Encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # Decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.z_2hid(z))
        x = torch.sigmoid(self.hid_2img(h))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        x_reconst = self.decode(z)
        return x_reconst, mu, sigma
    
    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(sigma)
        return mu + eps * sigma


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 128x128
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 64x64
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x32
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16x16
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8x8
        self.bn5 = nn.BatchNorm2d(256)
        
        # Fully-connected layers for the bottleneck
        self.fc_mu = nn.Linear(256 * 8 * 8, 20)
        self.fc_logvar = nn.Linear(256 * 8 * 8, 20)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(-1, 256 * 8 * 8)  # Flatten the output
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(20, 256 * 8 * 8)  # Match the output size of Encoder's last layer
        
        # Deconvolutional blocks
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 16x16
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 64x64
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 128x128
        self.bn4 = nn.BatchNorm2d(16)
        self.deconv5 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)  # 256x256
        self.bn5 = nn.BatchNorm2d(8)
        
        # Output layers
        self.out1 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1)
        self.out2 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)  # Reshape for deconvolution
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.relu(self.bn5(self.deconv5(x)))
        return torch.sigmoid(self.out1(x)), torch.sigmoid(self.out2(x))


class Exp2VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(Exp2VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar



if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    model = Exp2VariationalAutoEncoder(channel_in=3, ch=64, latent_channels=512)
    print(model(x)[0].shape)
    print(model(x)[1].shape)
    print(model(x)[2].shape)
