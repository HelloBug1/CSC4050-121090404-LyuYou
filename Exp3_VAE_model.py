import torch
import torch.nn as nn
import torch.nn.functional as F

def output_size(in_size, kernel_size, stride, padding, type='conv'):
    if type == 'conv':
        o_size = (in_size - kernel_size + 2*(padding)) / stride + 1
        if o_size != int(o_size):
            raise ValueError('Invalid configuration')
        return int(o_size)
    elif type == 'deconv':
        o_size = (in_size - 1) * stride - 2*padding + kernel_size
        if o_size != int(o_size):
            raise ValueError('Invalid configuration')
        return int(o_size)
    else:
        raise ValueError('Invalid type')
    
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        input_size = config['input_size']
        ch = config['first_ch']
        latent_channels = config['latent_channels']
        conv1_config = config['conv1']
        conv2_config = config['conv2']
        conv3_config = config['conv3']
        conv4_config = config['conv4']
        conv5_config = config['conv5']

        conv1_o_size = output_size(input_size, conv1_config['kernel_size'], conv1_config['stride'], conv1_config['padding'])
        conv2_o_size = output_size(conv1_o_size, conv2_config['kernel_size'], conv2_config['stride'], conv2_config['padding'])
        conv3_o_size = output_size(conv2_o_size, conv3_config['kernel_size'], conv3_config['stride'], conv3_config['padding'])
        conv4_o_size = output_size(conv3_o_size, conv4_config['kernel_size'], conv4_config['stride'], conv4_config['padding'])
        conv5_o_size = output_size(conv4_o_size, conv5_config['kernel_size'], conv5_config['stride'], conv5_config['padding'])

        self.conv1 = nn.Conv2d(3, 3, kernel_size=conv1_config['kernel_size'], stride=conv1_config['stride'], padding=conv1_config['padding'])
        self.conv2 = nn.Conv2d(3, 3, kernel_size=conv2_config['kernel_size'], stride=conv2_config['stride'], padding=conv2_config['padding'])
        self.conv3 = nn.Conv2d(3, ch, kernel_size=conv3_config['kernel_size'], stride=conv3_config['stride'], padding=conv3_config['padding'])
        self.conv4 = nn.Conv2d(ch, ch*2, kernel_size=conv4_config['kernel_size'], stride=conv4_config['stride'], padding=conv4_config['padding'])
        self.conv5 = nn.Conv2d(ch*2, ch*4, kernel_size=conv5_config['kernel_size'], stride=conv5_config['stride'], padding=conv5_config['padding'])

        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm2d(ch)
        self.bn4 = nn.BatchNorm2d(ch*2)
        self.bn5 = nn.BatchNorm2d(ch*4)

        self.relu = nn.ReLU()

        # Fully-connected layers for the bottleneck
        self.fc_i_size = ch*4 * conv5_o_size * conv5_o_size
        self.fc_mu = nn.Linear(self.fc_i_size, latent_channels)
        self.fc_logvar = nn.Linear(self.fc_i_size, latent_channels)
        
        # Print the output sizes for debugging
        print('Model initialized with the following configuration:')
        print('Conv1 output size:', conv1_o_size)
        print('Conv2 output size:', conv2_o_size)
        print('Conv3 output size:', conv3_o_size)
        print('Conv4 output size:', conv4_o_size)
        print('Conv5 output size:', conv5_o_size)
        print('FC input size:', self.fc_i_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        input_size = config['input_size']
        ch = config['first_ch']
        self.ch = ch
        latent_channels = config['latent_channels']
        conv1_config = config['conv1']
        conv2_config = config['conv2']
        conv3_config = config['conv3']
        conv4_config = config['conv4']
        conv5_config = config['conv5']
        
        conv1_o_size = output_size(input_size, conv1_config['kernel_size'], conv1_config['stride'], conv1_config['padding'])
        conv2_o_size = output_size(conv1_o_size, conv2_config['kernel_size'], conv2_config['stride'], conv2_config['padding'])
        conv3_o_size = output_size(conv2_o_size, conv3_config['kernel_size'], conv3_config['stride'], conv3_config['padding'])
        conv4_o_size = output_size(conv3_o_size, conv4_config['kernel_size'], conv4_config['stride'], conv4_config['padding'])
        conv5_o_size = output_size(conv4_o_size, conv5_config['kernel_size'], conv5_config['stride'], conv5_config['padding'])
        self.conv5_o_size = conv5_o_size
        self.linear_o_size = ch*4 * conv5_o_size * conv5_o_size
        
        deconv1_config = config['deconv1']
        deconv2_config = config['deconv2']
        deconv3_config = config['deconv3']
        deconv4_config = config['deconv4']
        deconv5_config = config['deconv5']
        out_config = config['out']
        
        deconv1_o_size = output_size(conv5_o_size, deconv1_config['kernel_size'], deconv1_config['stride'], deconv1_config['padding'], type='deconv')
        deconv2_o_size = output_size(deconv1_o_size, deconv2_config['kernel_size'], deconv2_config['stride'], deconv2_config['padding'], type='deconv')
        deconv3_o_size = output_size(deconv2_o_size, deconv3_config['kernel_size'], deconv3_config['stride'], deconv3_config['padding'], type='deconv')
        deconv4_o_size = output_size(deconv3_o_size, deconv4_config['kernel_size'], deconv4_config['stride'], deconv4_config['padding'], type='deconv')
        deconv5_o_size = output_size(deconv4_o_size, deconv5_config['kernel_size'], deconv5_config['stride'], deconv5_config['padding'], type='deconv')
        
        self.fc = nn.Linear(latent_channels, self.linear_o_size)
        self.deconv1 = nn.ConvTranspose2d(ch*4, ch*2, kernel_size=deconv1_config['kernel_size'], stride=deconv1_config['stride'], padding=deconv1_config['padding'])
        self.deconv2 = nn.ConvTranspose2d(ch*2, ch*1, kernel_size=deconv2_config['kernel_size'], stride=deconv2_config['stride'], padding=deconv2_config['padding'])
        self.deconv3 = nn.ConvTranspose2d(ch, 6, kernel_size=deconv3_config['kernel_size'], stride=deconv3_config['stride'], padding=deconv3_config['padding'])
        self.deconv4 = nn.ConvTranspose2d(6, 6, kernel_size=deconv4_config['kernel_size'], stride=deconv4_config['stride'], padding=deconv4_config['padding'])
        self.deconv5 = nn.ConvTranspose2d(6, 6, kernel_size=deconv5_config['kernel_size'], stride=deconv5_config['stride'], padding=deconv5_config['padding'])

        self.bn1 = nn.BatchNorm2d(ch*2)
        self.bn2 = nn.BatchNorm2d(ch)
        self.bn3 = nn.BatchNorm2d(6)
        self.bn4 = nn.BatchNorm2d(6)
        self.bn5 = nn.BatchNorm2d(6)

        self.out1 = nn.ConvTranspose2d(6, 3, kernel_size=out_config['kernel_size'], stride=out_config['stride'], padding=out_config['padding'])
        self.out2 = nn.ConvTranspose2d(6, 3, kernel_size=out_config['kernel_size'], stride=out_config['stride'], padding=out_config['padding'])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Print the output sizes for debugging
        print('Model initialized with the following configuration:')
        print('Deconv1 output size:', deconv1_o_size)
        print('Deconv2 output size:', deconv2_o_size)
        print('Deconv3 output size:', deconv3_o_size)
        print('Deconv4 output size:', deconv4_o_size)
        print('Deconv5 output size:', deconv5_o_size)
        if deconv5_o_size != input_size:
            raise ValueError('Invalid configuration')

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.ch*4, self.conv5_o_size, self.conv5_o_size)
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        o1 = self.sigmoid(self.out1(x))
        o2 = self.sigmoid(self.out2(x))
        return o1, o2


class Exp3VariationalAutoEncoder(nn.Module):
    def __init__(self, config):
        super(Exp3VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Calculate the number of parameters
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(f'Model initialized with {encoder_params + decoder_params} ({(encoder_params+decoder_params)/1e6:.2f}M) parameters')

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    