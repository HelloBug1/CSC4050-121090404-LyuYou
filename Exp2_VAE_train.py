import numpy as np
import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from VAE_model import Exp2VariationalAutoEncoder


# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 3e-4  # Karpathy's constant

# Load the dataset
transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
dataset = datasets.CIFAR10(root="data", download=True, transform=transforms)
# Select only the images of the first class
indices = np.array(dataset.targets) == 0
dataset.data = dataset.data[indices]
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Exp2VariationalAutoEncoder(channel_in=3, ch=64, latent_channels=512).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss(reduction="sum")

# Training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for idx, (x, _) in loop:
        x = x.to(DEVICE)
        x_reconst, mu, log_var = model(x)

        reconst_loss = loss_fn(x_reconst, x)
        kl_div = -torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        
    # Output image handling
    with torch.no_grad():
        z = torch.randn(64, model.encoder.latent_channels).view(-1, 512, 1, 1).to(DEVICE)
        out = model.decoder(z)
        save_image(out, f"exp2_outputs/{epoch}.png")