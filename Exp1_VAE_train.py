import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from VAE_model import Exp1VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


# Configeration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 3e-4 # Karpathy's constant

# Load the dataset
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
indices = dataset.targets == 1
dataset.data = dataset.data[indices]
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = Exp1VariationalAutoEncoder(input_dim=INPUT_DIM, h_dim=H_DIM, z_dim=Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss(reduction="sum")

# Training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for idx, (x, _) in loop:
        x = x.view(-1, INPUT_DIM).to(DEVICE)
        x_reconst, mu, sigma = model(x)
        
        # Compute the loss
        reconst_loss = loss_fn(x_reconst, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # Backpropagation
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        
    with torch.no_grad():
        z = torch.randn(64, Z_DIM).to(DEVICE)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, f"exp1_outputs/{epoch}.png")