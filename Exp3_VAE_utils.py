import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from scipy.stats import norm
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from statsmodels.stats.multitest import multipletests

from Exp3_VAE_model import Exp3VariationalAutoEncoder

# BraTS2020 Data files structure
# data
# ├── test
# │   ├── healthy_test_images.npy
# │   ├── unhealthy_test_images.npy
# ├── |── unhealthy_test_masks.npy
# ├── train
# │   ├── healthy_train_images.npy

class BraTS2020Dataset(Dataset):
    def __init__(self, root_dir, train=True, healthy=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (bool): True to load training data, False for test data.
            healthy (bool): True to load healthy images, False for unhealthy.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.healthy = healthy
        self.transform = transform
        self.data = None
        self.load_data(root_dir)

    def load_data(self, root_dir):
        # Construct file path based on the provided arguments
        data_type = 'train' if self.train else 'test'
        health_status = 'healthy' if self.healthy else 'unhealthy'
        file_name = f'{health_status}_{data_type}_images.npy'
        print(f"Loading {data_type} {health_status} data from {file_name}")
        
        # Load the data
        file_path = f'{root_dir}/{data_type}/{file_name}'
        print(f"Loading data from: {file_path}")
        self.data = np.load(file_path)

        # Check the data shape
        print(f"Loaded {data_type} data with shape: {self.data.shape}")
        
        # For unhealthy test data, we could also load masks here if needed
        if not self.train and not self.healthy:
            mask_file_path = f'{root_dir}/{data_type}/unhealthy_test_masks.npy'
            self.masks = np.load(mask_file_path)
        else:
            self.masks = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        # Check the image shape
        # print(f"Image shape: {image.shape}")
        
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image}
        
        # Include masks in the sample if this is unhealthy test data
        if self.masks is not None:
            mask = self.masks[idx]
            if self.transform:
                mask = self.transform(mask)
            sample['mask'] = mask

        return sample

def load_dataset(data_dir, dataset_class, batch_size=32, train=True, healthy=True, shuffle=True):
    # Define the transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad((8, 8), fill=0, padding_mode='constant'),
    ])
    # Load the dataset
    dataset = dataset_class(data_dir, train=train, healthy=healthy, transform=data_transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    return data_loader

def get_train_val_dataloader(dataset, fold_idx, train_batch_size, val_batch_size, k=5, single_transfer=True):
    indices = list(range(len(dataset)))
    
    val_split = int(np.floor(len(dataset) / k))
    train_indices, val_indices = indices[:fold_idx * val_split] + indices[(fold_idx + 1) * val_split:], indices[fold_idx * val_split:(fold_idx + 1) * val_split]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = prepare_dataloader(train_dataset, train_batch_size, shuffle=True, single_transfer=single_transfer)
    val_loader = prepare_dataloader(val_dataset, val_batch_size, shuffle=False, single_transfer=single_transfer)
    del dataset
    return train_loader, val_loader

def prepare_dataloader(dataset, batch_size, shuffle, single_transfer):
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=0, pin_memory=True)
    if single_transfer:
        # Preload all data to GPU at once
        all_inputs = []
        for inputs in dataloader:
            all_inputs.append(inputs['image'].to('cuda:0'))
        return all_inputs
    else:
        return dataloader

def qr_loss(x, q1, q2, mu, logvar, qval1=0.15, qval2=0.5):
    # Quantile Regression Loss
    q1_loss = torch.sum(torch.max(qval1 * (x - q1), (qval1 - 1) * (x - q1)))
    q2_loss = torch.sum(torch.max(qval2 * (x - q2), (qval2 - 1) * (x - q2)))
    recon_loss = q1_loss + q2_loss
    kld_loss = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def calculate_rejection_mask(images, mean_recon, std_recon, threshold=0.05):
    """
    Calculate the rejection mask for each image based on the reconstruction error,
    rejecting a pixel if two or more channels are rejected.

    Args:
        images (Tensor): The original images tensor.
        mean_recon (Tensor): The mean of the reconstructed images.
        std_recon (Tensor): The standard deviation of the reconstruction errors.
        output_size (tuple): The spatial size of the output images.
        threshold (float): The significance level for rejecting the null hypothesis.

    Returns:
        Tensor: A tensor of rejection masks shaped according to output_size.
    """
    # Adjust standard deviation to avoid division by zero
    std_recon_adjusted = torch.where(std_recon == 0, torch.tensor(1e-8, device=std_recon.device), std_recon)
    
    # Calculate z-scores
    z_scores = (images - mean_recon) / std_recon_adjusted
    
    # Calculate p-values based on z-scores, keeping data on the same device
    p_values = 2 * torch.distributions.Normal(0, 1).cdf(-torch.abs(z_scores))
    
#     # Calculate the mean for each voxel
#     p_values = torch.mean(p_values, dim=1)
    p_values = p_values.cpu().numpy()
    
    reject_mask = np.zeros_like(p_values)
    # Apply multiple testing correction
    reject, _, _, _ = multipletests(p_values.flatten(), alpha=threshold, method='fdr_bh')
    reject_mask = reject.reshape(p_values.shape)
    reject_mask = np.sum(reject_mask, axis=1) >= 2

    return reject_mask[:,np.newaxis,:,:]

def train_qr_vae_incremental(
        device,
        model,
        train_loader, 
        val_loader, 
        optimizer, 
        epochs=10, 
        save_checkpoint=True, 
        checkpoint_dic_path='checkpoints/', 
        checkpoint_range=(100, 200), 
        checkpoint_interval=1, 
        single_transfer=True,
        mixed_precision=False
    ):
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Load checkpoint if exists
    checkpoint_path = f'{checkpoint_dic_path}/checkpoint.pt'
    losses_path = f'{checkpoint_dic_path}/losses.pt'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer's state_dict to the correct device after loading
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch} with best loss {best_val_loss:.4f}")
        if os.path.exists(losses_path):
            losses_checkpoint = torch.load(losses_path)
            train_losses = losses_checkpoint['train_losses'][0:start_epoch]
            val_losses = losses_checkpoint['val_losses'][0:start_epoch]
            avg_val_loss = val_losses[-1]
            print(f"Loaded losses from file.")

    model.to(device)
    if mixed_precision:
        scaler = GradScaler()
        print("Mixed precision training enabled.")
    
    print("Training started.")
    for epoch in (pb := tqdm(range(start_epoch, epochs))):
        model.train()
        # Training
        avg_train_loss = 0
        length = 0
        for train_sample in train_loader:
            if not single_transfer:
                train_data = train_sample['image'].to(device)
            else:
                train_data = train_sample
            optimizer.zero_grad()
            if mixed_precision:
                with autocast():
                    recon_batch, mu, logvar = model(train_data)
                    train_loss = qr_loss(train_data, recon_batch[0], recon_batch[1], mu, logvar)
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon_batch, mu, logvar = model(train_data)
                train_loss = qr_loss(train_data, recon_batch[0], recon_batch[1], mu, logvar)
                train_loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            avg_train_loss += train_loss.item()
#             print(f"avg_train_loss: {avg_train_loss}")
            length += len(train_data)
#             print(f"len(train_data): {len(train_data)}")
        avg_train_loss /= length
        train_losses.append(avg_train_loss)

        if epoch % checkpoint_interval == 0:
            avg_val_loss = 0
            data_length = 0
            for val_sample in val_loader:
                if not single_transfer:
                    val_data = val_sample['image'].to(device)
                else:
                    val_data = val_sample
                # Validation
                data_length += len(val_data)
                if mixed_precision:
                    with autocast():
                        avg_val_loss += validate_qr_vae_pinball_loss(model, val_data)
                else:
                    avg_val_loss += validate_qr_vae_pinball_loss(model, val_data)
            avg_val_loss /= data_length

        # Checkpoint saved if the loss is improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_checkpoint and epoch >= checkpoint_range[0] and epoch <= checkpoint_range[1] and epoch % checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, checkpoint_path)
                torch.save({'train_losses': train_losses, 'val_losses': val_losses}, losses_path)
                # print("Checkpoint saved.")
        val_losses.append(avg_val_loss)
        pb.set_postfix(train_loss=f'{avg_train_loss:.4f}', val_loss=f'{avg_val_loss:.4f}', best_val_loss=f'{best_val_loss:.4f}')

    print("Training completed.")

def train_qr_vae_incremental_kfold(
        device, 
        config,
        dataset, 
        epochs=10, 
        train_batch_size=32, 
        val_batch_size=32,
        k_fold = 5,
        lr=3e-4,
        save_checkpoint=True, 
        checkpoint_dic_path='checkpoints/', 
        checkpoint_ranges=None, 
        checkpoint_interval=1,
        single_transfer=True,
        mixed_precision=False
    ):
    os.makedirs(checkpoint_dic_path, exist_ok=True)
    for fold_idx in range(k_fold):
        train_loader, val_loader = get_train_val_dataloader(
            dataset, fold_idx, train_batch_size=train_batch_size, val_batch_size=val_batch_size, k=5, single_transfer=single_transfer
        )

        model = Exp3VariationalAutoEncoder(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Clear the gradients before training
        optimizer.zero_grad()

        checkpoint_dic_path_fold = f'{checkpoint_dic_path}/fold_{fold_idx}'
        os.makedirs(checkpoint_dic_path_fold, exist_ok=True)

        if checkpoint_ranges[fold_idx] == None:
            checkpoint_ranges[fold_idx] = (0, epochs)

        train_qr_vae_incremental(
            device, 
            model, 
            train_loader, 
            val_loader, 
            optimizer, 
            epochs, 
            save_checkpoint, 
            checkpoint_dic_path_fold, 
            checkpoint_ranges[fold_idx], 
            checkpoint_interval,
            single_transfer,
            mixed_precision
        )
    
def validate_qr_vae_pinball_loss(model, val_data):
    model.eval()

    with torch.no_grad():
        # recon_batch is expected to contain quantiles and mean as follows:
        # recon_batch[0] = 0.15 quantile
        # recon_batch[1] = median
        recon_batch, mu, logvar = model(val_data)
        val_loss = qr_loss(val_data, recon_batch[0], recon_batch[1], mu, logvar)
        val_loss = val_loss.item()
        
        return val_loss

def load_and_plot_losses(start, end, checkpoint_dic_path='checkpoints/', k_fold=5):
    for fold_idx in range(k_fold):
        checkpoint_dic_path_fold = f'{checkpoint_dic_path}/fold_{fold_idx}'
        losses_path = f'{checkpoint_dic_path_fold}/losses.pt'
        checkpoint = torch.load(losses_path)
        train_losses = checkpoint['train_losses'][start:end]
        val_losses = checkpoint['val_losses'][start:end]
        # Print the best validation loss and the corresponding epoch
        print(f"Fold {fold_idx} Best Validation Loss: {min(val_losses):.4f} at Epoch {np.argmin(val_losses) + start}")
        # Plot the losses in different subplots
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
        