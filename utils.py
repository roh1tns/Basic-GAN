# Importing the libraries
import torch, pdb
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# Visualization Function
def show(tensor, ch=1, size=(28, 28), num=16):
    # Incoming tensor is of the form 128(batch size) x 784(28x28=784)
    data = tensor.detach().cpu().view(-1, ch, *size)  # Converts the tensor to 1 x 28 x 28 (ch x width x height)
    grid = make_grid(data[:num], nrow=4).permute(1, 2, 0)  # Matplotlib requires width x height x ch (28 x 28 x 1)
    plt.imshow(grid)
    plt.show()


# Setup of the main parameters and hyperparameters
epochs = 500
cur_step = 0
info_step = 300
mean_gen_loss = 0
mean_disc_loss = 0

z_dim = 64
lr = 0.00001
loss_func = nn.BCEWithLogitsLoss()

bs = 128  # Batch Size
device = 'cpu'

dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), shuffle=True, batch_size=bs)


# Number of steps = Number of images in dataset / Batch Size = 60000 / 128
