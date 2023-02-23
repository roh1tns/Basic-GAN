from utils import *


def genBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True)
    )


class Generator(nn.Module):
    def __init__(self, z_dim=64, i_dim=784, h_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            genBlock(z_dim, h_dim),  # 64, 128
            genBlock(h_dim, h_dim * 2),  # 128, 256
            genBlock(h_dim * 2, h_dim * 4),  # 256, 512
            genBlock(h_dim * 4, h_dim * 8),  # 512, 1024
            nn.Linear(h_dim * 8, i_dim),  # 1024, 784 (28x28)
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.gen(noise)


def gen_noise(number, z_dim):
    return torch.randn(number, z_dim).to(device)
