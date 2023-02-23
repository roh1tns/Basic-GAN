from utils import *


def discBlock(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(0.2)
    )


class Discriminator(nn.Module):
    def __init__(self, i_dim=784, h_dim=256):
        super().__init__()
        self.disc = nn.Sequential(
            discBlock(i_dim, h_dim * 4),  # 784, 1024
            discBlock(h_dim * 4, h_dim * 2),  # 1024, 512
            discBlock(h_dim * 2, h_dim),  # 512, 256
            nn.Linear(h_dim, 1)  # 256, 1
        )

    def forward(self, image):
        return self.disc(image)
