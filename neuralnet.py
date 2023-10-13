import torch
from torch import nn

# Set the seed
seed = 42
torch.manual_seed(seed)

class AlexNetLike(nn.Module):
    def __init__(self, input_dim, output_dim, filter_len, dropout = 0.5, pretrained = False):
        super().__init__()
        c, h, w = input_dim

        nf = 16

        self.relu = torch.nn.ReLU()
        self.flat = nn.Flatten(1)
        self.pool = nn.MaxPool2d(2)
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=nf, kernel_size=filter_len, padding = 'same', stride=1)
        self.batch_norm1 = nn.BatchNorm2d(nf)
        
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=2*nf, kernel_size=filter_len, padding = 'same', stride=1)
        self.batch_norm2 = nn.BatchNorm2d(2*nf)

        self.batch_norm = nn.BatchNorm1d(3)

        ww = int(w/2**2)
        hh = int(h/2**2)

        self.fc1 = nn.Linear(2*nf*ww*hh + 3, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.drop = nn.Dropout(p = dropout)

        if pretrained:
            self.conv1.requires_grad_(False)
            self.conv2.requires_grad_(False)
            self.fc1.requires_grad_(False)

            
    def forward(self, state, scale):
        state = state/3.

        x = self.conv1(state)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)

        x = self.flat(x)

        scale = self.batch_norm(scale)
        x = torch.cat((x, scale), dim = 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.fc3(x)

        return x
