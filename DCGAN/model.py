import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__() #The notations on the side of each Conv2d is the output size of each convolution in the format C H W.
        self.modelCNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels= 64, kernel_size = 4, padding = 1, stride =2  ), # 64 32 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels= 128, kernel_size = 4, padding = 1, stride = 2 ), # 128 16 16
            nn.BatchNorm2d(128, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),          
            nn.Conv2d(in_channels=128, out_channels= 256, kernel_size = 4, padding = 1, stride = 2 ), # 256 8 8
            nn.BatchNorm2d(256, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels= 512, kernel_size = 4, padding = 1, stride = 2 ), # 512 4 4
            nn.BatchNorm2d(512, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels= 1, kernel_size = 4, padding = 0, stride = 1 ), # 512 1 1
            nn.Sigmoid()
        )      
    def forward(self, x):
        x = self.modelCNN(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__() #The notations on the side of each Conv2d is the output size of each convolution in the format C H W.
        self.model = nn.Sequential( # 256 100 1 1
            nn.ConvTranspose2d(in_channels=100, out_channels=256, kernel_size=4, stride=1, padding=0), #512 4 4
            nn.BatchNorm2d(256),
            nn.ReLU(),          
            nn.ConvTranspose2d(256, 512, 4, 2, 1), #512 8 8 
            nn.BatchNorm2d(512),
            nn.ReLU(), 
            nn.ConvTranspose2d(512, 256, 4, 2, 1), #256 16 16 
            nn.BatchNorm2d(256),
            nn.ReLU(),                                                                                        
            nn.ConvTranspose2d(256, 64, 4, 2, 1), # 64 32 32
            nn.BatchNorm2d(64),
            nn.ReLU(),          
            nn.ConvTranspose2d(64, 1, 4, 2, 1), # 3 64 64                                         
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x)
        return output
