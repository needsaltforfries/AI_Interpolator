import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable

class IDiscriminator(nn.Module):
    def __init__(self):
        super(IDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AvgPool2d(2),
            
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.Sigmoid(),
            
        )
        self.linear = nn.Sequential(
            nn.Linear(512*256*1*1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.net(x)
        #flatten
        x = x.view(-1, 512*256*1*1)
        x = self.linear(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, batch_size, IHeight, IWidth):
        super(Generator, self).__init__()
        self.IHeight = IHeight
        self.IWidth = IWidth
        #64
        self.e1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        #128
        self.e2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
        )
        #256
        self.e3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        #512
        self.e4 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        #1024
        self.e5 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Tanh(),
        )
        #1024
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        #512
        self.d2 = nn.Sequential(

            nn.ConvTranspose2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),  
        )
        #256
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        #128
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, 1, 1),
            nn.LeakyReLU(inplace=True),     
        )
        #64
        self.d5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.d6 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.down = nn.AvgPool2d(2)
        
    def forward(self, x):
        e1 = self.e1(x)
        # print('e1 ', e1.shape)
        e2 = self.e2(e1)
        # print('e2 ', e2.shape)
        e3 = self.e3(e2)
        # print('e3 ', e3.shape)
        e4 = self.e4(e3)
        # print('e4 ', e4.shape)
        e5 = self.e5(e4)
        # print('e5 ', e5.shape)
        d1 = self.d1(e5)
        # print('d1 ', d1.shape)
        d1 = torch.cat([d1, e4], dim=1) 
        d1 = self.up(d1)
        # print("d1 cat ", d1.shape)
        d2 = self.d2(d1)
        # print('d2', d2.shape)
        d2 = torch.cat([d2, e3], dim=1)
        e3 = self.up(e3)
        d2 = self.up(d2)
        # print("d2 cat ", d2.shape)
        d3 = self.d3(d2)
        # print('d3', d3.shape)
        d3 = self.up(d3)
        e2 = self.up(e2)
        d3 = torch.cat([d3, e2], dim=1)
        # print("d3 cat ", d3.shape)
        d4 = self.d4(d3)
        d4 = self.up(d4)
        # print('d4', d4.shape)
        e1 = self.up(e1)
        d4 = torch.cat([d4, e1], dim=1)
        # print("d4 cat ", d4.shape)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d6 = self.down(d6)
        # print(d6.shape)
        x = d6
        return x