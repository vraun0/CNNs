import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=1):
        super(ResBlock, self).__init__()
            
        self.conv1 = nn.Conv2d(in_channels,out_channels, 3,stride = s, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels,3,  padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if s != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x) 
        out = self.relu(out)
        return out
        

class ResNet(nn.Module):
    def __init__(self, n ):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3 ,16 ,3 ,padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace = True)
        self.layers = nn.ModuleList()
        self.out_channels = 16
        for i in range(n):
            self.layers.append(ResBlock(16,16))
        self.layers.append(ResBlock(16,32,2))
        for i in range(n-1):
            self.layers.append(ResBlock(32,32))
        self.layers.append(ResBlock(32,64,2))
        for i in range(n-1):
            self.layers.append(ResBlock(64,64))
        
        
        self.res_network = nn.Sequential(*self.layers)
        self.adaPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res_network(x)
        x = self.adaPool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        

        return x



