import torch.nn as nn


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64) 
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64) 

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128) 
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128) 

        self.conv5_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(256) 
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(256) 
        self.conv5_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(256) 

        self.conv7_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(512)  
        self.conv7_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(512) 
        self.conv7_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn7_3 = nn.BatchNorm2d(512) 


        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)

        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        x = self.relu(self.bn5_3(self.conv5_3(x)))
        x = self.pool(x)

        x = self.relu(self.bn7_1(self.conv7_1(x)))
        x = self.relu(self.bn7_2(self.conv7_2(x)))
        x = self.relu(self.bn7_3(self.conv7_3(x)))
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x


