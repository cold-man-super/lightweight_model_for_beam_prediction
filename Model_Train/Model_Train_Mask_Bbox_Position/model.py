import shutil
import pandas as pd
from torch.utils.data import Dataset
import datetime
import torch as t
import torch.optim as optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as transf
import numpy as np
from cbam import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1)
        self.cbam1 = CBAM(6, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.cbam2 = CBAM(16, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 32)
        self.fc4 = nn.Linear(38,128)
        self.fc5 = nn.Linear(128,256)
        self.fc6 = nn.Linear(256,512)
        self.fc7 = nn.Linear(512, 256)
        self.fc8 = nn.Linear(256,128)
        self.fc9 = nn.Linear(128,32)
    def forward(self, img, bbox_pos):
        img = F.max_pool2d(F.relu(self.conv1(img)), (2, 2))
        img = F.relu(self.cbam1(img))
        img = F.max_pool2d(F.relu(self.conv2(img)), (2, 2))
        img = F.relu(self.cbam2(img))
        img = img.view(-1, self.num_flat_features(img))
        img = F.relu(self.fc1(img))
        img = F.relu(self.fc2(img))
        img = F.relu(self.fc3(img))
        bbox_pos = bbox_pos.view(-1, self.num_flat_features(bbox_pos))
        concatenated = torch.cat((img, bbox_pos), dim=1)
        concatenated = F.relu(self.fc4(concatenated))
        concatenated = F.relu(self.fc5(concatenated))
        concatenated = F.relu(self.fc6(concatenated))
        concatenated = F.relu(self.fc7(concatenated))
        concatenated = F.relu(self.fc8(concatenated))
        output = self.fc9(concatenated)
        return output
    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)