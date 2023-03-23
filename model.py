import torch
import torch.nn as nn
import torch. nn.functional as f


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1),
        self.relu1 = nn.ReLU(inplace=True),
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2),
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.relu4 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    net = Net()
    print(net)
