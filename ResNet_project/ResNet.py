import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import pdb


class baseBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels, stride=1, dim_change=None):
        super(baseBlock, self).__init__()
        # declare convolutional layers with batch norms
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, stride=stride, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channels)
        self.conv2 = torch.nn.Conv2d(output_channels, output_channels, stride=1, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channels)
        self.dim_change = dim_change

    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res

        return output


class bottleNeck(torch.nn.Module):
    expansion = 4

    def __init__(self, input_channels, output_channels, stride=1, dim_change=None):
        super(bottleNeck, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channels)
        self.conv2 = torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channels)
        self.conv3 = torch.nn.Conv2d(output_channels, output_channels * self.expansion, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channels * self.expansion)
        self.dim_change = dim_change

    def forward(self, x):
        res = x

        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)
        return output


class ResNet(torch.nn.Module):

    def __init__(self, block, num_layers, classes=10):
        super(ResNet, self).__init__()

        self.input_channels = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._layer(block, 64, num_layers[0], stride=1)
        self.layer2 = self._layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._layer(block, 512, num_layers[3], stride=2)
        self.averagePool = torch.nn.AvgPool2d(512 * block.expansion, classes)
        self.fc = torch.nn.Linear(512 * block.expansion, classes)

    def _layer(self, block, output_channels, num_layers, stride=1):
        dim_change = None

        if stride == 1 or output_channels != self.input_channels * block.expansion:
            dim_change = torch.nn.Sequential(torch.nn.Conv2d(self.input_channels, output_channels * block.expansion,
                                                             kernel_size=1, stride=stride),
                                             torch.nn.BatchNorm2d(output_channels * block.expansion))
            netlayers = []
            netlayers.append(block(self.input_channels, output_channels, stride=stride, dim_change=dim_change))
            self.input_channels = output_channels * block.expansion
            for i in range(1, num_layers):
                netlayers.append(block(self.input_channels, output_channels))
                self.input_channels = output_channels * block.expansion

            return torch.nn.Sequential(*netlayers)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def test():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)

    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testset = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # ResNet-18
    net = ResNet(baseBlock, [2, 2, 2, 2], 10)
    net.to(device)
    costFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9)

    for epoch in range(50):
        closs = 0
        for i, batch in enumerate(trainset, 0):
            data, output = batch
            data, output = data.to(device), output.to(device)
            prediction = net(data)
            loss = costFunc(prediction, output)
            closs = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"[{epoch + 1}, {i + 1}], loss = {closs / 1000}")
                closs = 0

        correctHits = 0
        total = 0
        for batches in testset:
            data, output = batches
            data, output = data.to(device), output.to(device)
            prediction = net(data)
            _, prediction = torch.max(prediction.data, 1)
            total += output.size(0)
            correctHits += (prediction == output).sum().item()
        print(f"Accuracy = {(correctHits / total) * 100} %")


if __name__ == '__main__':
    test()
