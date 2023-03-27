import torch
from utils import *


warnings.filterwarnings('ignore')

class ReservoirNet(nn.Module):
    def __init__(self, inSize, resSize, a):
        super(ReservoirNet, self).__init__()
        self.inSize = inSize
        self.resSize = resSize
        self.a = a
        self.Win = (torch.rand([self.resSize, 1 + self.inSize]) - 0.5) * 2.4
        self.W = (torch.rand(self.resSize, self.resSize) - 0.5)
        self.Win[abs(self.Win) > 0.6] = 0
        self.rhoW = max(abs(torch.linalg.eig(self.W)[0]))
        self.W *= 1.25 / self.rhoW
        self.reg = 1e-12
        self.one = torch.ones([1, 1])


    def RCPred(self, Wout, RCin):
        T = RCin.size(0)
        X = torch.zeros([1 + self.inSize + self.resSize, T])
        x = torch.zeros((self.resSize, 1))
        for t in range(RCin.size(0)):
            u = RCin[t:t + 1, :].T
            x = (1 - self.a) * x + self.a * sigmoid(torch.matmul(self.Win, torch.vstack((self.one, u))) + torch.matmul(self.W, x))
            X[:, t] = torch.vstack((self.one, u, x))[:, 0]

        pred = Wout @ X
        return pred


    def forward(self, data, labels):
        self.U = data
        self.Yt = labels
        self.T = labels.size(0)
        self.X = torch.zeros([1 + self.inSize + self.resSize, self.T])
        self.x = torch.zeros((self.resSize, 1))

        for t in range(self.U.size(0)):
            self.u = self.U[t:t + 1, :].T
            self.x = (1 - self.a) * self.x + self.a * sigmoid(
                torch.matmul(self.Win, torch.vstack((self.one, self.u))) + torch.matmul(self.W, self.x))
            self.X[:, t] = torch.vstack((self.one, self.u, self.x))[:, 0]

        self.Wout = torch.matmul(torch.matmul(self.Yt.T, self.X.T),
                                 torch.linalg.inv(
                                     torch.matmul(self.X, self.X.T) + self.reg * torch.eye(1 + self.inSize + self.resSize)))

        return self.Wout


class Siamese2RC(nn.Module):
    def __init__(self, inSize, filters):
        super(Siamese2RC, self).__init__()
        self.inSize = inSize
        self.filter1, self.filter2, self.filter3, self.filter4 = filters

        self.layer1 = nn.Sequential(nn.Conv2d(1, self.filter1, kernel_size=3, bias=False),
                                    nn.PReLU(),
                                    nn.MaxPool2d(3, stride=1))

        self.layer2 = nn.Sequential(nn.Conv2d(self.filter1, self.filter2, kernel_size=3, bias=False),
                                    nn.PReLU(),
                                    nn.AvgPool2d(3, stride=1))

        self.layer3 = nn.Sequential(nn.Conv2d(self.filter2, self.filter3, kernel_size=3, bias=False),
                                    nn.PReLU(),
                                    nn.AvgPool2d(3, stride=2))

        self.layer4 = nn.Sequential(nn.Conv2d(self.filter3, self.filter4, kernel_size=3, bias=False),
                                    nn.PReLU())

    def forward_once(self, x):
        x = x.unsqueeze(1).to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, img0, img1):
        img0 = self.forward_once(img0)
        img1 = self.forward_once(img1)

        img0 = img0.view((-1, int(self.inSize / 2)))
        img1 = img1.view((-1, int(self.inSize / 2)))

        out = torch.cat((img0, img1), dim=-1)

        return out