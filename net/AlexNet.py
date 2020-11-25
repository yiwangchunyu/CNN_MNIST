#定义网络结构
from torch import nn
from torch.autograd.grad_mode import F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()

        # 由于MNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #AlexCONV1(3,96, k=11,s=4,p=0)
            nn.MaxPool2d(kernel_size=2, stride=2),#AlexPool1(k=3, s=2)
            nn.ReLU(),

        # self.conv2 = nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),#AlexCONV2(96, 256,k=5,s=1,p=2)
            nn.MaxPool2d(kernel_size=2,stride=2),#AlexPool2(k=3,s=2)
            nn.ReLU(),


            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),#AlexCONV3(256,384,k=3,s=1,p=1)
        # self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),#AlexCONV4(384, 384, k=3,s=1,p=1)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),#AlexCONV5(384, 256, k=3, s=1,p=1)
            nn.MaxPool2d(kernel_size=2, stride=2),#AlexPool3(k=3,s=2)
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),  #AlexFC6(256*6*6, 4096)
            nn.ReLU(),
            nn.Linear(1024, 512), #AlexFC6(4096,4096)
            nn.ReLU(),
            nn.Linear(512, 10),  #AlexFC6(4096,1000)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1, 256 * 3 * 3)#Alex: x = x.view(-1, 256*6*6)
        x = self.fc(x)
        return x
