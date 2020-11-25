from torch import nn
from torch.autograd.grad_mode import F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()    # super用法:Cnn继承父类nn.Model的属性，并用父类的方法初始化这些属性
        self.conv = nn.Sequential(     #padding=2保证输入输出尺寸相同(参数依次是:输入深度，输出深度，ksize，步长，填充)
            nn.Conv2d(1, 6, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out