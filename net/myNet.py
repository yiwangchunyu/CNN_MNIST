from torch import nn


class MyNet(nn.Module):
    # nn.Module是一个特殊的nn模块，加载nn.Module，这是为了继承父类
    def __init__(self):
        super(MyNet, self).__init__()
        # super 加载父类中的__init__()函数
        self.VGG_layer = self.vgg_block([64, 'M', 128, 'M', 256,'M', 512, 'M', 512,'M'])
        self.FC_layer = self.fc_block([512,10])
    #前向传播算法
    def forward(self, x):
        out_vgg = self.VGG_layer(x)
        out = out_vgg.view(out_vgg.size(0), -1)
        # 这一步将out拉成out.size(0)的一维向量
        out = self.FC_layer(out)
        return out
    #VGG模块
    def vgg_block(self, cfg_vgg):
        layers = []
        in_channels = 1
        for out_channels in cfg_vgg:
            if out_channels == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
        return nn.Sequential(*layers)
    #全连接模块
    def fc_block(self, cfg_fc):
        fc_net = nn.Sequential()
        fc = []
        for i in range(len(cfg_fc) - 2):
            fc.append(nn.Linear(cfg_fc[i], cfg_fc[i + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.5))
        fc.append(nn.Linear(cfg_fc[-2], cfg_fc[-1]))
        return fc_net(*fc)