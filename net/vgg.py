from torch import nn

#定义VGG各种不同的结构和最后的全连接层结构
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256,'M', 512, 'M', 512,'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'FC':    [512*7*7, 4096, 10],
    'VGG11_FC':    [512, 10]
}
#将数据展开成二维数据，用在全连接层之前和卷积层之后
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

class VGG(nn.Module):
    # nn.Module是一个特殊的nn模块，加载nn.Module，这是为了继承父类
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        # super 加载父类中的__init__()函数
        self.VGG_layer = self.vgg_block(cfg[vgg_name])
        self.FC_layer = self.fc_block(cfg['VGG11_FC'])
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

        # fc_features, fc_hidden_units, fc_output_units = cfg_fc[0:]
        # fc_net.add_module("fc", nn.Sequential(
        #     # FlattenLayer(),
        #     nn.Linear(fc_features, fc_hidden_units),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(fc_hidden_units, fc_hidden_units),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(fc_hidden_units, fc_output_units)
        # ))
        # return fc_net
        fc = []
        for i in range(len(cfg_fc) - 2):
            fc.append(nn.Linear(cfg_fc[i], cfg_fc[i + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.5))
        fc.append(nn.Linear(cfg_fc[-2], cfg_fc[-1]))
        return fc_net(*fc)