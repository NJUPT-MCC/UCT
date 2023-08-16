import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18
from torch.autograd import Variable
from utils import *
import torch.utils.data as data

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        # model_v = resnet50(pretrained=False, last_conv_stride=1, last_conv_dilation=1)

        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):               
                    setattr(self.visible,'layer'+str(i), getattr(model_v,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer'+str(i))(x)
            return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        # model_t = resnet50(pretrained=False, last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net
        
        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):               
                    setattr(self.thermal,'layer'+str(i), getattr(model_t,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):           
                    x = getattr(self.thermal, 'layer'+str(i))(x)             
            return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_net = share_net       
        if self.share_net == 0:
            self.base = model_base
        else:
            self.base = nn.ModuleList()

            if self.share_net > 4:
                pass
            else:
                for i in range(self.share_net, 5):
                    setattr(self.base,'layer'+str(i), getattr(model_base,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            return x
        elif self.share_net > 4:
            return x
        else:
            for i in range(self.share_net, 5):
                x = getattr(self.base, 'layer'+str(i))(x)
            return x



class embed_net(nn.Module):
    def __init__(self,  class_num, args=None):
        super(embed_net, self).__init__()
        self.thermal_module = thermal_module(arch='resnet50', share_net=args.share_net)
        self.visible_module = visible_module(arch='resnet50', share_net=args.share_net)
        self.base_resnet = base_resnet(arch='resnet50', share_net=args.share_net)
        pool_dim = 2048
        local_feat_dim = args.local_feat_dim
        self.feature_dim = 2048
        self.l2norm = Normalize(2)
        self.class_num = class_num
        self.args = args

        # pcb
        self.num_stripes = args.num_strips
        local_conv_out_channels = local_feat_dim

        self.local_conv_list = nn.ModuleList()
        for _ in range(self.num_stripes):
            conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)
            conv.apply(weights_init_kaiming)
            self.local_conv_list.append(nn.Sequential(
                conv,
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))

        self.fc_list = nn.ModuleList()
        for _ in range(self.num_stripes):
            fc = nn.Linear(local_conv_out_channels, class_num)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_list.append(fc)

        # global
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        x = self.base_resnet(x)  #64*2048*18*9

        # global feature
        x_pool = self.avgpool(x)   #gem pooling worth to try
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        feat_global = self.bottleneck(x_pool)

        # pcb
        feat = x
        assert feat.size(2) % self.num_stripes == 0
        stripe_h = int(feat.size(2) / self.num_stripes)
        local_feat_list = []
        logits_list = []
        for i in range(self.num_stripes):
            # gm pool
            local_feat = feat[:, :, i * stripe_h: (i + 1) * stripe_h, :]
            b, c, h, w = local_feat.shape
            local_feat = local_feat.view(b,c,-1)
            p = 3.0 if self.args.dataset == 'sysu' else 10.0  # regDB: 10.0    SYSU: 3.0
            local_feat = (torch.mean(local_feat**p, dim=-1) + 1e-12)**(1/p)

            # shape [N, c, 1, 1]
            local_feat = self.local_conv_list[i](local_feat.view(feat.size(0),feat.size(1),1,1))

            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)

            if hasattr(self, 'fc_list'):
                logits_list.append(self.fc_list[i](local_feat))

        feat_all = [lf for lf in local_feat_list]
        feat_all = torch.cat(feat_all, dim=1)

        return self.l2norm(feat_all)






