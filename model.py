import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def upsample_x2(bilinear=False, in_ch=None, out_ch=None, bias=False):
    if bilinear:
        return torch.nn.Upsample(scale_factor=2, mode='bilinear')
    else:
        return torch.nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=bias)


class Double_conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, leaky=0, bias=True):
        super(Double_conv, self).__init__()
        self.layer = []
        self.layer.append(torch.nn.Conv2d(in_ch, out_ch, 3, 1, padding=1, bias=bias))
        self.layer.append(torch.nn.BatchNorm2d(out_ch))
        self.layer.append(torch.nn.ReLU(inplace=True) if not leaky else torch.nn.LeakyReLU(leaky, inplace=True))
        self.layer.append(torch.nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=bias))
        self.layer.append(torch.nn.BatchNorm2d(out_ch))
        self.layer.append(torch.nn.ReLU(inplace=True) if not leaky else torch.nn.LeakyReLU(leaky, inplace=True))
        self.layer = torch.nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)

class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)

    def forward(self, x):
        return self.l(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class AppearanceComposability_localnonlocal(torch.nn.Module):
    def __init__(self, k, padding, stride):  # 9 4 1
        super(AppearanceComposability_localnonlocal, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)

    def forward(self, x):
        key_map, query_map = x  # [4, 64, 64, 64]
        k = self.k
        key_map_unfold = self.unfold(key_map).transpose(2, 1).contiguous()  # [N batch , H_out*Wout, channel * k*k]  [4, 4096, 5184]
        query_map_unfold = self.unfold(query_map).transpose(2,1).contiguous()  # [N batch , H_out*Wout, C channel * k*k]
        key_map_unfold = key_map_unfold.view(key_map.shape[0], -1, key_map.shape[1],
                                             key_map_unfold.shape[-1] // key_map.shape[1])  # [N batch, H*W, C/m, k*k] [4, 4096, 64, 81]
        query_map_unfold = query_map_unfold.view(query_map.shape[0], -1, query_map.shape[1],
                                                 query_map_unfold.shape[-1] // query_map.shape[1])  # [N batch, H*W, C/m, k*k] [4, 4096, 64, 81]

        key_map_unfold = key_map_unfold.transpose(2, 3).contiguous()  # [N batch, H*W, k*k, C/m]
        # query_map_unfold = query_map_unfold.transpose(2,3).contiguous()
        return torch.matmul(key_map_unfold, query_map_unfold[:, :, :, k ** 2 // 2:k ** 2 // 2 + 1])  #高维矩阵相乘 qmap取中心点  [4, 4096, 81, 1]


class Inception_LocalNonLocal_v2(torch.nn.Module):
    def __init__(self, channels, k1, k2, k3, k4, m=None, stride=1):
        super(Inception_LocalNonLocal_v2, self).__init__()
        self.channels = channels
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.stride = stride
        self.m = m or 8

        self.kmap = KeyQueryMap(channels, self.m)  #1x1卷积 通道数减少为 除m
        self.qmap = KeyQueryMap(channels, self.m)
        self.xmap = KeyQueryMap(channels, self.m)
        self.ac1 = AppearanceComposability_localnonlocal(k1, k1 // 2, self.stride)
        self.ac2 = AppearanceComposability_localnonlocal(k2, k2 // 2, self.stride)
        self.ac3 = AppearanceComposability_localnonlocal(k3, k3 // 2, self.stride)
        self.ac4 = AppearanceComposability_localnonlocal(k4, k4 // 2, self.stride)

        self.unfold1 = torch.nn.Unfold(k1, 1, k1 // 2, self.stride)
        self.unfold2 = torch.nn.Unfold(k2, 1, k2 // 2, self.stride)
        self.unfold3 = torch.nn.Unfold(k3, 1, k3 // 2, self.stride)
        self.unfold4 = torch.nn.Unfold(k4, 1, k4 // 2, self.stride)
        self.final1x1 = torch.nn.Conv2d(channels // m, channels, 1)
        self.bn = torch.nn.BatchNorm2d(channels)

    def forward(self, x, vessel):  # x = [N,C,H,W]
        km = self.kmap(x) * vessel  # [N,C/m,h,w] kmap(x)([4, 64, 64, 64]) vessel([4, 1, 64, 64]) km [4, 64, 64, 64]
        qm = self.qmap(x)  # [N,C/m,h,w]  [4, 64, 64, 64]
        ak1 = self.ac1((km, qm))  # [N,C/m,H_out*W_out, k,k]  [4, 4096, 81, 1]
        ak2 = self.ac2((km, qm))
        ak3 = self.ac3((km, qm))
        ak4 = self.ac4((km, qm))

        ck1 = torch.nn.functional.softmax(ak1, dim=-2)  # [N, H*W, k*k] [4, 4096, 81, 1]
        ck2 = torch.nn.functional.softmax(ak2, dim=-2)
        ck3 = torch.nn.functional.softmax(ak3, dim=-2)
        ck4 = torch.nn.functional.softmax(ak4, dim=-2)
        xm = self.xmap(x)

        xm_unfold1 = self.unfold1(xm).transpose(2, 1).contiguous()
        xm_unfold1 = xm_unfold1.view(xm.shape[0], -1, xm.shape[1], xm_unfold1.shape[-1] // xm.shape[1]) # [4, 4096, 64, 81]   对value map进行同样的操作
        xm_unfold2 = self.unfold2(xm).transpose(2, 1).contiguous()
        xm_unfold2 = xm_unfold2.view(xm.shape[0], -1, xm.shape[1], xm_unfold2.shape[-1] // xm.shape[1])
        xm_unfold3 = self.unfold3(xm).transpose(2, 1).contiguous()
        xm_unfold3 = xm_unfold3.view(xm.shape[0], -1, xm.shape[1], xm_unfold3.shape[-1] // xm.shape[1])
        xm_unfold4 = self.unfold4(xm).transpose(2, 1).contiguous()
        xm_unfold4 = xm_unfold4.view(xm.shape[0], -1, xm.shape[1], xm_unfold4.shape[-1] // xm.shape[1])

        pre_output1 = torch.matmul(xm_unfold1, ck1).squeeze(dim=-1).transpose(1, 2) #[4, 64, 4096]  value map与之前的kqmap运算结果相乘 去掉了local框 回来了通道
        # pre_output = torch.sum(xm_unfold*ck, dim=-2).transpose(1, 2)
        pre_output1 = pre_output1.view(pre_output1.shape[0], pre_output1.shape[1], x.shape[2], x.shape[3]) # [4, 64, 64, 64] reshape成传入时的大小和维度
        pre_output2 = torch.matmul(xm_unfold2, ck2).squeeze(dim=-1).transpose(1, 2)
        # pre_output = torch.sum(xm_unfold*ck, dim=-2).transpose(1, 2)
        pre_output2 = pre_output2.view(pre_output2.shape[0], pre_output2.shape[1], x.shape[2], x.shape[3])

        pre_output3 = torch.matmul(xm_unfold3, ck3).squeeze(dim=-1).transpose(1, 2)
        # pre_output = torch.sum(xm_unfold*ck, dim=-2).transpose(1, 2)
        pre_output3 = pre_output3.view(pre_output3.shape[0], pre_output3.shape[1], x.shape[2], x.shape[3])

        pre_output4 = torch.matmul(xm_unfold4, ck4).squeeze(dim=-1).transpose(1, 2)
        # pre_output = torch.sum(xm_unfold*ck, dim=-2).transpose(1, 2)
        pre_output4 = pre_output4.view(pre_output4.shape[0], pre_output4.shape[1], x.shape[2], x.shape[3])

        return x + self.bn(self.final1x1(pre_output1 + pre_output2 + pre_output3 + pre_output4))  #四个不同大小框运算结果相加 然后1x1卷积 批标准化 再残差


class ResUNet34_2task(torch.nn.Module):

    # 编码器使用resnet的原始模块
    def __init__(self, in_ch, bilinear=False, layer_return=False, bias=False):
        super(ResUNet34_2task, self).__init__()
        self.layer_return = layer_return
        self.bias = bias
        self.base_channel = 16
        self.up = upsample_x2
        self.decoder = Double_conv
        block = BasicBlock
        self.filters = [self.base_channel, self.base_channel * 2, self.base_channel * 4, self.base_channel * 8,
                        self.base_channel * 16]
        self.firstconv = torch.nn.Sequential(*[torch.nn.Conv2d(in_ch, self.base_channel, 3, padding=1),
                                               torch.nn.BatchNorm2d(self.base_channel),
                                               torch.nn.ReLU(inplace=True)])
        self.enc1 = self._make_layer(block, self.filters[0], self.filters[0], 3, 1)
        self.enc2 = self._make_layer(block, self.filters[0], self.filters[1], 4, 2)
        self.enc3 = self._make_layer(block, self.filters[1], self.filters[2], 6, 2)
        self.enc4 = self._make_layer(block, self.filters[2], self.filters[3], 3, 2)

        self.centerblock = Double_conv(self.filters[3], self.filters[3], bias=bias)


        self.up4_task1 = self.up(False, self.filters[4], self.filters[3])
        self.dec4_task1 = self.decoder(self.filters[3] * 2, self.filters[3], 0., bias=bias)
        self.up3_task1 = self.up(False, self.filters[3], self.filters[2])
        self.dec3_task1 = self.decoder(self.filters[2] * 2, self.filters[2], 0., bias=bias)
        self.up2_task1 = self.up(False, self.filters[2], self.filters[1])
        self.dec2_task1 = self.decoder(self.filters[1] * 2, self.filters[1], 0., bias=bias)
        self.up1_task1 = self.up(False, self.filters[1], self.filters[0])
        self.dec1_task1 = self.decoder(self.filters[0] * 2, self.filters[0], 0., bias=bias)
        self.finalconv_task1 = torch.nn.Conv2d(self.filters[0], 1, 1, 1, bias=bias)


        self.up4_task2 = self.up(False, self.filters[4], self.filters[3])
        self.dec4_task2 = self.decoder(self.filters[3] * 2, self.filters[3], 0., bias=bias)
        self.up3_task2 = self.up(False, self.filters[3], self.filters[2])
        self.dec3_task2 = self.decoder(self.filters[2] * 2, self.filters[2], 0., bias=bias)
        self.up2_task2 = self.up(False, self.filters[2], self.filters[1])
        self.dec2_task2 = self.decoder(self.filters[1] * 2, self.filters[1], 0., bias=bias)
        self.up1_task2 = self.up(False, self.filters[1], self.filters[0])
        self.dec1_task2 = self.decoder(self.filters[0] * 2, self.filters[0], 0., bias=bias)
        self.finalconv_task2 = torch.nn.Conv2d(self.filters[0], 4, 1, 1, bias=bias)
        self.dec4_conv = torch.nn.Conv2d(self.filters[3], 1, kernel_size=1)
        self.dec3_conv = torch.nn.Conv2d(self.filters[2], 1, kernel_size=1)

        # self.e_nl1 = Inception_LocalNonLocal_v2(self.filters[0], 11, 9, 7, 5, 2, 1)
        # self.e_nl2 = Inception_LocalNonLocal_v2(self.filters[1], 9, 7, 5, 3, 2, 1)
        self.e_nl3 = Inception_LocalNonLocal_v2(self.filters[2], 9, 7, 5, 3, 2, 1)
        self.e_nl4 = Inception_LocalNonLocal_v2(self.filters[3], 9, 7, 5, 3, 2, 1)

        initialize_weights(self)

    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # [4.3.512.512]
        #resnet的encoder
        out_first = self.firstconv(x)  # [4, 16, 512, 512]
        out_enc1 = self.enc1(out_first)# [4, 16, 512, 512] 第一层通道数不变 3个residual block
        out_enc2 = self.enc2(out_enc1) # [4, 32, 256, 256]
        out_enc3 = self.enc3(out_enc2) # [4, 64, 128, 128]
        out_enc4 = self.enc4(out_enc3) # [4, 128, 64, 64]
        out_center = self.centerblock(out_enc4) # [4, 128, 64, 64]  通道不变 两层卷积 分辨率不变
        
        #feature = torch.flatten(out_center, start_dim=1)
        # out_up4 = self.up4(out_center)

        #以下是unet的decoder的操作
        out_dec4_task1 = self.dec4_task1(torch.cat([out_enc4, out_center], dim=1)) #[4, 128, 64, 64]  先跳连 然后两层卷积 改变通道数
        out_up3_task1 = self.up3_task1(out_dec4_task1) # [4, 64, 128, 128]  上采样 减少通道数 增加分辨率
        out_dec3_task1 = self.dec3_task1(torch.cat([out_enc3, out_up3_task1], dim=1))
        out_up2_task1 = self.up2_task1(out_dec3_task1)
        out_dec2_task1 = self.dec2_task1(torch.cat([out_enc2, out_up2_task1], dim=1))
        out_up1_task1 = self.up1_task1(out_dec2_task1)
        out_dec1_task1 = self.dec1_task1(torch.cat([out_enc1, out_up1_task1], dim=1))  #[4, 16, 512, 512]
        out_task1 = self.finalconv_task1(out_dec1_task1)  # [4, 1, 512, 512]  1x1卷积将通道数减至0

        vessel1= self.dec4_conv(out_dec4_task1).sigmoid()   #[4, 1, 64, 64] 分割任务中的第一次double_conv的特征通过1x1卷积和sigmod激活 辅助
        out_dec4_task2 = self.dec4_task2(torch.cat([out_enc4, out_center], dim=1)) #[4, 128, 64, 64]
        out_dec4_task2 = self.e_nl4(out_dec4_task2,vessel1)  # [4, 128, 64, 64] VS-PDCB

        out_up3_task2 = self.up3_task2(out_dec4_task2) #上采样

        vessel2 = self.dec3_conv(out_dec3_task1).sigmoid()
        out_dec3_task2 = self.dec3_task2(torch.cat([out_enc3, out_up3_task2], dim=1))
        out_dec3_task2 = self.e_nl3(out_dec3_task2,vessel2)

        out_up2_task2 = self.up2_task2(out_dec3_task2)
        out_dec2_task2 = self.dec2_task2(torch.cat([out_enc2, out_up2_task2], dim=1))

        out_up1_task2 = self.up1_task2(out_dec2_task2)
        out_dec1_task2 = self.dec1_task2(torch.cat([out_enc1, out_up1_task2], dim=1))
        out_task2 = self.finalconv_task2(out_dec1_task2)


        return torch.sigmoid(out_task1), out_task2



