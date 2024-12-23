import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax

from functools import partial
import torch.nn.functional as F
from torch.nn import Softmax, Dropout

from typing import List, Callable
from torch import Tensor
import torch.nn.init as init
nonlinearity = partial(F.relu, inplace=True)

# def weight_init(module):
#     for n, m in module.named_children():
#         if isinstance(m, nn.Conv2d):
#             torch.nn.init.kaiming_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#         elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
#             nn.init.constant_(m.weight, 1.0)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#         elif isinstance(m, nn.Linear):
#             torch.nn.init.kaiming_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#         elif isinstance(m, nn.Sequential):
#             weight_init(m)
#         elif isinstance(m, nn.ReLU):
#             pass
#         elif isinstance(m, nn.AdaptiveAvgPool2d):
#             pass
#         elif isinstance(m, nn.AdaptiveMaxPool2d):
#             pass
#         elif isinstance(m, nn.Upsample):
#             pass
#         elif isinstance(m, nn.Sigmoid):
#             pass
#         elif isinstance(m, nn.MaxPool2d):
#             pass


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)  # inplace=True
    )


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

class PAM_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        # self.exp_feature = exp_feature_map
        # self.tanh_feature = tanh_feature_map
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()




# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=3, dropout_prob=0.2):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
#         self.dropout = nn.Dropout2d(p=dropout_prob)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = max_out
#         x = self.conv1(x)
#         x = self.dropout(x)
#         return self.sigmoid(x)



   # def init_weight(self):
   #     nn.init.xavier_uniform_(self.conv1.weight)




class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CG_block(nn.Module):
    def __init__(self, in_c, out_c, dilation=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=1, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x1 = self.bn1(x1)
        x = self.relu(x1)

        return x


class DWCon(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, dilation=1):
        super(DWCon, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=in_planes)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)
        self.groupN = nn.GroupNorm(4, out_planes)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.groupN(x)
        x = self.relu(x)
        return x

# class DWCon(nn.Module):
#     def __init__(self, in_planes, out_planes):
#         super(DWCon, self).__init__()
#         self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, groups=in_planes)
#         self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)
#         self.groupN = nn.GroupNorm(4, out_planes)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
#         self.relu = nn.ReLU()
#
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.groupN(x)
#         x = self.relu(x)
#         return x


#-----------------------------------------------------

#----------------------------------------------------------------------------------------
# class MFA(nn.Module):
#     def __init__(self, in_channels, out_channels, groups=4, dropout_rate=0.2):
#         super(MFA, self).__init__()
#         self.cv1 = nn.Sequential(
#             nn.Conv2d(in_channels, 256, kernel_size=1),
#             nn.BatchNorm2d(256),
#             nn.ELU(inplace=True)  # Using ELU activation function
#         )
#         self.groupN = nn.GroupNorm(groups, 128)
#         self.dwconv1 = GroupedDepthwiseSeparableConv(128, 128)
#         self.dwconv2 = GroupedDepthwiseSeparableConv(256, 256)
#         self.dropout = nn.Dropout2d(p=dropout_rate)
#         self.sigmoid = nn.Sigmoid()
#
#        # self.init_weight()
#
#     def forward(self, x):
#         x = self.cv1(x)
#         x = channel_shuffle(x, 4)  # channel shuffle
#         b, c, H, W = x.size()
#
#         channels_per_group = c // 2
#         x1, x2 = torch.split(x, channels_per_group, dim=1)
#
#         x1 = self.groupN(x1)
#         x2 = self.groupN(x2)
#
#         x1 = self.dwconv1(x1)
#         x2 = self.dwconv1(x2)
#
#         x1 = self.sigmoid(x1)
#         x2 = self.sigmoid(x2)
#
#         x1 = x1 * x1
#         x2 = x2 * x2
#
#         x = torch.cat([x1, x2], dim=1)
#         x = self.dwconv2(x)
#         x = self.dropout(x)
#
#         return x

    # def init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
#----------------------------------------------------------------------------------------
class MFA(nn.Module):
    def __init__(self):
        super(MFA, self).__init__()
        self.dwconv1 = DWCon(128, 128)
        self.sigmoid = nn.Sigmoid()
        self.dwconv2 = DWCon(128, 256)
        self.groupN = nn.GroupNorm(4, 128)
        self.cv1 = nn.Sequential(
            nn.Conv2d(320, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
       # self.init_weight()

    def forward(self, x5):
        x5 = self.cv1(x5)
        x5 = channel_shuffle(x5, 4)
        b, c, H, W = x5.size()  # torch.Size([1, 256, 8, 8])
        channels_per_group = c // 2

        # split channel  torch.Size([1, 512, 14, 14])
        x51, x52 = torch.split(x5, channels_per_group, dim=1)

        x51_ = self.groupN(x51)
        x51_ = self.dwconv1(x51_)
        x51_ = self.sigmoid(x51_)
        x51_ = x51 * x51_

        x52_ = self.groupN(x52)
        x52_ = self.dwconv1(x52_)
        x52_ = self.sigmoid(x52_)
        x52_ = x52 * x52_

        x5_ = torch.add([x51_, x52_], dim=1)
        out_MFA = self.dwconv2(x5_)
        return out_MFA


class MFA_E(nn.Module):
    def __init__(self):
        super(MFA_E, self).__init__()

        # Define convolutional layers with different configurations
        self.dwconv1_small = DWCon(128, 128, kernel_size=3, padding=1)  # Smaller kernel size
        self.dwconv1_large = DWCon(128, 128, kernel_size=5, padding=2)  # Larger kernel size
        self.dwconv2 = DWCon(256, 320)
        self.groupN = nn.GroupNorm(4, 128)
        self.cv1 = nn.Sequential(
            nn.Conv2d(320, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x5):
        x5 = self.cv1(x5)
        x5 = channel_shuffle(x5, 4)
        b, c, H, W = x5.size()  # torch.Size([1, 256, 8, 8])
        channels_per_group = c // 2

        # Split channels
        x51, x52 = torch.split(x5, channels_per_group, dim=1)

        # Process each branch with different configurations
        x51_small = self.groupN(x51)
        x51_small = self.dwconv1_small(x51_small)
        x51_small = self.sigmoid(x51_small)
        x51_small = x51 * x51_small

        x52_large = self.groupN(x52)
        x52_large = self.dwconv1_large(x52_large)
        x52_large = self.sigmoid(x52_large)
        x52_large = x52 * x52_large

        # Concatenate the processed branches
        x5_ = torch.cat([x51_small, x52_large], dim=1)

        # Final processing
        out_MFA = self.dwconv2(x5_)
        return out_MFA

#     def init_weight(self):
#         weight_init(self)


# In Python function annotations, -> Tensor denotes the expected return type of the function.
# In this case, Tensor indicates that the function channel_shuffle is expected to return a tensor.

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()  # unpack the size of the input tensor x into four variables
    channels_per_group = num_channels // groups   # calculate the number of channels per group
    x = x.view(batch_size, groups, channels_per_group, height, width)  # channels are grouped according to the specified number of groups.
                                                                       # The shape of x becomes [batch_size, groups, channels_per_group, height, width].
    x = torch.transpose(x, 1, 2).contiguous()   # This line transposes the tensor x such that the dimensions representing the groups
                                                                    # and channels are swapped. This operation is performed to facilitate shuffling of channels within groups.
    x = x.view(batch_size, -1, height, width)  # This line reshapes the tensor x back into its original shape, but now with channels shuffled within groups.
    return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

# class ESA_block(nn.Module):
#     def __init__(self, in_ch, split_factor=16, dropout_prob=0.2):
#         super(ESA_block, self).__init__()
#
#         self.SA1 = SpatialAttention()
#         self.SA2 = SpatialAttention()
#
#         self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.sa_fusion = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # Channel shuffle
#         x = channel_shuffle(x, 4)
#
#         # Split channels
#         b, c, H, W = x.size()
#         channels_per_group = c // 2
#         x1, x2 = torch.split(x, channels_per_group, dim=1)
#
#         # Apply CAM Module and PAM
#         s1 = self.SA1(x1)
#         s2 = self.SA2(x2)
#
#         # Concatenate attention maps
#         concat_att_map = torch.cat([s1, s2], dim=1)
#
#         # Apply attention fusion
#         attention_fused = self.sa_fusion(concat_att_map)
#
#         # Apply attention to input feature maps
#         out = attention_fused * x + x
#
#         return out

class ESA_block(nn.Module):
    def __init__(self, in_ch, split_factor=16, dropout_prob=0.2):
        super(ESA_block, self).__init__()

       # self.conv2d_2 = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=3, padding=1, dilation=2)
       # self.conv2d_4 = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=3, padding=1, dilation=4)

        self.conv2d_2 = DWCon(in_ch // 2, in_ch // 2, kernel_size=3, padding=2, dilation=2)  # Smaller kernel size
        self.conv2d_4 = DWCon(in_ch // 2, in_ch // 2, kernel_size=3, padding=4, dilation=4)  # Larger dilation rate

        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()

       # self.SA1 = PAM_Module(in_ch // 2)
      #  self.SA2 = PAM_Module(in_ch // 2)

        self.groupN = nn.GroupNorm(2, in_ch // 2)

        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.sa_fusion = nn.Sequential(
            BasicConv2d(1, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel shuffle
        x = channel_shuffle(x, 4)

        # Split channels
        b, c, H, W = x.size()  # torch.Size([1, 256, 8, 8])
        channels_per_group = c // 2
        x41, x42 = torch.split(x, channels_per_group, dim=1)

        # Apply conv2d with dilation rate 2 and 4 to the split tensors
        x41 = self.groupN(x41)
        x41 = self.conv2d_2(x41)

        x42 = self.groupN(x42)
        x42 = self.conv2d_4(x42)

        # Apply spatial attention modules
        s1 = self.SA1(x41)
        s2 = self.SA2(x42)

        # Resize s1 and s2 to match each other's spatial dimensions
       # s1 = F.interpolate(s1, size=(H, W), mode='bilinear', align_corners=True)
       # s2 = F.interpolate(s2, size=(H, W), mode='bilinear', align_corners=True)

        # Perform fusion
        nor_weights = F.softmax(self.weight, dim=0)
        s_all = s1 * nor_weights[0] + s2 * nor_weights[1]
        out = self.sa_fusion(s_all) * x + x

        return out

class Att_block_1(nn.Module):
    def __init__(self, in_channel, dropout_prob=0.5):
        super(Att_block_1, self).__init__()
        self.conv1 = conv3otherRelu(in_channel, in_channel)
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        edge = x - self.avg_pool(x)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x
        out = self.PReLU(out)
        out = self.dropout(out)  # Applying dropout here
        return out

# class Att_block_1(nn.Module):
#     def __init__(self, in_channel):
#         super(Att_block_1, self).__init__()
#         self.conv1 = conv3otherRelu(in_channel, in_channel)
#         self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
#         self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(in_channel)
#         self.sigmoid = nn.Sigmoid()
#         self.PReLU = nn.PReLU(in_channel)
#
#
#     def forward(self, x):
#         edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
#         weight = self.sigmoid(self.bn1(self.conv_1(edge)))
#         out = weight * x + x
#         out = self.PReLU(out)
#
#        # x_out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
#
#         return out


# class Att_block_2(nn.Module):
#     def __init__(self, in_ch):
#         super(Att_block_2, self).__init__()
#         self.conv1 = conv3otherRelu(in_ch, in_ch)
#
#       #  self.PAM = PAM_Module(in_ch)
#         self.CAM = CAM_Module()
#
#         self.conv2P = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))
#         self.conv2C = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))
#         self.conv3 = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2C(self.CAM(x))
#         return self.conv3(x)



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CFFANet_2(nn.Module):
    def __init__(self, num_channels=3, num_classes=1,channel=32, split_factor=16, pretrained=True):
        super(CFFANet_2, self).__init__()

        # Backbone model
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.layer1 = mobilenet.features[0]
        self.layer2 = mobilenet.features[1]
        self.layer3 = nn.Sequential(
            mobilenet.features[2],
            mobilenet.features[3], )
        self.layer4 = nn.Sequential(
            mobilenet.features[4],
            mobilenet.features[5],
            mobilenet.features[6], )
        self.layer5 = nn.Sequential(
            mobilenet.features[7],
            mobilenet.features[8],
            mobilenet.features[9],
            mobilenet.features[10], )
        self.layer6 = nn.Sequential(
            mobilenet.features[11],
            mobilenet.features[12],
            mobilenet.features[13], )
        self.layer7 = nn.Sequential(
            mobilenet.features[14],
            mobilenet.features[15],
            mobilenet.features[16], )
        self.layer8 = nn.Sequential(
            mobilenet.features[17],
        )

        self.MFA_block = MFA_E()  # Multiscale features Aggregation bridge
       # self.multi_context = CG_block()  # For every encoder block 1,2,3,4

        self.esa_4 = ESA_block(96)
        self.esa_3 = ESA_block(32)
        #self.ESAM_3 = ESAM(filters[2], 32)
        self.attention2 = Att_block_1(24)
        self.attention1 = Att_block_1(16)

        self.decoder5 = DecoderBlock(320, 128)
        self.decoder4 = DecoderBlock(224, 96)
        self.decoder3 = DecoderBlock(128, 24)
        self.decoder2 = DecoderBlock(48, 16)
        self.decoder1 = DecoderBlock(32, 16)

        self.finaldeconv1 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(16, num_classes, 3, padding=1)


    def forward(self, input):
        # Encoder
        # input 256*256*3
        # conv1 128*128*16
        # conv2 64*64*24
        # conv3 32*32*32
        # conv4 16*16*96
        # conv5 8*8*320
        x = self.layer1(input)

        e1 = self.layer2(x)  # x / 2   # torch.Size([1, 16, 128, 128]
        e2 = self.layer3(e1)  # 24, x / 4   torch.Size([1, 24, 64, 64])
        e3 = self.layer4(e2)  # 32, x / 8   torch.Size([1, 32, 32, 32])

        l5 = self.layer5(e3)  # 64, x / 16
        e4 = self.layer6(l5)  # 96, x / 16    torch.Size([1, 96, 16, 16])

        l7 = self.layer7(e4)  # 160, x / 32
        e5 = self.layer8(l7)  # 320, x / 32   torch.Size([1, 320, 8, 8])

       # print(e1.shape)

        encoder_booster = self.MFA_block(e5)
       # print(encoder_booster.shape)

    #    encoder_booster = encoder_booster + e5
    #    encoder_booster = torch.cat([encoder_booster, e5], dim=1)

        #------------- Decoder -------------------
        d5 = self.decoder5(encoder_booster)  # torch.Size([1, 128, 16, 16])
      #  print(d5.shape)

        # Decoder 4
        d4_ESAM_4 = self.esa_4(e4)  # torch.Size([1, 256, 16, 16])
       # print(d4_ESAM_4.shape)

        d4_final = torch.cat([d5, d4_ESAM_4], dim=1)
       # print(d4_final.shape)

        # Decoder 3
        d4 = self.decoder4(d4_final)  # torch.Size([1, 128, 16, 16])
      #  print(d4.shape)

        d3_ESAM_3 = self.esa_3(e3)  # torch.Size([1, 256, 16, 16])
       # print(d3_ESAM_3.shape)

        d3_final = torch.cat([d4, d3_ESAM_3], dim=1)
       # print(d3_final.shape)

        #----------------------------

      #  print(e2_att.shape)

        d3= self.decoder3(d3_final)
      #  print(d3.shape)

        e2_att = self.attention2(e2)



        d2_final = torch.cat([d3, e2_att], dim=1)
       # print(d2_final.shape)
        #----------------------------

        d2 = self.decoder2(d2_final)
       # print(d2.shape)

        e1_att = self.attention1(e1)
       # print(e1_att.shape)

        d1_final = torch.cat([d2, e1_att], dim=1)
        print(d1_final.shape)

        out = self.finaldeconv1(d1_final)  # torch.Size([1, 16, 256, 256])
       # print(out.shape)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        # out = self.finalrelu2(out)

        # d3 = F.interpolate(d3, size=x.size()[2:], mode='bilinear', align_corners=False)
        out = torch.sigmoid(out)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    num_classes = 1
    in_batch, inchannel, in_h, in_w = 1, 3, 256, 256
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = CFFANet_2()
    out = net(x)
   # print(out.shape)

    num_params = count_parameters(net)
   # print("Number of trainable parameters:", num_params)