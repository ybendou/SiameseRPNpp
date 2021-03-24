import torch.nn as nn
import torch.nn.functional as F
import torch
from models.resnet import encoder


def conv_block(in_channels, out_channels):
  """
  Returns a convolution block of a 3x3 Conv2d and a Batchnorm2d
  """
  return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
                       nn.BatchNorm2d(out_channels))

def xcorr_depthwise(x, kernel):
    """depthwise cross correlation from  https://github.com/HonglinChu/SiamTrackers/blob/570f2ad833b03a1340831e94d050e3b3c2ac3f0e/3-SiamRPN/SiamRPNpp-UP/siamrpnpp/core/xcorr.py#L39
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

class RPNModule(nn.Module):
    def __init__(self, features_in=256, anchor=5):
        super(RPNModule, self).__init__()
        self.conv_adj1 = conv_block(features_in, features_in)
        self.conv_adj2 = conv_block(features_in, features_in)
        self.conv_adj3 = conv_block(features_in, features_in)
        self.conv_adj4 = conv_block(features_in, features_in)

        self.box_head = nn.Conv2d(features_in, 4*anchor, kernel_size=1)
        self.cls_head = nn.Conv2d(features_in, 2*anchor, kernel_size=1)

    def forward(self,target_f_z, search_f_z):
        target_f_z = target_f_z[:,:,4:-4,4:-4] # crop center 7x7 regions
        out_adj1 = self.conv_adj1(target_f_z)
        out_adj_2 = self.conv_adj2(search_f_z)

        out_adj3 = self.conv_adj3(target_f_z)
        out_adj4 = self.conv_adj3(search_f_z)

        dw_corr_1 = xcorr_depthwise(out_adj_2, out_adj1)
        dw_corr_2 = xcorr_depthwise(out_adj4, out_adj3)

        out_box_head = self.box_head(dw_corr_1)
        out_cls_head = self.cls_head(dw_corr_2)

        return out_box_head, out_cls_head



class SiameseRPNpp(nn.Module):
    def __init__(self, anchor=5):
        super(SiameseRPNpp, self).__init__()
        
        self.encoder =  encoder(pretrained=True)
#         self.targetEncoder = encoder(pretrained=True)
#         self.searchEncoder = encoder(pretrained=True)

        self.conv3_RPN = RPNModule(anchor=anchor)
        self.conv4_RPN = RPNModule(anchor=anchor)
        self.conv5_RPN = RPNModule(anchor=anchor)

        self.box_weighted_sum = nn.Conv2d(3*4*anchor, 4*anchor, kernel_size=1, padding=0, groups=4*anchor)
        self.cls_weighted_sum = nn.Conv2d(3*2*anchor, 2*anchor, kernel_size=1, padding=0, groups=2*anchor)


    def forward(self,x_target, x_search):
        target_conv3, target_conv4, target_conv5 = self.encoder(x_target)
        search_conv3, search_conv4, search_conv5 = self.encoder(x_search)

        out_box_conv3, out_cls_conv3 = self.conv3_RPN(target_conv3, search_conv3) 
        out_box_conv4, out_cls_conv4 = self.conv3_RPN(target_conv4, search_conv4) 
        out_box_conv5, out_cls_conv5 = self.conv3_RPN(target_conv5, search_conv5) 

        box_conv_all = torch.cat([out_box_conv3, out_box_conv4, out_box_conv5], dim=1)
        cls_conv_all = torch.cat([out_cls_conv3, out_cls_conv4, out_cls_conv5], dim=1)
        out_box = self.box_weighted_sum(box_conv_all)
        out_cls = self.cls_weighted_sum(cls_conv_all)

        return out_box, out_cls


