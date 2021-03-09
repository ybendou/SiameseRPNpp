import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.model_zoo as model_zoo

model_urls = {
      'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  }

########### Custom Resnet

# Inspired from https://github.com/PengBoXiangShang/SiamRPN_plus_plus_PyTorch
class Bottleneck(nn.Module):
  """
  Basic block of a custom Resnet for Siamese Paper, has expansion and downsampling function in case the residuals and the identity are different
  """
  def __init__(self,in_channels, out_channels, expansion=4, stride_conv1=1, stride_conv2=2 , dilation=1, padding=1, is_downsample=False, downsample_ksize=1):
      super().__init__()
      self.expansion = expansion
    
      self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1, stride=stride_conv1, bias=False)
      self.bn1   = nn.BatchNorm2d(out_channels)
      self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, stride=stride_conv2, dilation=dilation, bias=False) 
      self.bn2   = nn.BatchNorm2d(out_channels)
      self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=stride_conv1, bias=False)
      self.bn3   = nn.BatchNorm2d(out_channels*self.expansion)
      self.relu  = nn.ReLU(inplace=True)

      if dilation==1 and stride_conv1==1 and stride_conv2==1 and is_downsample==False and padding==1: 
        self.downsample = None
      # If the residuals output are different than the identity output
      else : 
        downsample_padding = 0
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=downsample_ksize, dilation=dilation, padding=downsample_padding, stride=stride_conv2, bias=False),
                                                  nn.BatchNorm2d(out_channels*self.expansion))
      
  def forward(self,x):
      identity = x
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.conv2(x)
      x = self.bn2(x)
      x = self.conv3(x)
      x = self.bn3(x)

      if self.downsample != None : 
          identity = self.downsample(identity)
      return self.relu(x+identity)
        
class Resnet(nn.Module):
  """
  Custom Resnet50 module, has dilation in layer 3 and 4, extra conv layers after layer 2, 3 and 4 and has a stride of 1 in layer 1, 3 and 4 
  """
  def __init__(self,in_channels, init=False):
    super(Resnet, self).__init__()
    self.expansion = 4
    self.conv1   = nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, stride=2, bias=False)
    self.bn1     = nn.BatchNorm2d(64)
    self.relu    = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)

    self.layer1, _                  = self._make_layer(layer_id=1, blocks=3, dilation=1, padding=1)
    
    self.layer2, out_channel_layer2 = self._make_layer(layer_id=2, blocks=4, dilation=1, padding=0, downsample_ksize=3) # adding dilation + padding 
    self.extra_layer2 = nn.Conv2d(out_channel_layer2, 256, kernel_size=1, stride=1, padding=0, bias=False)

    self.layer3, out_channel_layer3 = self._make_layer(layer_id=3, blocks=6, dilation=2, padding=2) # adding dilation + padding 
    self.extra_layer3 = nn.Conv2d(out_channel_layer3, 256, kernel_size=1, stride=1, padding=0, bias=False)
    

    self.layer4, out_channel_layer4 = self._make_layer(layer_id=4, blocks=3, dilation=4, padding=4) # adding dilation + padding 
    self.extra_layer4 = nn.Conv2d(out_channel_layer4, 256, kernel_size=1, stride=1, padding=0, bias=False)
    
    #Initialize weights
    if init:
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(
                  m.weight, mode='fan_out', nonlinearity='relu')
          elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)

  def _make_layer(self, layer_id=1, blocks=3, dilation=1, padding=1, downsample_ksize=1):
    layer = []
    stride_conv1, stride_conv2 = 1, 2

    if layer_id in [1, 3, 4]:
      stride_conv2 = 1

    if layer_id == 1 : 
      in_channel_bot1, out_channel_bot1 = 64, 64
    
    else : 
      in_channel_bot1, out_channel_bot1 = 64*2**(layer_id), 64*2**(layer_id-1)
    
    layer.append(Bottleneck(in_channel_bot1, out_channel_bot1, stride_conv1 = stride_conv1, 
                            stride_conv2 = stride_conv2, expansion=self.expansion, dilation=dilation, 
                            padding=padding, is_downsample=True, downsample_ksize=downsample_ksize))          

    in_channel  = out_channel_bot1*self.expansion
    out_channel = out_channel_bot1
    
    for _ in range(blocks-1) : 
      layer.append(Bottleneck(in_channel, out_channel, stride_conv1 = 1, 
                              stride_conv2 = 1, expansion=self.expansion))
      
    return nn.Sequential(*layer), out_channel*self.expansion

  def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    out_layer2 = self.extra_layer2(x)

    x = self.layer3(x)
    out_layer3 = self.extra_layer3(x)

    x = self.layer4(x)
    out_layer4 = self.extra_layer4(x)

    return out_layer2, out_layer3, out_layer4

def encoder(in_channels=3, pretrained=False):
  """
  Load resnet50 parameters to the custom resnet
  """
  model = Resnet(in_channels=in_channels, init=not pretrained)

  if pretrained:
    print('Loading pretrained model weights')
    resnet50_state_dict = model_zoo.load_url(model_urls['resnet50'])
    del resnet50_state_dict['fc.weight'] # unused weights
    del resnet50_state_dict['fc.bias']   # unused weights
    del resnet50_state_dict['layer2.0.downsample.0.weight']   # weight of this layer is not compatible anymore (changed kernel size from 1 to 3)


    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    resnet50_state_dict = {k: v for k, v in resnet50_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(resnet50_state_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
  
  return model