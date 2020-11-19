import torch
import torch.nn.functional as F
from torch import nn

import torchvision
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import segmentation_models_pytorch as smp

from torch.autograd import Variable

# from resnet import *
# from resnet import *
#=====================================================================#
#===========================resunet network===========================#
#=====================================================================#

class clas_pretrain_unet(nn.Module):
    def __init__(self,in_channels=1,out_channels=5):
        super(clas_pretrain_unet,self).__init__()
        feature = [64,128,256,512]
        self.model = smp.Unet('resnet34',in_channels=in_channels,classes=out_channels,encoder_weights=None)
        self.decoders = list(self.model.decoder.children())[1]
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.classconv = nn.Conv2d(512,out_channels,7,7)
        # print(list(self.model.encoder.children())[-2][2])
        # print(self.model.encoder)
        
        self.classLiner = nn.Linear(feature[-1]*4*4,out_channels)

        self.last = nn.Conv2d(32,out_channels,1)
        # self.classliner = nn.Linear
    def encoder_forward(self, x):
        return self.model.encoder.forward(x)
    
    def forward(self,x,phase='train'):
        encoders = self.encoder_forward(x)
        
        class_layer = encoders[-1]
        # print(encoders[-1].shape,'22',class_layer.view(class_layer.size(0),-1).shape)
        
        # class_feature = F.linear(class_layer.view(class_layer.size(0),-1),(512*(2**len(encoders))*(2**len(encoders)))))
        result = torch.cat([encoders[-2],self.upsample(encoders[5])],1)
        
        d0 = self.decoders[0](result) 
        
        d0 = torch.cat([encoders[-3],d0],1)
        d1 = self.decoders[1](d0) 
        
        d1 = torch.cat([encoders[-4],d1],1)
        d2 = self.decoders[2](d1)

        d2 = torch.cat([encoders[-5],d2],1)
        
        d3 = self.decoders[3](d2)
        
        result = self.last(d3)
        # result = self.softmax(self.finals(d3))
        if phase == 'train':
            class_feature = self.classLiner(class_layer.view(class_layer.size(0),-1))
            return result,F.softmax(class_feature)
        else:
            return result,None
###load model 
class pretrain_unet(nn.Module):
    
    def __init__(self,in_channels=1,classes=4,active='sigmoid'):
        super(pretrain_unet,self).__init__()
        self.model = smp.Unet('timm-efficientnet-b4',in_channels=1,classes=classes,activation=None)
        if active == 'softmax':
            self.sigmoid = nn.Softmax(dim=1)
        elif active == 'sigmoid':
            self.sigmoid = nn.Sigmoid()
        self.active = active
    def forward(self,x):
        x = self.model(x)
        if self.active == 'sigmoid' or self.active == 'softmax':
            result = self.sigmoid(x)
        else: 
            result = x

        return result,x 