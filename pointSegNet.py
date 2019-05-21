
""" SqueezeSeg Model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import time
import torch.autograd as autograd

class Conv0(nn.Module):
    def __init__(self, inputs, outputs, size, stride, padding=0, xavier=True, dilations=(1,1)):
        super(Conv, self).__init__()           
        
        # self.conv = nn.Conv2d(inputs, outputs, kernel_size=size, stride=stride, padding=padding, dilation=dilations)
        self.conv=nn.Conv2d(inputs,inputs ,size, stride, padding,dilation=dilations,groups=inputs)
        self.pointwise = nn.Conv2d(inputs, outputs ,1,1,0,1,1)
        if xavier:
            nn.init.xavier_uniform_(self.conv.weight)
            nn.init.constant_(self.conv.bias, 0.0)
        else:
            nn.init.normal_(self.conv.weight)
            nn.init.constant_(self.conv.bias, 0.0)
        # self.bn=nn.BatchNorm2d(outputs,eps=1e-03)
        self.bn=nn.BatchNorm2d(outputs)
    def forward(self, x, BN=True, relu=True):
        p0=self._count_pad(x.size()[2], self.conv.stride[0], self.conv.dilation[0], self.conv.kernel_size[0])
        p1=self._count_pad(x.size()[3], self.conv.stride[1], self.conv.dilation[1], self.conv.kernel_size[1])
        p0=ceil(p0)
        p1=ceil(p1)
        x=F.pad(x,(p1,p1,p0,p0))
        
        out=self.conv(x)
        out=self.pointwise(out)
        if BN:
            out=self.bn(out)
        if relu:
            out=F.relu(out, inplace=True)
        return out
    def _count_pad(self, i, stride, dilation, size):
        x=1+(i-1)*stride-i+dilation*(size-1)
        return x/2        
    
class Conv(nn.Module):
    def __init__(self, inputs, outputs, size, stride, padding=0, xavier=True, dilations=(1,1)):
        super(Conv, self).__init__()           
        self.inputs=inputs
        self.outputs=outputs
        if type(size) is int:
            self.size=(size,size)
        else:
            self.size=size
        if type(stride) is int:
            self.stride=(stride,stride)
        else:
            self.stride=stride        
        self.xavier=xavier
        self.dilations=dilations
        self.bn=nn.BatchNorm2d(outputs)
    def forward(self, x, BN=True, relu=True):            
        p0=self._count_pad(x.size()[2], self.stride[0], self.dilations[0], self.size[0])
        p1=self._count_pad(x.size()[3], self.stride[1], self.dilations[1], self.size[1])
        p0=ceil(p0)
        p1=ceil(p1)
        weight0=autograd.Variable(torch.randn(self.inputs, 1, self.size[0], self.size[1])).cuda()
        bias0=autograd.Variable(torch.zeros([self.inputs])).cuda()
        weight1=autograd.Variable(torch.randn(self.outputs, self.inputs, 1,1)).cuda()
        bias1=autograd.Variable(torch.zeros([self.outputs])).cuda()                                  
        out=F.conv2d(x, weight0, bias0, self.stride, (p0, p1), self.dilations, self.inputs)                       
        out=F.conv2d(out, weight1, bias1, 1, 0, 1, 1)    
        if BN:
            out=self.bn(out)
        if relu:
            out=F.relu(out, inplace=True)
        return out
    def _count_pad(self, i, stride, dilation, size):
        x=1+(i-1)*stride-i+dilation*(size-1)
        return x/2        
        

class Deconv(nn.Module):
    def __init__(self, inputs, outputs,  size, stride, padding, xavier=False):
        super(Deconv, self).__init__()
        self.deconv=nn.ConvTranspose2d(inputs, outputs, kernel_size=size, 
                                       stride=stride, padding=padding)
        if xavier:
            nn.init.xavier_uniform_(self.deconv.weight)
            nn.init.constant_(self.deconv.bias, 0.0)
        # TODO add bilinear if need
        else:
            nn.init.normal_(self.deconv.weight)
            nn.init.constant_(self.deconv.bias, 0.0)
    def forward(self, x):
        return F.relu(self.deconv(x),inplace=True)
    
    
class Scale(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(Scale, self).__init__()
        self.pool=nn.AvgPool2d(kernel_size=kernel_size, stride=(1,stride), padding=padding)
    
    def forward(self, x):
        return self.pool(x)
    
class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(MaxPool, self).__init__()
        self.pool=nn.MaxPool2d(kernel_size=kernel_size,stride=(1,stride), padding=padding)
    def forward(self, x):
        p0=(self.pool.kernel_size-1)/2
        r=ceil(p0/2)       
        p1=(self.pool.kernel_size-1)/2
        b=ceil(p1/2)
        x=F.pad(x,(r,r,b,b))
        return self.pool(x)
    
class ASPP(nn.Module):
    def __init__(self, inputs, depth=128):
        super(ASPP,self).__init__()
        self.conv=Conv(inputs, depth, 1, (1,1), padding=0)
        self.pool1x1=Conv(inputs, depth, 3, (1,1), padding=0, dilations=(1,1))
        self.pool3x3_1=Conv(inputs, depth, 3, (1,1), padding=0, dilations=(6,6))
        self.pool3x3_2=Conv(inputs, depth, 3, (1,1), padding=0, dilations=(9,9))
        self.pool3x3_3=Conv(inputs, depth, 3, (1,1), padding=0, dilations=(12,12))
        self.conv1=Conv(640,depth, 1, (1,1))
    
    def forward(self, x):
        feature_map_size=x.size()
        out1=x.mean(3,True).mean(2,True)
        out1=self.conv(out1, False)# TODO
        out1=F.interpolate(out1, (feature_map_size[2],feature_map_size[3]))
        pool1=self.pool1x1(x, False)
        pool3_1=self.pool3x3_1(x, False)
        pool3_2=self.pool3x3_2(x, False)
        pool3_3=self.pool3x3_3(x, False)
        out=torch.cat([out1, pool1, pool3_1, pool3_2, pool3_3],1)
        out=self.conv1(out)
        return out
        
        
        
class FC(nn.Module):
    """Fully connected layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      hiddens: number of (hidden) neurons in this layer.
      flatten: if true, reshape the input 4D tensor of shape 
          (batch, height, weight, channel) into a 2D tensor with shape 
          (batch, -1). This is used when the input to the fully connected layer
          is output of a convolutional layer.
      relu: whether to use relu or not.
      xavier: whether to use xavier weight initializer or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A fully connected layer operation.
    """
    def __init__(self, inputs, outputs, flatten=False, xavier=False, bias_init_val=0.0):
        super(FC, self).__init__()
        self.fc=nn.Linear(inputs, outputs)
        if xavier:
            nn.init.xavier_uniform_(self.fc.weight)
        else:
            nn.init.normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, bias_init_val)
    def forward(self, x, relu=True):
        # TODO flatten how to manage it
        out=self.fc(x)
        if relu:
            out=F.relu(out, inplace=True)        
        return out
    
class Fire(nn.Module):
    '''
    Args:
        inputs: input channels
        s1x1: output channels of squeeze layer
        e1x1: output channels of expand layer(1x1)
        e3x3: output channels of expand layer(3x3)
    '''
    def __init__(self, inputs, s1x1, out_channels):
        super(Fire, self).__init__()
        self.sq1x1=Conv(inputs, s1x1, 1, (1,1), 0)
        self.ex1x1=Conv(s1x1, int(out_channels/2), 1, (1,1), 0)
        self.ex3x3=Conv(s1x1, int(out_channels/2), 3, (1,1), 0)
    def forward(self, x):
        out=self.sq1x1(x)
        return torch.cat([self.ex1x1(out), self.ex3x3(out)],1)

class FireDeconv(nn.Module):
    """Fire deconvolution layer constructor.

    Args:
      inputs: input channels
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      factors: spatial upsampling factors.[1,2]
    Returns:
      fire layer operation.
    """
    def __init__(self, inputs, s1x1, e1x1, e3x3, factors=[1,2], padding=(0,1)):
        super(FireDeconv, self).__init__()
        ksize_h = factors[0] * 2 - factors[0] % 2
        ksize_w = factors[1] * 2 - factors[1] % 2
        self.sq1x1=Conv(inputs, s1x1, 1, (1,1), 0)
        self.deconv=Deconv(s1x1, s1x1, (ksize_h,ksize_w), (factors[0], factors[1]), padding)
        self.ex1x1=Conv(s1x1, e1x1, 1, (1,1), 0)
        self.ex3x3=Conv(s1x1, e3x3, 3, (1,1), 0)
    def forward(self, x):
        out=self.sq1x1(x)#TODO 
        out=self.deconv(out)
        ex1=self.ex1x1(out)
        ex3=self.ex3x3(out)
        return torch.cat([ex1, ex3],1)
        
class SqueezeEx(nn.Module):
    '''Squeeze reweighting layer
     
    '''
    def __init__(self, inputs, outs, ratio):
        super(SqueezeEx, self).__init__()
        self.fc=FC(inputs, int(outs/ratio))
        self.fc1=FC(int(outs/ratio), outs)
    def forward(self, x):
        squeeze=x.mean(2).mean(2)
        ex=self.fc(squeeze)
        ex=self.fc1(ex,False)
        ex=torch.sigmoid(ex)
        ex=ex.view(-1,squeeze.size()[-1], 1,1)
        scale=x*ex
        return scale
    
class SqueeOri(nn.Module):
    def __init__(self, mc):
        super(SqueeOri, self).__init__()
         
        self.mc=mc
        self.conv1=Conv(5,64,3,(1,2),0, True)
        self.conv1_skip=Conv(5, 64, 1, 1, 0, True)         
        self.pool1=MaxPool(3, 2, 0)
        self.fire2=Fire(64,16,128)
        self.fire3=Fire(128, 16, 128)
        self.SR1=SqueezeEx(128, 128, 2)
        self.pool3=MaxPool(3, 2, 0)
        self.fire4=Fire(128, 32, 256)
        self.fire5=Fire(256, 32, 256)
        self.SR2=SqueezeEx(256,256,2)
        self.pool5=MaxPool(3, 2, 0)
        self.fire6=Fire(256, 48, 384)
        self.fire7=Fire(384, 48, 384)
        self.fire8=Fire(384, 64, 512)
        self.fire9=Fire(512, 64, 512)
        self.SR3=SqueezeEx(512, 512, 2)
        self.ASPP=ASPP(512)
        
        self.fire9_ASPP=FireDeconv(128, 32, 128, 128)
        self.fire10=FireDeconv(512, 64, 128, 128)
        self.fire11=FireDeconv(512,64,64,64)
        self.fire12=FireDeconv(128, 16, 32,32)
        self.fire13=FireDeconv(64, 16, 32, 32,padding=(0,257))
        self.drop=nn.Dropout2d()
        self.conv14=Conv(64, mc.NUM_CLASS, 3, (1,1),0)
        # TODO fire....
    def forward(self, x):
        conv1=self.conv1(x, False)        
        conv1_skip=self.conv1_skip(x, False)
        pool1=self.pool1(conv1)        
        fire2=self.fire2(pool1)   
        fire3=self.fire3(fire2)
        #2027
        sr_fire3=self.SR1(fire3)     
        #2091
        pool3=self.pool3(sr_fire3)
        #2223
        fire4=self.fire4(pool3)      
        #2489
        sr_fire5=self.SR2(self.fire5(fire4))        
        #2851
        pool5=self.pool5(sr_fire5) 
        # 2983
        fire6=self.fire6(pool5)      
        # 3195
        fire7=self.fire7(fire6)   
        # 3419
        fire8=self.fire8(fire7)  
        # 3723
        sr_fire9=self.SR3(self.fire9(fire8))     
        # 4085
        ASPP=self.ASPP(sr_fire9)
        #  4765
        fireASPP=self.fire9_ASPP(ASPP)# c-256
        # 4957
        fire10=self.fire10(sr_fire9) 
        fire10_fuse_=torch.add(fire10, sr_fire5) # c-256     
        fire10_fuse=torch.cat([fire10_fuse_, fireASPP], 1) # c-512          
        fire11=self.fire11(fire10_fuse) #c-128
        fire11_fuse=torch.add(fire11, sr_fire3)
        fire12=self.fire12(fire11_fuse)      
        # 6323
        fire12_fuse=torch.add(fire12, conv1)       
        fire13=self.fire13(fire12_fuse)      
        fire13_fuse=torch.add(fire13, conv1_skip)
        drop13=self.drop(fire13_fuse)
        out=self.conv14(drop13,True, False)
        # 6937
        '''
        print('conv1',conv1.size())
        print('pool1',pool1.size())
        print('fire2', fire2.size())
        print('fire3', fire3.size())
        print('pool3',pool3.size())
        print('fire4', fire4.size())
        print('fire5', sr_fire5.size())
        print('pool5',pool5.size())
        print('fire6', fire6.size())
        print('fire7', fire7.size())
        print('fire8', fire8.size())
        print('fire9', sr_fire9.size())
        print('fireASPP', fireASPP.size())
        print('fire10', fire10.size())
        print('fire10_fuse_', fire10_fuse_.size())
        print('fire10_fuse', fire10_fuse.size())
        print('fire11', fire11.size())
        print('fire11_fuse', fire11_fuse.size())
        print('fire12',fire12.size())
        print('fire12_fuse',fire12_fuse.size())
        print('fire13',fire13.size())
        print('fire13_fuse',fire13_fuse.size())
        print(out.size())
        '''
       
        return out

            
        
        
        