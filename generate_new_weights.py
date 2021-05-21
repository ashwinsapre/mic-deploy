import os
import numpy as np
import sys
from configs import *
from densenet_models import *
import torch
import torch.nn.parameter as Parameter
from collections import OrderedDict

class new_chexnet(nn.Module):
  def __init__(self, num_classes = 19, checkpoint = None):
    super(new_chexnet, self).__init__()
    self.densenet = DenseNet121().cuda()
    self.densenet = torch.nn.DataParallel(self.densenet).cuda()
    
    if checkpoint is not None:
      self.densenet.load_state_dict(checkpoint)
      print("Weights loaded!")

    num_fc_kernels = 1024
    self.densenet.module.dense_net_121.classifier = nn.Sequential(nn.Linear(num_fc_kernels, num_classes), nn.Softmax())
    self.densenet = self.densenet.cuda()
    print("Model ready!")

  def forward(self, inp):
    op = self.densenet(inp)
    return op
