#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Author: Weijie Lin
# @Email : berneylin@gmail.com
# @Date  : 2019-01-19

import os
import numpy as np
import time
import sys
from configs import *
from densenet_models import *
import torch
import torch.nn.parameter as Parameter
from densenet_models import * 
from collections import OrderedDict

def Convert(a):
    it = iter(a)
    res_dct = dict(zip(it, it))
    return res_dct


def main2():
    # start_train()
    network_model = DenseNet121(14, True).cuda()
    network_model = torch.nn.DataParallel(network_model).cuda()  # make model available multi GPU cores training
    model_checkpoint = torch.load("model0122.pth", map_location = "cpu")
    for (key, val) in list(model_checkpoint["state_dict"].items()):
      value = val.clone()
      del model_checkpoint["state_dict"][key]
      model_checkpoint["state_dict"][key.replace('module.', '')] = value
    network_model.load_state_dict(model_checkpoint["state_dict"])

    # print(type(model_checkpoint["state_dict"]))
    # model_checkpoint["state_dict"].popitem(last = True)
    # model_checkpoint["state_dict"].popitem(last = True)
    # model_checkpoint["state_dict"] = model_checkpoint["state_dict"].module
    # network_model.load_state_dict(model_checkpoint["state_dict"])
    
    
    # print(model_checkpoint["state_dict"].items())

    print("complete!")


class new_chexnet(nn.Module):
  def __init__(self, num_classes = 19, checkpoint = None):
    super(new_chexnet, self).__init__()
    self.densenet = DenseNet121().cuda()
    self.densenet = torch.nn.DataParallel(self.densenet).cuda()
    # print(list(self.densenet.dense_net_121.classifier.in_features))
    
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint.items():
    #     name = k.replace("module.densenet.module.", "")  # remove module.
    #     new_state_dict[name] = v
    # checkpoint = new_state_dict
    
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

def main():
  model_checkpoint = torch.load("CheXNet-153516-15032021loss0.0896397.pth.tar")
  model = new_chexnet(checkpoint = model_checkpoint["state_dict"])

  inp = torch.randn(1, 3, 224, 224).cuda()
  print(inp.shape)
  op = model(inp)
  print(op.shape)
  # print(model)


if __name__ == '__main__':
    main()
