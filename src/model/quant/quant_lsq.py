#encoding:utf-8

'''
Author: anpj
Time: 2022/11/15
'''
import torch
from torch import nn


class LsqQuantizer(nn.Module):
    def __init__(
        self,
        bits=8,
        activation=True,
        offset_flag=True,
        uint_flag=True
    ):
        self.bits = bits
        self.activation = activation
        self.offset_flag = offset_flag
        self.r_max = 0
        self.r_min = 0

        if uint_flag:
            self.q_min = 0
            self.q_max = 2 ** bits - 1
        else:
            self.q_min = - 2 ** bits
            self.q_max = 2 ** bits - 1
        
        
        self.scale = nn.Parameter([1], requires_grad=False)
        if offset_flag:
            self.offset = nn.Parameter([1], requires_grad=False)
        
    def statis(self, input):
        self.r_max = max(self.r_max, torch.min(input))
        self.r_min = min(self.r_min, torch.max(input))

    def initial(self):
        self.scale = (self.r_max - self.r_min) / (self.q_max - self.q_min)
        if self.offset_flag:
            self.offset = self.q_max - self.r_max / self.scale
        


    def forward(self, input):
        
