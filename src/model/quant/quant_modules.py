#encoding:utf-8

'''
Author: anpj
Time: 2022/11/15
'''
import torch
from torch import nn

from quant_lsq import 

class QuantConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode="zeros",
        steps=200,
    ):
        super(QuantConv2d, self).__init__(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                    padding_mode,
                )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.activation_quantizer = None
        self.weight_quantizer = None
        self.steps = steps
        self.initial_flag = False

    def forward(self, input):
        if not self.initial_flag:
            pass
        else:
            quant_input = self.activation_quantizer(input)
            quqnt_weight = self.weight_quantizer(self.weight)

            output = F.conv2d(quant_input, quant_weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)
        return output



def add_quant_ops(modules):
    for name, module in modules.named_children():
        print(name, module)
        if isinstance(module, nn.Conv2d):
            pass
        elif isinstance(module, nn.Linear):
            pass
        elif isinstance(module, nn.ConvTranspose2d):
            pass
        else:
            add_quant_ops(module)


if __name__ == "__main__":
    # 1. 导入model
    from ..gcrn.model import Net
    model = Net()

    # 2. 开始量化


