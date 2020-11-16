import torch
import numpy as np
from alisuretool.Tools import Tools
from torchvision.models import vgg16


class ParamNum(object):

    def __init__(self):
        self.model = vgg16()
        pass

    def info(self):
        Tools.print()
        from thop import profile
        flops, params = profile(self.model, inputs=(torch.randn(1, 3, 224, 224), ), verbose=False)
        Tools.print("profile: param={}, flops={}".format(params, flops))
        Tools.print("param={}".format(self.params_count(self.model)))
        Tools.print()
        pass

    @staticmethod
    def params_count(model):
        return np.sum([p.numel() for p in model.parameters()]).item()

    pass


if __name__ == '__main__':
    param_num = ParamNum()
    param_num.info()
    pass
