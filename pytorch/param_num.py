import torch
import numpy as np
from alisuretool.Tools import Tools
from torchvision.models import vgg16, vgg16_bn, resnet18, resnet34, resnet50


class ParamNum(object):

    def __init__(self, model_fn=vgg16_bn):
        self.model = model_fn()
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
    ParamNum(vgg16).info()
    ParamNum(vgg16_bn).info()
    ParamNum(resnet18).info()
    ParamNum(resnet34).info()
    ParamNum(resnet50).info()
    pass
