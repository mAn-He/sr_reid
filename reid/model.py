"""BoT 기준선: ResNet-50, last stride 1, BNNeck."""
import torch
import torch.nn as nn
import torchvision


class BoT(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super().__init__()
        net = torchvision.models.resnet50(weights=None)
        if pretrained:
            net.load_state_dict(torch.load(pretrained, map_location="cpu"))
        net.layer4[0].conv2.stride = (1, 1)
        net.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        g = self.backbone(x).mean(dim=(2, 3))
        f = self.bnneck(g)
        if self.training:
            return self.classifier(f), g
        return f
