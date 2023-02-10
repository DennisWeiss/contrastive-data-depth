import torch.nn as nn
import torchvision


class DataDepthTwinsModel(nn.Module):
    def __init__(self):
        super(DataDepthTwinsModel, self).__init__()

        self.backbone = torchvision.models.resnet18()
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

    def forward(self, x):
        return self.projector(self.backbone(x))