import torch
import torch.nn as nn
import timm


class Encoder(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

    def forward(self, x):
        features = self.model(x)
        features = features.permute(0, 2, 3, 1)
        return features


if __name__ == "__main__":
    images = torch.randn(32, 3, 224, 224)
    model = Encoder()
    print(model(images).shape)
