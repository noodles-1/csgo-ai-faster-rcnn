from torch import nn

def build_model(backbone, num_classes):
    num_ftrs = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes + 1)
    )
    return backbone