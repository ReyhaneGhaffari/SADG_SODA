import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from Settings import get_args

args = get_args()


class Backbone(nn.Module):
    """
    MobileNet-V2 or ResNet-50 backbone feature extractor.
    Returns last convolutional feature map.
    """
    def __init__(self, backbone_name='mobilenet_v2'):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == 'mobilenet_v2':
            # use pretrained=True in your actual environment
            full_model = models.mobilenet_v2(weights=None)
            self.features = full_model.features
            self.out_channels = 1280

        elif self.backbone_name == 'resnet50':
            # use pretrained=True in your actual environment
            full_model = models.resnet50(weights=None)
            # remove avgpool and fc — keep only conv layers
            self.features = nn.Sequential(*list(full_model.children())[:-2])
            self.out_channels = 2048

        else:
            raise NotImplementedError(f"Backbone {backbone_name} not supported.")

    def forward(self, x):
        return self.features(x)


class RSAM(nn.Module):
    """
    Residual Style-Aware Module (RSAM).
    Bottleneck design: 1x1 -> 3x3 -> 1x1 with residual connection.
    Inserted at the last convolutional layer of the backbone.
    """
    def __init__(self, in_channels, compress_ratio=0.5):
        super().__init__()
        compressed = max(1, int(in_channels * compress_ratio))

        self.residual_branch = nn.Sequential(
            # 1x1 conv: channel compression
            nn.Conv2d(in_channels, compressed, kernel_size=1, bias=False),
            nn.BatchNorm2d(compressed),
            nn.ReLU(inplace=True),
            # 3x3 conv: spatial feature refinement
            nn.Conv2d(compressed, compressed, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(compressed),
            nn.ReLU(inplace=True),
            # 1x1 conv: channel restoration
            nn.Conv2d(compressed, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.residual_branch(x)


class FCLayers(nn.Module):
    """
    Shared FC layer structure for both SourceNet and TargetNet.
    Projects backbone output to 512-dimensional feature vector.
    1280 (or 2048) -> 1024 -> 512
    """
    def __init__(self, in_features, hidden=[1024, 512], dropout=0.5):
        super().__init__()
        layers = []
        for hidden_dim in hidden:
            layers += [
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ]
            in_features = hidden_dim
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class SourceNet(nn.Module):
    """
    Source network: Backbone + GAP + FC layers -> zst (512-dim)
    No RSAM module.
    """
    def __init__(self, backbone_name='mobilenet_v2',
                 fc_hidden=[1024, 512], dropout=0.5):
        super().__init__()
        self.backbone = Backbone(backbone_name)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = FCLayers(self.backbone.out_channels,
                                   fc_hidden, dropout)
        self.feat_dim = fc_hidden[-1]  # 512

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = self.gap(x).view(x.size(0), -1)
        zst = self.fc_layers(x)
        return zst


class TargetNet(nn.Module):
    """
    Target network: Backbone + RSAM + GAP + FC layers -> zt (512-dim)
    RSAM inserted between backbone and GAP.
    FC layers are independent from SourceNet.
    """
    def __init__(self, backbone_name='mobilenet_v2',
                 fc_hidden=[1024, 512], dropout=0.5,
                 compress_ratio=0.5):
        super().__init__()
        self.backbone = Backbone(backbone_name)
        self.rsam = RSAM(self.backbone.out_channels, compress_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = FCLayers(self.backbone.out_channels,
                                   fc_hidden, dropout)
        self.feat_dim = fc_hidden[-1]  # 512

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        # backbone features
        x_backbone = self.backbone(x)
        # RSAM: residual style modulation
        x_rsam = self.rsam(x_backbone)
        # GAP + flatten
        x_gap = self.gap(x_rsam).view(x_rsam.size(0), -1)
        # FC layers -> zt
        zt = self.fc_layers(x_gap)
        return zt


class ClosedSetClassifier(nn.Module):
    """
    Closed-set classifier Ccs.
    Single FC layer: 512 -> C (known classes) with softmax.
    Frozen during target adaptation.
    """
    def __init__(self, feat_dim=512, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


class OpenSetClassifier(nn.Module):
    """
    Open-set classifier Cos.
    Single FC layer: 512 -> C+1 (known + unknown).
    Includes learnable temperature scaling tau.
    """
    def __init__(self, feat_dim=512, num_classes=4, tau_init=0.1):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes + 1)
        self.tau = nn.Parameter(torch.tensor(tau_init))

    def forward(self, x):
        logits = self.fc(x) / self.tau
        return F.softmax(logits, dim=1)

    def get_logits(self, x):
        return self.fc(x) / self.tau


def build_models(args):
    """
    Build and return all models as a dictionary.
    """
    source_net = SourceNet(
        backbone_name=args.backbone,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout
    )
    target_net = TargetNet(
        backbone_name=args.backbone,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
        compress_ratio=args.rsam_ratio
    )
    ccs = ClosedSetClassifier(
        feat_dim=args.fc_hidden[-1],
        num_classes=args.num_class
    )
    cos = OpenSetClassifier(
        feat_dim=args.fc_hidden[-1],
        num_classes=args.num_class,
        tau_init=args.tau_init
    )
    return {
        'source_net': source_net,
        'target_net': target_net,
        'ccs': ccs,
        'cos': cos
    }


if __name__ == '__main__':
    args = get_args()
    models = build_models(args)

    # test forward pass
    x = torch.randn(4, 3, 224, 224)

    # source net
    zst = models['source_net'](x)
    print(f"zst shape: {zst.shape}")  # [4, 512]

    # target net
    zt = models['target_net'](x)
    print(f"zt shape: {zt.shape}")  # [4, 512]

    # closed-set classifier
    pcs = models['ccs'](zst)
    print(f"pcs shape: {pcs.shape}")  # [4, 4]

    # open-set classifier
    pos = models['cos'](zt)
    print(f"pos shape: {pos.shape}")  # [4, 5]

    print("\nAll models built and tested successfully.")
    print(f"Feature dimension (zst, zt): {zst.shape[1]}")
    print(f"Closed-set outputs (C): {pcs.shape[1]}")
    print(f"Open-set outputs (C+1): {pos.shape[1]}")
