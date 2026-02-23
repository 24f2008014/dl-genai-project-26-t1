import torch
from torchvision import models
import torch.nn as nn

class ResNet(nn.Module):
    """
    Input: (B, 1, n_mels, T)
    
    Output: logits (B, num_classes)
    """
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        super().__init__()
        res = models.resnet18(weights="DEFAULT" if pretrained else None)
        old_conv = res.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )

        # Taking the mean of the old 3-channel input weights and converting it to 1 channel
        with torch.inference_mode():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        res.conv1 = new_conv
        self.backbone = res

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for m in self.backbone.modules(): # Set batchnorm layers to eval if freezing!
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove final FC layer from backbone
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor, return_features=False) -> torch.Tensor:
        features = self.backbone(x) # (B, feat_dim)
        if return_features:
            return features
        logits = self.classifier(features) # (B, num_classes)
        return logits
    
class EffNetB0(nn.Module):
    """
    Input: (B, 1, n_mels, T)
    
    Output: logits (B, num_classes)
    """
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        first_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=(first_conv.bias is not None)
        )

        with torch.inference_mode():
            old_w = first_conv.weight  # (out_ch, 3, k, k)
            new_conv.weight[:] = old_w.mean(dim=1, keepdim=True)
            if first_conv.bias is not None:
                new_conv.bias[:] = first_conv.bias
        self.backbone.features[0][0] = new_conv

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for m in self.backbone.modules(): # Set batchnorm layers to eval if freezing!
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        last_linear = self.backbone.classifier[-1]
        feat_dim = last_linear.in_features
            
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor, return_features=False) -> torch.Tensor:
        # x: (B, 1, H, W)
        features = self.backbone(x) # (B, feat_dim)
        if return_features:
            return features
        logits = self.classifier(features) # (B, num_classes)
        return logits

# For defining the scratch CNN
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2))
        )
    def forward(self, x):
        return self.net(x)

class ScratchCNN(nn.Module):
    """
    Input: (B, 1, n_mels, T)

    Output: logits (B, num_classes)
    """
    def __init__(self, num_classes=10, in_ch=1, base_ch=32, dropout=0.4):
        super().__init__()
        # base_ch: 32 -> 64 -> 128 -> 256
        self.block1 = ConvBlock(in_ch, base_ch)
        self.block2 = ConvBlock(base_ch, base_ch*2)
        self.block3 = ConvBlock(base_ch*2, base_ch*4)
        # Extra conv layer to reduce spatial dimensions
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))   # (B, C, 1, 1)
        )

        feat_dim = base_ch * 8
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout*0.5),
            nn.Linear(512, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.conv4(x) # (B, C, 1, 1)
        x = x.view(x.size(0), -1) # (B, feat_dim)
        logits = self.classifier(x)
        return logits
    
class ScratchCRNN(nn.Module):
    """
    CRNN: Conv front-end (ScratchCNN) + biGRU head + classifier.
    Input: (B, 1, n_mels, T), Output: logits (B, num_classes)
    """
    def __init__(self, num_classes=10, in_ch=1, base_ch=32, lstm_hidden=128, lstm_layers=1, dropout=0.4):
        super().__init__()
        self.block1 = ConvBlock(in_ch, base_ch) # -> (B, base_ch, F/2, T/2)
        self.block2 = ConvBlock(base_ch, base_ch*2) # -> (B, base_ch*2, F/4, T/4)
        self.block3 = ConvBlock(base_ch*2, base_ch*4) # -> (B, base_ch*4, F/8, T/8)

        self.conv_time = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(inplace=True)
            # pool over frequency here, not time
        )

        # BiLSTM head: input size = channels after freq-pooling
        self.lstm_input_dim = base_ch * 8
        # self.lstm = nn.LSTM(
        #     input_size=self.lstm_input_dim,
        #     hidden_size=lstm_hidden,
        #     num_layers=lstm_layers,
        #     batch_first=True,
        #     bidirectional=True,
        #     dropout=0.0 if lstm_layers == 1 else 0.1
        # )
        self.lstm = nn.GRU(input_size=self.lstm_input_dim,
                   hidden_size=lstm_hidden,
                   num_layers=lstm_layers,
                   batch_first=True,
                   bidirectional=True)

        # classifier: pool LSTM outputs in time (mean) -> MLP
        feat_dim = lstm_hidden * 2 # bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout*0.5),
            nn.Linear(512, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 1, n_mels, T)
        returns logits: (B, num_classes)
        """
        x = self.block1(x)          # (B, C1, F1, T1)
        x = self.block2(x)          # (B, C2, F2, T2)
        x = self.block3(x)          # (B, C3, F3, T3)
        x = self.conv_time(x)       # (B, C4, F3, T3)  (keep time axis)

        # pool across frequency to produce per-frame features
        # x.mean(dim=2) collapses frequency dimension -> (B, C4, T3)
        x = x.mean(dim=2)           # (B, C, T)
        # transpose to (B, T, C) for LSTM
        x = x.permute(0, 2, 1).contiguous()  # (B, T, C)

        # LSTM expects (B, T, input_size)
        lstm_out, _ = self.lstm(x)  # (B, T, 2*lstm_hidden)
        # temporal pooling: mean over time
        feat = lstm_out.mean(dim=1) # (B, 2*lstm_hidden)

        logits = self.classifier(feat)
        return logits