import torch
import torch.nn as nn
import torchvision

from torchvision.models import resnet101

class BackBone(nn.Module):
    def __init__(self, num_unfreeze_layers=3):
        super(BackBone, self).__init__()
        model = resnet101(weights='IMAGENET1K_V2', progress=True)
        feature_maps = list(model.children())[:8]
        
        # Adding an AdaptiveAvgPooling (batch_size, 2048, 8, 8) -> (batch_size, 2048, 1, 8)
        feature_maps.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*feature_maps)
        
        for layer in list(self.backbone.parameters())[-(num_unfreeze_layers+1):]:
            layer.requires_grad = True
    
    def forward(self, image):
        return self.backbone(image)

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.2, num_unfreeze_layers=3):
        super(CRNN, self).__init__()
        self.backbone = BackBone(num_unfreeze_layers=num_unfreeze_layers)
        
        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Dense layers
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size),
            nn.LogSoftmax(dim=2)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        # (batch_size, 2048, 1, 8) -> (batch_size, 8, 2048, 1)
        x = x.permute(0, 3, 1, 2)
        # flatten -> (batch_size, 8, 2048)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        # (batch_size, 8, vocab_size) -> (8, batch_size, vocab_size)
        x = x.permute(1, 0, 2)
        
        return x