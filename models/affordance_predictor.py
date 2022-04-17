import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34


class AffordanceLayer(nn.Module):

    def __init__(self, input_dim=512, hidden_dim=64, output_dim=1, dropout_p=0.5, conditional=False):
        super().__init__()

        self.conditional = conditional
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if not conditional:
            self.layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.layer = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
             ) for _ in range(4)
            ])
            self.out_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, command=None):
        
        if not self.conditional:
            x = self.layer(x)
        else:
            B = x.shape[0]
            out = torch.zeros((B, self.hidden_dim)).to(x.device)
            for i in range(B):
                out[i] = self.layer[command[i]](x[i])
            
            x = self.out_layer(out)

        return x

class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self, backbone='resnet18', hidden_dim=64, dropout_p=0.5):
        super().__init__()
        
        self.backbone = self.get_backbone(backbone)

        self.lane_dist_layer = AffordanceLayer(
            input_dim=512,
            hidden_dim=hidden_dim, 
            output_dim=1, 
            dropout_p=dropout_p,
            conditional=True,
        )

        self.route_angle_layer = AffordanceLayer(
            input_dim=512,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout_p=dropout_p,
            conditional=True,
        )

        self.tl_dist_layer = AffordanceLayer(
            input_dim=512,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout_p=dropout_p,
            conditional=False,
        )

        self.tl_state_layer = AffordanceLayer(
            input_dim=512,
            hidden_dim=hidden_dim,
            output_dim=2,
            dropout_p=dropout_p,
            conditional=False,
        )


    def forward(self, img, command):
        
        B = img.shape[0]

        features = self.backbone(img)
        features = features.reshape(B, -1)

        lane_dist = self.lane_dist_layer(features, command)
        route_angle = self.route_angle_layer(features, command)
        tl_dist = self.tl_dist_layer(features, command)
        tl_state = self.tl_state_layer(features, command)

        return lane_dist, route_angle, tl_dist, tl_state
    
    def get_backbone(self, backbone):
        
        model = None
        if backbone == 'resnet18':
            model = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            model = resnet18(pretrained=True)
        
        model = nn.Sequential(*list(model.children())[:-1])

        return model
