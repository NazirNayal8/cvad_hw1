import torch.nn as nn
import torch
from torchvision.models import resnet18, resnet34


def one_hot_vector(x, sz):

    B = x.shape[0]

    vec = torch.zeros((B, sz))
    vec[torch.arange(B), x] = 1

    return vec.to(x.device)


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self, backbone='resnet18', cond_module='command_input', dropout_p=0.5):
        super().__init__()

        self.cond_module = cond_module
        self.backbone = self.get_backbone(backbone)
        self.speed_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.speed_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.combine_layer = nn.Linear(512 + 128, 512)

        if cond_module == 'command_input':
            self.control = nn.Sequential(
                nn.Linear(512 + 4, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Dropout(dropout_p),
                nn.ReLU(),
                nn.Linear(256, 3)    
            )
        
        elif cond_module == 'branched':
            self.control = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.Dropout(dropout_p),
                    nn.ReLU(),
                    nn.Linear(256, 3)    
                ) for _ in range(4)
            ])
            
        else:
            raise Exception(f'Undefined Conditional Module: {cond_module}')


    def forward(self, img, v, command):

        B, _, _, _ = img.shape
        
        x = self.backbone(img)
        
        x = x.reshape(B, -1)

        v_pred = self.speed_predictor(x)

        v_x = self.speed_module(v.unsqueeze(1))


        concat = self.combine_layer(torch.cat([x, v_x], dim=1))

        if self.cond_module == 'command_input':
            
            command_vec = one_hot_vector(command, 4)
            concat = torch.cat([concat, command_vec], dim=1)
            out = self.control(concat)
        
        else:
            
            out = torch.zeros((B, 3)).type(torch.FloatTensor).to(x.device)
            
            for i in range(B):
                out[i] = self.control[command[i]](concat[i])


        throttle_out = torch.sigmoid(out[:, 0])
        steer_out = torch.tanh(out[:, 1])
        brake_out = torch.sigmoid(out[:, 2])

        return throttle_out, steer_out, brake_out, v_pred


    def get_backbone(self, backbone):
        
        model = None
        if backbone == 'resnet18':
            model = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            model = resnet18(pretrained=True)
        
        model = nn.Sequential(*list(model.children())[:-1])

        return model