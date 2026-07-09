import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import PositionalEncoding2D, MultiHead, FeedForwardNetwork
import torchvision.models as models
from defs import axis_angle_to_matrix

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.up_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ])

        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        ])

    def forward(self, features):
        feats = list(features) 
        x = feats.pop()

        outputs = []
        for i in range(4):
            x = self.up_layers[i](x)
            skip = feats.pop()
            x = torch.concat([x, skip], dim=1)
            x = self.conv_layers[i](x)
            outputs.append(x)

        x = self.up_layers[4](x)
        x = self.conv_layers[4](x)
        outputs.append(x)

        return outputs

class DepthHead(nn.Module):
    def __init__(self, in_channel=32, min_disp=0.4, max_disp=5.0):
        super(DepthHead, self).__init__()
        
        self.feature_mixer = nn.Sequential( 
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.predictor = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=3, padding=1)

        self.min_disp = min_disp
        self.max_disp = max_disp

        nn.init.normal_(self.predictor.weight, std=1e-5)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, all_G):
        mixed_feat = self.feature_mixer(all_G) 
        
        out = self.predictor(mixed_feat) # [B, 1, 224, 224]

        disp_raw = torch.sigmoid(out) # 0 ~ 1
        
        scaled_disp = self.min_disp + (self.max_disp - self.min_disp) * disp_raw 
        
        return scaled_disp
    
class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.extrinsic_conv = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(2560, 6) # 회전 3, 방향 3
        )
        nn.init.normal_(self.extrinsic_conv[-1].weight, mean=0.0, std=1e-5)
        nn.init.normal_(self.extrinsic_conv[-1].bias, mean=0.0, std=1e-5)

    def forward(self, From_F, To_F):
        E_FROM_TO = self.predict_E(From_F, To_F)

        return E_FROM_TO
    
    def predict_E(self, From_F, To_F):
        B = From_F.shape[0]

        combined = torch.cat([From_F, To_F], dim=1) # [B, 512 + 512, ...]
        extrinsic_raw = self.extrinsic_conv(combined)

        axis_angle = torch.tanh(extrinsic_raw[:, :3]) * 3.14159 / 3.0 # -3.14159 ~ 3.14159
        R = axis_angle_to_matrix(axis_angle) # [B, 3, 3]
        
        translation = torch.tanh(extrinsic_raw[:, 3:6]) * 1.0
        
        E = torch.eye(4, device=From_F.device).unsqueeze(0).repeat(B, 1, 1) # [B, 4, 4]
        E[:, :3, :3] = R
        E[:, :3, 3] = translation

        return E

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 출력 채널: 64
        self.layer2 = resnet.layer2  # 출력 채널: 128
        self.layer3 = resnet.layer3  # 출력 채널: 256
        self.layer4 = resnet.layer4  # 출력 채널: 512

    def forward(self, x):

        x1 = self.layer0(x)
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        return [x1, x2, x3, x4, x5]

class MonoMirror(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.projection_head = ProjectionHead()

        self.d_min = 0.4
        self.d_max = 15.0

        self.depth_Head_16 = DepthHead(in_channel=256)
        self.depth_Head_8 = DepthHead(in_channel=128)
        self.depth_Head_4 = DepthHead(in_channel=64)
        self.depth_Head_2 = DepthHead(in_channel=32)
        self.depth_Head_1 = DepthHead(in_channel=16)

    def forward(self, prev_img, curr_img, next_img, _K, _C, sfs=False):
        B, C, H, W = curr_img.shape

        PREV = self.encoder(prev_img)
        CURR = self.encoder(curr_img)
        NEXT = self.encoder(next_img)

        K = self.get_K(_K, _C)
        E_CURR_PREV = self.projection_head(CURR[-1], PREV[-1])
        E_CURR_NEXT = self.projection_head(CURR[-1], NEXT[-1])

        CURR_F = self.decoder(CURR)

        DISP_16 = self.depth_Head_16(CURR_F[0])
        DISP_8 = self.depth_Head_8(CURR_F[1])
        DISP_4 = self.depth_Head_4(CURR_F[2])
        DISP_2 = self.depth_Head_2(CURR_F[3])
        DISP_1 = self.depth_Head_1(CURR_F[4])

        MATRIX_CURR_PREV = self.get_MATRIX(B, K, E_CURR_PREV)
        MATRIX_CURR_NEXT = self.get_MATRIX(B, K, E_CURR_NEXT)

        if sfs:
            print(f"--- [Fixed Sample Monitoring] ---")
            print(f"K : \n{K}")
            print(f"E_CURR_PREV : \n{E_CURR_PREV}")
            print(f"E_CURR_NEXT : \n{E_CURR_NEXT}")
            print(f"Z min: {(1.0 / (DISP_1 + 1e-6)).min().item():.4f}, Z max: {(1.0 / (DISP_1 + 1e-6)).max().item():.4f}, 갭: {((1.0 / (DISP_1 + 1e-6)).max() - (1.0 / (DISP_1 + 1e-6)).min()).item():.4f}")
            print(f"---------------------------------")

        return {
            'DISP' : [DISP_16, DISP_8, DISP_4, DISP_2, DISP_1],
            'MATRIX_CURR_PREV' : [MATRIX_CURR_PREV],
            'MATRIX_CURR_NEXT' : [MATRIX_CURR_NEXT],
        }

    def get_K(self, _K, _C):
        B = _K[0].shape[0]

        K = torch.zeros((B, 3, 3), device=_K[0].device)
        K[:, 0, 0] = _K[0]
        K[:, 1, 1] = _K[1]
        K[:, 0, 2] = _C[0]
        K[:, 1, 2] = _C[1]
        K[:, 2, 2] = 1.0

        return K

    def get_MATRIX(self, B, K, E):
        K44 = torch.eye(4, device=K.device).unsqueeze(0).repeat(B, 1, 1)
        K44[:, :3, :3] = K

        MATRIX = torch.bmm(K44, E)[:, :3, :]

        return MATRIX