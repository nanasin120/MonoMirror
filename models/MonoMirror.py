import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import PositionalEncoding2D, MultiHead, FeedForwardNetwork
from defs import axis_angle_to_matrix

class FeatureUpsampler(nn.Module):
    def __init__(self, in_channels=384, out_channels=32):
        super().__init__()
        self.in_channels = in_channels

        rgb_ch_list = [128, 64, 32, 16, 8] 
        up_ch_list = [256, 128, 64, out_channels]
        gn_groups_list = [16, 8, 8, 4]

        self.gates = nn.ModuleList()
        for ch in rgb_ch_list:
            gate = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(ch, 1, kernel_size=1),
                nn.Sigmoid()
            )
            nn.init.constant_(gate[-2].weight, 0.0)
            nn.init.constant_(gate[-2].bias, -2.0)
            self.gates.append(gate)

        self.fuse0 = nn.Sequential(
            nn.Conv2d(in_channels + rgb_ch_list[0], in_channels, kernel_size=3, padding=1), 
            nn.GELU()
        )

        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.fuses = nn.ModuleList()

        prev_ch = in_channels
        for i in range(4):
            curr_up_ch = up_ch_list[i]
            curr_rgb_ch = rgb_ch_list[i + 1] # 다음 단계의 RGB 피처 채널

            # UP
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(prev_ch, curr_up_ch, kernel_size=3, padding=1),
                nn.GroupNorm(gn_groups_list[i], curr_up_ch), 
                nn.GELU()
            ))

            # SKIP
            self.skips.append(nn.Sequential(
                nn.Conv2d(in_channels, curr_up_ch, kernel_size=1)
            ))

            # FUSE
            fuse_in_ch = curr_up_ch + curr_up_ch + curr_rgb_ch
            self.fuses.append(nn.Sequential(
                nn.Conv2d(fuse_in_ch, curr_up_ch, kernel_size=3, padding=1),
                nn.GELU()
            ))
            
            prev_ch = curr_up_ch

    def forward(self, DINO_F, IMG_F, PATCH_H, PATCH_W):
        B = DINO_F[-1].shape[0]

        DINO_F_spatial = [f.transpose(1, 2).view(B, self.in_channels, PATCH_H, PATCH_W) for f in DINO_F]
        x = DINO_F_spatial[-1] # [B, 384, PATCH_H, PATCH_W]
        
        # DINO_F 역순 정렬 (F_3, F_2, F_1, F_0)
        skip_features = list(reversed(DINO_F_spatial))

        # RGB Feature 역순 정렬 (14, 28, 56, 112, 224)
        IMG_F_rev = list(reversed(IMG_F))

        # Gate 적용
        GATED_IMG_F = [img_f * gate(img_f) for img_f, gate in zip(IMG_F_rev, self.gates)]

        # Fuse0은 예외사항이기에 따로 처리
        x = torch.cat([x, GATED_IMG_F[0]], dim=1)
        x = self.fuse0(x)

        outputs = []
        for i in range(4):
            x = self.ups[i](x)
            
            # Skip Feature
            s = self.skips[i](skip_features[i])
            s = F.interpolate(s, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            
            # 결합 및 Fuse
            x = torch.cat([x, s, GATED_IMG_F[i + 1]], dim=1)
            x = self.fuses[i](x)
            
            outputs.append(x)

        # [out_28, out_56, out_112, out_224]
        return outputs

class Decoder(nn.Module):
    def __init__(self, d_model=768, h=12):
        super(Decoder, self).__init__()
        self.h = h
        self.d_model = d_model

        self.proj_to_bias = nn.Linear(16, d_model)

        self.p2_norm = nn.LayerNorm(self.d_model)
        
        self.self_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_0 = nn.LayerNorm(self.d_model)

        self.cross_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_1 = nn.LayerNorm(self.d_model)

        self.FFN = FeedForwardNetwork(self.d_model, self.d_model * 4)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)

    def forward(self, p1, p2, matrix):
        B = matrix.shape[0]

        matrix_bias = self.proj_to_bias(matrix.view(B, -1))
        matrix_bias = matrix_bias.unsqueeze(1)

        norm_p1 = self.layer_norm_0(p1)
        norm_p2 = self.p2_norm(p2)
        after_self_attention = self.self_attention(norm_p1, norm_p1, norm_p1)
        after_self_attention = after_self_attention + p1

        norm_cross = self.layer_norm_1(after_self_attention)
        after_cross_attention = self.cross_attention(norm_cross, norm_p2 + matrix_bias, norm_p2)
        after_cross_attention = after_cross_attention + after_self_attention
        
        norm_ffn = self.layer_norm_2(after_cross_attention)
        after_ffn = self.FFN(norm_ffn)
        after_ffn = after_ffn + after_cross_attention

        return after_ffn
    
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
    def __init__(self, d_model=768):
        super(ProjectionHead, self).__init__()

        self.extrinsic_conv = nn.Sequential(
            nn.Conv2d(d_model * 3, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 6) # 회전 3, 방향 3
        )
        # 여기 0으로 초기화했을때는 1.0으로 고정되었었는데 지금은 아님
        nn.init.normal_(self.extrinsic_conv[-1].weight, mean=0.0, std=1e-5)
        nn.init.normal_(self.extrinsic_conv[-1].bias, mean=0.0, std=1e-5)

    def forward(self, prev_F, curr_F, next_F, curr_K, PATCH_H, PATCH_W, H, W):
        K = self.predict_K(curr_K, H, W)
        E_CURR_PREV = self.predict_E(curr_F, prev_F, PATCH_H, PATCH_W)
        E_CURR_NEXT = self.predict_E(curr_F, next_F, PATCH_H, PATCH_W)
        E_CURR_PREV_INV = self.predict_E(prev_F, curr_F, PATCH_H, PATCH_W)
        E_CURR_NEXT_INV = self.predict_E(next_F, curr_F, PATCH_H, PATCH_W)

        return K, E_CURR_PREV, E_CURR_NEXT, E_CURR_PREV_INV, E_CURR_NEXT_INV
    
    def predict_K(self, curr_K, H, W):
        B = curr_K[0].shape[0]

        K = torch.zeros((B, 3, 3), device=curr_K[0].device)
        K[:, 0, 0] = curr_K[0]
        K[:, 1, 1] = curr_K[1]
        K[:, 0, 2] = W / 2.0
        K[:, 1, 2] = H / 2.0
        K[:, 2, 2] = 1.0

        return K
    
    def predict_E(self, F1, F2, PATCH_H, PATCH_W):
        B = F1.shape[0]

        # DINOv2는 패치가 16x16으로 나옴
        F1_spatial = F1.transpose(1, 2).view(B, -1, PATCH_H, PATCH_W)
        F2_spatial = F2.transpose(1, 2).view(B, -1, PATCH_H, PATCH_W)
        Diff_spatial = F1_spatial - F2_spatial

        combined = torch.cat([F1_spatial, F2_spatial, Diff_spatial], dim=1) # [B, d_model * 2, PATCH, PATCH]
        extrinsic_raw = self.extrinsic_conv(combined)

        axis_angle = torch.tanh(extrinsic_raw[:, :3]) * 3.14159 / 3.0 # -3.14159 ~ 3.14159
        R = axis_angle_to_matrix(axis_angle) # [B, 3, 3]
        
        translation = torch.tanh(extrinsic_raw[:, 3:6]) * 1.0
        
        E = torch.eye(4, device=F1.device).unsqueeze(0).repeat(B, 1, 1) # [B, 4, 4]
        E[:, :3, :3] = R
        E[:, :3, 3] = translation

        return E

class RGBFeatureHead(nn.Module):
    def __init__(self):
        super(RGBFeatureHead, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )

    def forward(self, img):
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        return [x1, x2, x3, x4, x5]

class DINOv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # 3번째, 6번째, 9번째, 12번째 층
        self.out_blocks = [2, 5, 8, 11] 

    def forward(self, x):
        # x: [B, 3, 224, 224] 크기의 원본 이미지
        
        features = self.backbone.get_intermediate_layers(
            x, 
            n=self.out_blocks, 
            reshape=False
        )
        
        # [B, 256, 384] (Patch 개수 256개, 채널 384)
        
        F_list = features
        G_curr = features[-1]

        return G_curr, F_list

class MonoMirror(nn.Module):
    def __init__(self, H=224, W=224):
        super().__init__()

        self.encoder = DINOv2Encoder()

        self.PATCH_H = H // 16
        self.PATCH_W = W // 16

        self.DINO_H = self.PATCH_H * 14
        self.DINO_W = self.PATCH_W * 14
        
        self.patch_embedded_dim = 384

        self.positionalEncoding2D = PositionalEncoding2D(d_model=self.patch_embedded_dim, h_patches=self.PATCH_H, w_patches=self.PATCH_W)

        self.projection_head = ProjectionHead(d_model=self.patch_embedded_dim)

        self.decoder_layers = 8
        self.decoders = nn.ModuleList([
            Decoder(d_model=self.patch_embedded_dim, h=12) for _ in range(self.decoder_layers)
        ])

        self.rgbFeatureHead = RGBFeatureHead()

        self.upsampler = FeatureUpsampler(in_channels=self.patch_embedded_dim)

        self.d_min = 0.4
        self.d_max = 15.0

        self.depth_Head_28 = DepthHead(in_channel=256)
        self.depth_Head_56 = DepthHead(in_channel=128)
        self.depth_Head_112 = DepthHead(in_channel=64)
        self.depth_Head_224 = DepthHead(in_channel=32)

    def forward(self, prev_img, curr_img, next_img, curr_K, sfs=False):
        # image : [B, 3, H, W] [B, 3, 448, 448] 위 아래가 검정색으로 패딩된 상태
        B, C, H, W = curr_img.shape

        prev_img_enc = F.interpolate(prev_img, size=(self.DINO_H, self.DINO_W), mode='bilinear', align_corners=False)
        curr_img_enc = F.interpolate(curr_img, size=(self.DINO_H, self.DINO_W), mode='bilinear', align_corners=False)
        next_img_enc = F.interpolate(next_img, size=(self.DINO_H, self.DINO_W), mode='bilinear', align_corners=False)

        # [B, PATCH_H * PATCH_W, 384]

        PREV_G, _ = self.encoder(prev_img_enc)
        CURR_G, CURR_F = self.encoder(curr_img_enc)
        NEXT_G, _ = self.encoder(next_img_enc)

        K, E_CURR_PREV, E_CURR_NEXT, E_CURR_PREV_INV, E_CURR_NEXT_INV = self.projection_head(PREV_G, CURR_G, NEXT_G, curr_K, self.PATCH_H, self.PATCH_W, H, W)
        
        PREV_G = self.positionalEncoding2D(PREV_G)
        CURR_G = self.positionalEncoding2D(CURR_G)
        NEXT_G = self.positionalEncoding2D(NEXT_G)

        for decoder in self.decoders:
            tmp_curr = CURR_G
            
            curr_from_prev = decoder(tmp_curr, PREV_G, E_CURR_PREV)
            curr_from_next = decoder(tmp_curr, NEXT_G, E_CURR_NEXT)

            CURR_G = (curr_from_prev + curr_from_next) / 2.0
            PREV_G = decoder(PREV_G, tmp_curr, E_CURR_PREV_INV)
            NEXT_G = decoder(NEXT_G, tmp_curr, E_CURR_NEXT_INV)

        IMG_F = self.rgbFeatureHead(curr_img)

        CURR_UP_F = self.upsampler(CURR_F, IMG_F, self.PATCH_H, self.PATCH_W)

        DISP_28 = F.interpolate(self.depth_Head_28(CURR_UP_F[0]), (H, W), mode='bilinear', align_corners=False)
        DISP_56 = F.interpolate(self.depth_Head_56(CURR_UP_F[1]), (H, W), mode='bilinear', align_corners=False)
        DISP_112 = F.interpolate(self.depth_Head_112(CURR_UP_F[2]), (H, W), mode='bilinear', align_corners=False)
        DISP_224 = self.depth_Head_224(CURR_UP_F[3])

        PREV_G = PREV_G.transpose(1, 2).view(B, self.patch_embedded_dim, self.PATCH_H, self.PATCH_W)
        CURR_G = CURR_G.transpose(1, 2).view(B, self.patch_embedded_dim, self.PATCH_H, self.PATCH_W)
        NEXT_G = NEXT_G.transpose(1, 2).view(B, self.patch_embedded_dim, self.PATCH_H, self.PATCH_W)

        MATRIX_CURR_PREV, MATRIX_INV_CURR_PREV = self.get_MATRIX(B, K, E_CURR_PREV, E_CURR_PREV_INV)
        MATRIX_CURR_NEXT, MATRIX_INV_CURR_NEXT = self.get_MATRIX(B, K, E_CURR_NEXT, E_CURR_NEXT_INV)

        if sfs:
            print(f"--- [Fixed Sample Monitoring] ---")
            print(f"True fx: {K[0, 0, 0].item():.2f}, True fy: {K[0, 1, 1].item():.2f}")
            print(f"K : \n{K}")
            print(f"E_CURR_PREV : \n{E_CURR_PREV}")
            print(f"E_CURR_NEXT : \n{E_CURR_NEXT}")
            print(f"Z min: {(1.0 / (DISP_224 + 1e-6)).min().item():.4f}, Z max: {(1.0 / (DISP_224 + 1e-6)).max().item():.4f}, 갭: {((1.0 / (DISP_224 + 1e-6)).max() - (1.0 / (DISP_224 + 1e-6)).min()).item():.4f}")
            print(f"---------------------------------")

        return {
            'DISP' : [DISP_28, DISP_56, DISP_112, DISP_224],
            'MATRIX_CURR_PREV' : [MATRIX_CURR_PREV, MATRIX_INV_CURR_PREV],
            'MATRIX_CURR_NEXT' : [MATRIX_CURR_NEXT, MATRIX_INV_CURR_NEXT],
        }

    def get_MATRIX(self, B, K, E, E_INV):
        K44 = torch.eye(4, device=K.device).unsqueeze(0).repeat(B, 1, 1)
        K44[:, :3, :3] = K

        MATRIX = torch.bmm(K44, E)[:, :3, :]
        MATRIX_INV = torch.bmm(K44, E_INV)[:, :3, :]

        return MATRIX, MATRIX_INV