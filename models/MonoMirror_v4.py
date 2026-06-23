import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CroCo import CroCo
from models.blocks import PositionalEncoding2D, MultiHead, FeedForwardNetwork
from defs import axis_angle_to_matrix

class Decoder(nn.Module):
    def __init__(self, d_model=768, h=12):
        super(Decoder, self).__init__()
        self.h = h
        self.d_model = d_model

        self.proj_cv = nn.Conv2d(64, d_model, kernel_size=1)

        self.proj_to_bias = nn.Linear(16, d_model)

        self.p2_norm = nn.LayerNorm(self.d_model)
        
        self.self_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_0 = nn.LayerNorm(self.d_model)

        self.cross_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_1 = nn.LayerNorm(self.d_model)

        self.FFN = FeedForwardNetwork(self.d_model, self.d_model * 4)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)

    def forward(self, p1, p2, cost_volume):
        B = p1.shape[0] 

        if cost_volume is None:
            matrix_bias = 0
        else:
            cv_features = self.proj_cv(cost_volume)
            matrix_bias = cv_features.flatten(2).transpose(1, 2)

        norm_p1 = self.layer_norm_0(p1)
        norm_p2 = self.p2_norm(p2)
        after_self_attention = self.self_attention(norm_p1, norm_p1, norm_p1)
        after_self_attention = after_self_attention + p1

        norm_cross = self.layer_norm_1(after_self_attention)
        after_cross_attention = self.cross_attention(norm_cross + matrix_bias, norm_p2, norm_p2)
        after_cross_attention = after_cross_attention + after_self_attention
        
        norm_ffn = self.layer_norm_2(after_cross_attention)
        after_ffn = self.FFN(norm_ffn)
        after_ffn = after_ffn + after_cross_attention

        return after_ffn
    
class DepthHead(nn.Module):
    def __init__(self, in_channel=32, min_disp=0.4, max_disp=5.0):
        super(DepthHead, self).__init__()
        
        # 물결 제거를 목표로 DepthHead 변경
        self.feature_mixer = nn.Sequential( 
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.predictor = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=1)

        self.min_disp = min_disp
        self.max_disp = max_disp

        nn.init.normal_(self.predictor.weight, std=1e-5)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, all_G):
        mixed_feat = self.feature_mixer(all_G) 
        
        out = self.predictor(mixed_feat) # [B, 1, 224, 224]

        disp_raw = torch.sigmoid(out) # 0 ~ 1
        
        scaled_disp = self.min_disp + (self.max_disp - self.min_disp) * disp_raw 
        
        # 1 / min_disp 아니면
        # 1 / max_disp 이거임
        Z_coord_safe = 1.0 / scaled_disp 

        return Z_coord_safe, scaled_disp
    
class ProjectionHead(nn.Module):
    def __init__(self, d_model=768):
        super(ProjectionHead, self).__init__()
        self.img_size = 224

        # self.intrinsic_mlp = nn.Sequential(
        #     nn.Linear(d_model, 256),
        #     nn.GELU(),
        #     nn.Linear(256, 2)
        # )
        # nn.init.normal_(self.intrinsic_mlp[-1].weight, mean=0.0, std=1e-5)
        # nn.init.zeros_(self.intrinsic_mlp[-1].bias)

        self.extrinsic_conv = nn.Sequential(
            nn.Conv2d(d_model * 3, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 6) # 회전 3, 방향 3, 앵커 5
        )
        # 여기 0으로 초기화했을때는 1.0으로 고정되었었는데 지금은 아님
        nn.init.normal_(self.extrinsic_conv[-1].weight, mean=0.0, std=0.01) # std를 1e-5에서 0.01로 확 키웁니다!
        nn.init.normal_(self.extrinsic_conv[-1].bias, mean=0.0, std=0.01)

    def forward(self, prev_F, curr_F, next_F):
        K = self.predict_K(prev_F, curr_F, next_F)
        E_CURR_PREV = self.predict_E(curr_F, prev_F)
        E_CURR_NEXT = self.predict_E(curr_F, next_F)
        E_CURR_PREV_INV = self.predict_E(prev_F, curr_F)
        E_CURR_NEXT_INV = self.predict_E(next_F, curr_F)

        return K, E_CURR_PREV, E_CURR_NEXT, E_CURR_PREV_INV, E_CURR_NEXT_INV
    
    def predict_K(self, prev_F, curr_F, next_F):
        B = curr_F.shape[0]

        # prev_F_mean = prev_F.mean(dim=1)
        # curr_F_mean = curr_F.mean(dim=1)
        # next_F_mean = next_F.mean(dim=1)

        # combined_mean = (prev_F_mean + curr_F_mean + next_F_mean) / 3.0

        # intrinsic_raw = self.intrinsic_mlp(combined_mean)

        # f = torch.tanh(intrinsic_raw) * 50.0 + 160.0 # 100 ~ 300
        # fx, fy = f[:, 0], f[:, 1]

        # fx = F.softplus(intrinsic_raw[:, 0]) + 150.0  # 최소 100 픽셀 보장
        # fy = F.softplus(intrinsic_raw[:, 1]) + 150.0
        
        K = torch.zeros((B, 3, 3), device=curr_F.device)
        K[:, 0, 0] = 315.0
        K[:, 1, 1] = 315.0
        K[:, 0, 2] = self.img_size / 2.0  # cx (정중앙 고정)
        K[:, 1, 2] = self.img_size / 2.0  # cy (정중앙 고정)
        K[:, 2, 2] = 1.0

        return K
    
    def predict_E(self, F1, F2):
        B = F1.shape[0]

        # DINOv2는 패치가 16x16으로 나옴
        F1_spatial = F1.permute(0, 2, 1).view(B, -1, 14, 14)
        F2_spatial = F2.permute(0, 2, 1).view(B, -1, 14, 14)
        Diff_spatial = F1_spatial - F2_spatial

        combined = torch.cat([F1_spatial, F2_spatial, Diff_spatial], dim=1) # [B, d_model * 2, 16, 16]
        extrinsic_raw = self.extrinsic_conv(combined)

        axis_angle = torch.tanh(extrinsic_raw[:, :3]) * 3.14159 / 3.0 # -3.14159 ~ 3.14159
        R = axis_angle_to_matrix(axis_angle) # [B, 3, 3]
        
        translation = torch.tanh(extrinsic_raw[:, 3:6]) * 0.1
        
        E = torch.eye(4, device=F1.device).unsqueeze(0).repeat(B, 1, 1) # [B, 4, 4]
        E[:, :3, :3] = R
        E[:, :3, 3] = translation

        return E

class FeatureUpsampler(nn.Module):
    def __init__(self, in_channels=768, out_channels=32):
        super(FeatureUpsampler, self).__init__()
        self.in_channels = in_channels

        self.cv_fuse = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, kernel_size=1),
            nn.GELU()
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256), nn.GELU()
        )
        self.skip1 = nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1))
        self.fuse1 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=1), nn.GELU()) 
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.GELU()
        )
        self.skip2 = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=1))
        self.fuse2 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.GELU())

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU()
        )
        self.fuse3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.GELU())

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels), nn.GELU()
        )
        self.fuse4 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=3, padding=1), nn.GELU())

        self.depth_head_28 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.depth_head_56 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.depth_head_112 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1), nn.Sigmoid())
        self.depth_head_224 = nn.Sequential(nn.Conv2d(out_channels, 1, kernel_size=3, padding=1), nn.Sigmoid())
    
    def forward(self, G, F_list, rgb_feats, cost_volume=None):
        B = G.shape[0]
        c_28, c_56, c_112, c_224 = rgb_feats 
        
        x = G.transpose(1, 2).view(B, self.in_channels, 14, 14) 

        if cost_volume is not None:
            x_cat = torch.cat([x, cost_volume], dim=1) 
            x = self.cv_fuse(x_cat)

        f_list = [f.transpose(1, 2).view(B, self.in_channels, 14, 14) for f in F_list] 
        
        # 14 -> 28
        x = self.up1(x) 
        s1 = F.interpolate(self.skip1(f_list[3]), size=(28, 28), mode='bilinear', align_corners=False)
        x = torch.cat([x, s1, c_28], dim=1) 
        out_28 = self.fuse1(x) 

        disp1 = self.depth_head_28(out_28)

        # 28 -> 56
        x = self.up2(out_28) 
        s2 = F.interpolate(self.skip2(f_list[2]), size=(56, 56), mode='bilinear', align_corners=False)
        x = torch.cat([x, s2, c_56], dim=1) 
        out_56 = self.fuse2(x) 
        
        disp2 = self.depth_head_56(out_56)

        x = self.up3(out_56) 
        x = torch.cat([x, c_112], dim=1) 
        out_112 = self.fuse3(x) 
        
        disp3 = self.depth_head_112(out_112)

        # 112 -> 224
        x = self.up4(out_112) 
        x = torch.cat([x, c_224], dim=1) 
        out_224 = self.fuse4(x) 
        
        disp4 = self.depth_head_224(out_224)

        return [out_28, out_56, out_112, out_224], [disp1, disp2, disp3, disp4]

class ImplicitDepthHead(nn.Module):
    def __init__(self, in_channels=384, img_size=224):
        super(ImplicitDepthHead, self).__init__()

        # ---------------------------------------------------------
        # [해결책 적용] 1x1 Conv -> 3x3 Conv로 변경하여 패치 경계선 허물기
        # padding_mode='replicate'를 주어 외곽선 아티팩트도 방지합니다.
        # ---------------------------------------------------------
        
        # 1. 글로벌 정보(G) 사포질
        self.proj_G = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU()
        )
        
        # 2. 4개의 F정보를 굳이 따로 압축할 필요 없이, 한 번에 합쳐서 3x3으로 섞어줍니다.
        # 입력: in_channels * 4 (384 * 4 = 1536) -> 출력: 64
        self.proj_F = nn.Sequential(
            # 1. 소화제 (1x1 Conv): 연산량을 대폭 줄이면서 1536채널의 중복 정보를 256으로 먼저 압축!
            nn.Conv2d(in_channels * 4, 256, kernel_size=1),
            nn.GELU(),
            
            # 2. 사포질 (3x3 Conv): 가벼워진 256채널 상태에서 주변 패치(타일)들과 부드럽게 융합
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            
            # 3. 최종 출력 (1x1 Conv): 사포질이 끝난 정보를 MLP가 먹기 좋게 64채널로 전달
            nn.Conv2d(256, 64, kernel_size=1),
            nn.GELU()
        )

        # 3. MLP 입력 차원 계산: G_low(64) + F_low(64) + uv_encoded(6) = 134
        self.mlp = nn.Sequential(
            nn.Linear(138, 128),  # <--- 134에서 146으로 변경!
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        nn.init.normal_(self.mlp[-1].weight, std=1e-5)
        nn.init.zeros_(self.mlp[-1].bias)

        y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, img_size), torch.linspace(-1.0, 1.0, img_size), indexing='ij')
        uv_grid = torch.stack([x, y], dim=-1).unsqueeze(0) 
        self.register_buffer('uv_grid', uv_grid)
    
    def forward(self, G, F_list):
        B = G.shape[0]

        G_spatial = G.transpose(1, 2).view(B, -1, 16, 16)
        f1_spatial = F_list[3].transpose(1, 2).view(B, -1, 16, 16)
        f2_spatial = F_list[2].transpose(1, 2).view(B, -1, 16, 16)
        f3_spatial = F_list[1].transpose(1, 2).view(B, -1, 16, 16)
        f4_spatial = F_list[0].transpose(1, 2).view(B, -1, 16, 16)

        # 4개의 특징을 채널 방향으로 하나로 통째로 묶음 [B, 1536, 16, 16]
        F_all = torch.cat([f1_spatial, f2_spatial, f3_spatial, f4_spatial], dim=1)

        # 3x3 Conv를 통과하며 16x16 모자이크 경계선이 부드럽게 융합됨!
        G_low = self.proj_G(G_spatial)
        F_low = self.proj_F(F_all)

        batch_uv_grid = self.uv_grid.repeat(B, 1, 1, 1)

        # 이제 부드러워진 단서를 바탕으로 샘플링
        G_sampled = F.grid_sample(G_low, batch_uv_grid, mode='bilinear', align_corners=False)
        F_sampled = F.grid_sample(F_low, batch_uv_grid, mode='bilinear', align_corners=False)

        G_flat = G_sampled.flatten(2).transpose(1, 2)
        F_flat = F_sampled.flatten(2).transpose(1, 2)
        
        uv_flat_batch = batch_uv_grid.flatten(1, 2)
        pi = 3.14159
        
        uv_encoded = torch.cat([
            uv_flat_batch, 
            torch.sin(1.0 * pi * uv_flat_batch), torch.cos(1.0 * pi * uv_flat_batch),
            torch.sin(2.0 * pi * uv_flat_batch), torch.cos(2.0 * pi * uv_flat_batch)
        ], dim=-1) # [B, 50176, 10]

        combined = torch.cat([G_flat, F_flat, uv_encoded], dim=-1) 
        out = self.mlp(combined) 

        disp_raw = torch.sigmoid(out.transpose(1, 2).view(B, 1, 224, 224))
        
        min_disp = 0.1 
        max_disp = 5.0 # (Z_max가 9.9m까지 튀는 것을 방지하기 위해 5.0으로 약간 조였습니다)
        
        scaled_disp = min_disp + (max_disp - min_disp) * disp_raw 
        Z_coord_safe = 1.0 / scaled_disp 

        return Z_coord_safe, scaled_disp
    
class RGBFeatureExtractor(nn.Module):
    def __init__(self):
        super(RGBFeatureExtractor, self).__init__()
        # 입력: [B, 3, 224, 224] 원본 사진
        
        self.conv_224 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
        )
        self.conv_112 = nn.Sequential(
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
        )
        self.conv_56  = nn.Sequential(
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
        )
        self.conv_28  = nn.Sequential(
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f_224 = self.conv_224(x)   # [B, 32, 224, 224]
        f_112 = self.conv_112(f_224) # [B, 64, 112, 112]
        f_56  = self.conv_56(f_112)  # [B, 128, 56, 56]
        f_28  = self.conv_28(f_56)   # [B, 256, 28, 28]
        
        return f_28, f_56, f_112, f_224

class DINOv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. DINOv2 vits14 모델 불러오기
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # 2. 강제로 뽑아낼 트랜스포머 블록 번호 지정 (vits14는 총 12개 블록: 0~11)
        # 보통 1/4 지점마다 하나씩 뽑아냅니다. (3번째, 6번째, 9번째, 12번째 층)
        self.out_blocks = [2, 5, 8, 11] 

    def forward(self, x):
        # x: [B, 3, 224, 224] 크기의 원본 이미지
        
        # 3. 마법의 함수! get_intermediate_layers를 사용하면 지정한 블록의 결과물을 리스트로 반환합니다.
        # reshape=True를 하면 1D 시퀀스를 알아서 2D 격자(16x16) 형태 텐서로 바꿔줍니다.
        features = self.backbone.get_intermediate_layers(
            x, 
            n=self.out_blocks, 
            reshape=False # (ImplicitDepthHead 코드에 맞게 1D 시퀀스로 받습니다)
        )
        
        # features는 지정한 4개 블록의 출력 텐서가 담긴 리스트입니다.
        # 각각의 형태는 [B, 256, 384] (Patch 개수 256개, 채널 384)
        
        F_list = features      # 이것이 바로 ImplicitDepthHead에 들어갈 4개의 원시 정보!
        G_curr = features[-1]  # 가장 마지막(12번째 층) 정보가 곧 글로벌 정보(G)가 됩니다.

        return G_curr, F_list

class CostVolumeBuilder(nn.Module): # Cost Volume 만들기
    def __init__(self, num_depth_bins=64):
        super(CostVolumeBuilder, self).__init__()
        self.D = num_depth_bins

    def forward(self, curr_F, prev_F, E_CURR_PREV, d_min, d_max, K):
        # curr_F : [B, 384, 14 * 14]
        # prev_F : [B, 384, 14 * 14]
        # E_CURR_PREV : [B, 4, 4]
        # d_min, d_max : [B, 1]
        # K : [B, 3, 3]

        curr_F = curr_F.transpose(1, 2)
        prev_F = prev_F.transpose(1, 2)

        B, C, _ = curr_F.shape
        device = curr_F.device

        curr_F = curr_F.view(B, C, 14, 14)
        prev_F = prev_F.view(B, C, 14, 14)

        B, C, H, W = curr_F.shape

        scale = float(H) / 224.0

        K_scaled = K.clone()
        K_scaled[:, 0, :] *= scale
        K_scaled[:, 1, :] *= scale

        inv_K = torch.inverse(K_scaled)

        # depths는 3D 공간에 세워둘 64개의 거리 표지판
        depths = self.get_depth(B, d_min, d_max, device) # [B, self.D, 1, 1]

        # pixel_coords는 픽셀 좌표계
        # [B, 3, H * W]
        pixel_curr_coords = self.get_pixelcoords(B, H, W, device) # [B, 3, 14 * 14]

        # torch.bmm은 배치는 건드리지 않고 뒤의 2D 행렬만 곱하는 배치 전용 행렬 곱셈 연산
        # 맨 뒤의 2D 행렬만 계산하기에 배치 차원 에러가 나지 않음
        # torch.bmm은 3차원만 받음. 주의할것

        # cam_coords는 픽셀 좌표계(2D)에서 카메라 좌표계(3D)로 올라간 것
        cam_curr_coords = self.get_cam_coords(inv_K, pixel_curr_coords) # [B, self.D, 3, 14 * 14]
        
        # 현재 
        cam_curr_homo = self.get_cam_curr_homo(B, cam_curr_coords, depths, H, W, device)

        cam_prev_homo = self.get_cam_prev_homo(B, E_CURR_PREV, cam_curr_homo)

        pixel_prev_homo = self.get_pixel_prev_homo(B, K, cam_prev_homo)

        pixel_prev_coords = self.get_pixel_prev_coords(B, H, W, pixel_prev_homo)
        
        # [-1, 1]로 정규화
        grid = (pixel_prev_coords * 2.0) - 1.0

        cost_volume = self.get_cost_volume(B, C, H, W, prev_F, grid, curr_F)

        cost_volume_prob = F.softmax(cost_volume, dim=1)

        # cost_volume_prob에는 curr의 특징과 prev 특징간 재투영 깊이의 확률이 들어있음
        # [B, self.D, H, W]
        return cost_volume_prob

    def transformation_from_parameters(self, pose):
        # B : [B, 6]
        B = pose.shape[0]
        device = pose.device
        
        # [B, 6] 벡터를 회전(rot_vec)과 이동(trans_vec)으로 분리
        rot_vec = pose[:, :3]      # [B, 3]
        trans_vec = pose[:, 3:]    # [B, 3]

        # -------------------------------------------------------------
        # STEP A: 로드리게스 공식으로 [B, 3] -> [B, 3, 3] 회전 행렬 R 만들기
        # -------------------------------------------------------------
        angle = torch.norm(rot_vec, dim=-1, keepdim=True) # [B, 1]
        axis = rot_vec / (angle + 1e-7) # [B, 3]
        
        # K와 차원을 맞추기 위해 [B, 1, 1] 형태로 커스텀 unsqueeze
        cos_a = torch.cos(angle).unsqueeze(-1) # [B, 1, 1]
        sin_a = torch.sin(angle).unsqueeze(-1) # [B, 1, 1]
        
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        
        zero = torch.zeros_like(x)
        K = torch.stack([
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1)
        ], dim=1) # [B, 3, 3]
        
        I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1) # [B, 3, 3]
        
        # 3x3 회전 행렬 R
        R = I + sin_a * K + (1 - cos_a) * torch.bmm(K, K) # [B, 3, 3]

        # -------------------------------------------------------------
        # STEP B: R과 T를 엮어서 최종 [B, 4, 4] 텐서로 조립하기
        # -------------------------------------------------------------
        # 단위 행렬 [B, 4, 4]
        T_mat = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1) # [B, 4, 4]
        
        # 좌측 상단 3x3에 R 복사
        T_mat[:, :3, :3] = R
        
        # 우측 상단 3x1에 이동 벡터(trans_vec) 복사
        T_mat[:, :3, 3] = trans_vec
        
        return T_mat # [B, 4, 4]

    def get_depth(self, B, d_min, d_max, device):
        depth_bins = torch.linspace(0, 1, self.D, device=device) # [self.D] 0~1을 self.D개로 쪼갬
        depths = d_min + depth_bins * (d_max - d_min) # [self.D] 최소 d_min 최대 d_max
        depths = depths.view(B, self.D, 1, 1) # [B, self.D, 1, 1]

        return depths

    def get_pixelcoords(self, B, H, W, device):
        # j에는 세로축(행, Y) 좌표판, i에는 가로축(열, X) 좌표판
        # 예를 들면
        # j에는 [0, 0, 0], [1, 1, 1], [2, 2, 2]
        # i에는 [0, 1, 2], [0, 1, 2], [0, 1, 2]
        # 둘 다 [H, W]
        j, i = torch.meshgrid(torch.linspace(0, H-1, H, device=device),
                              torch.linspace(0, W-1, W, device=device),
                              indexing='ij')
        
        # pixel_coords는 픽셀 동차 좌표계 형태
        # i가 쌓이고 j가 쌓이고 맨 뒤에는 1만 가득차서 쌓이고
        # x가 쌓이고 y가 쌓이고 맨 뒤에는 1만 가득차서 쌓이고
        # [0, H * W]를 하면 x좌표가 보이고, [1, H * W]를 하면 y좌표가 보이고
        pixel_coords = torch.stack([i, j, torch.ones_like(i)], dim=0).view(3, -1) # [3, H * W]
        pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1) # [B, 3, H * W]

        return pixel_coords

    def get_cam_coords(self, inv_K, pixel_coords):
        # cam_coords는 픽셀 좌표계(2D)에서 카메라 좌표계(3D)로 올라간 것
        # 아직 깊이가 적용되지 않음
        cam_coords = torch.bmm(inv_K, pixel_coords) # [B, 3, H*W]
        cam_coords = cam_coords.unsqueeze(1).repeat(1, self.D, 1, 1) # [B, self.D, 3, H*W]

        return cam_coords

    def get_cam_curr_homo(self, B, cam_coords, depths, H, W, device):
        # [B, self.D, 3, H*W] * [B, self.D, 1, 1]
        # 그냥 Broadcasting된다음 원소별로 1대1로 곱해짐
        # xyz_curr은 깊이가 적용된 카메라 좌표계(3D)
        xyz_curr = cam_coords * depths # [B, self.D, 3, H*W]

        # cat을 통해 동차 좌표계 형태로 만들어줌
        ones = torch.ones(B, self.D, 1, H*W, device=device) # [B, self.D, 1, H*W]
        xyz_curr_homo = torch.cat([xyz_curr, ones], dim=2) # [B, self.D, 4, H*W]

        # torch.bmm은 3차원만 받기에 3차원으로 변경해줌
        xyz_curr_homo = xyz_curr_homo.view(B * self.D, 4, H * W) # [B * self.D, 4, H*W]

        return xyz_curr_homo

    def get_cam_prev_homo(self, B, T_curr_to_prev, cam_curr_homo):
        # T_curr_to_prev는 curr에서 prev로의 외부 파라미터
        # 행렬 연산을 통해 xyz_curr_homo를 xyz_prev_homo로 변경
        # xyz_prev_homo는 prev 카메라 좌표계 (3D)
        T_repeat = T_curr_to_prev.unsqueeze(1).repeat(1, self.D, 1, 1).view(B * self.D, 4, 4) # [B * self.D, 4, 4]
        cam_prev_homo = torch.bmm(T_repeat, cam_curr_homo) # [B * self.D, 4, H*W]

        return cam_prev_homo

    def get_pixel_prev_homo(self, B, K, cam_prev_homo):
        # prev 카메라 좌표계(3D)에서 prev 이미지 좌표계(2D)로 내리기
        K_repeat = K.unsqueeze(1).repeat(1, self.D, 1, 1).view(B * self.D, 3, 3) # [B * self.D, 3, 3]
        pixel_prev_homo = torch.bmm(K_repeat, cam_prev_homo[:, :3, :]) # [B * self.D, 3, H*W]

        return pixel_prev_homo

    def get_pixel_prev_coords(self, B, H, W, pixel_prev_homo):
        # 0, 1만 가져옴 -> x와 y만 가져옴, 이걸 2로 나눔 -> 깊이로 나눔
        # pixel_prev_coords x와 y
        pixel_prev_coords = pixel_prev_homo[:, :2, :] / (pixel_prev_homo[:, 2:3, :] + 1e-7) # [B * self.D, 2, H*W]
        pixel_prev_coords = pixel_prev_coords.view(B * self.D, 2, H, W).permute(0, 2, 3, 1) # [B * self.D, H, W, 2]

        # [0, 1]로 정규화
        pixel_prev_coords[..., 0] = pixel_prev_coords[..., 0] / (W-1)
        pixel_prev_coords[..., 1] = pixel_prev_coords[..., 1] / (H-1)

        return pixel_prev_coords

    def get_cost_volume(self, B, C, H, W, prev_F, grid, curr_F):
        # prev의 특징을 늘려서 재투영
        prev_f_repeat = prev_F.unsqueeze(1).repeat(1, self.D, 1, 1, 1).view(B * self.D, C, H, W) # [B * self.D, C, H, W]
        warped_prev_features = F.grid_sample(prev_f_repeat, grid, padding_mode="border", align_corners=True) # [B * self.D, C, H, W]
        warped_prev_features = warped_prev_features.view(B, self.D, C, H, W) # [B, self.D, C, H, W]

        # prev의 특징과 curr의 특징의 차이
        curr_features_extend = curr_F.unsqueeze(1).repeat(1, self.D, 1, 1, 1) # [B, self.D, C, H, W]
        cost_volume = torch.mean(curr_features_extend * warped_prev_features, dim=2) # [B, self.D, H, W]

        return cost_volume

class MonoMirror_v4(nn.Module):
    def __init__(self):
        super(MonoMirror_v4, self).__init__()

        self.encoder = DINOv2Encoder()
        
        self.patch_embedded_dim = 384

        self.positionalEncoding2D = PositionalEncoding2D(d_model=self.patch_embedded_dim, h_patches=14, w_patches=14)

        self.projection_head = ProjectionHead(d_model=self.patch_embedded_dim)

        self.decoder_layers = 8
        self.decoders = nn.ModuleList([
            Decoder(d_model=self.patch_embedded_dim, h=12) for _ in range(self.decoder_layers)
        ])

        self.rgb_extractor = RGBFeatureExtractor()

        self.cost_volume_builder = CostVolumeBuilder(num_depth_bins=64)

        self.upsampler = FeatureUpsampler(in_channels=self.patch_embedded_dim)

        self.d_min = 0.4
        self.d_max = 15.0

        self.depth_Head_224 = DepthHead(in_channel=32)

        self.implictDepthHead = ImplicitDepthHead(in_channels=384)

        H, W = 224, 224
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        u_flat = (x.float() + 0.5).view(-1, 1)
        v_flat = (y.float() + 0.5).view(-1, 1)

        self.register_buffer('u', u_flat)
        self.register_buffer('v', v_flat)

    def forward(self, prev_img, curr_img, next_img, sfs=False):
        # image : [B, 3, H, W] [B, 3, 224, 224]
        B, C, H, W = prev_img.shape

        # 224를 196으로 바꿈, 196은 14x14임, DINOv2에 딱 맞음.
        prev_img_196 = F.interpolate(prev_img, size=(196, 196), mode='bilinear', align_corners=False)
        curr_img_196 = F.interpolate(curr_img, size=(196, 196), mode='bilinear', align_corners=False)
        next_img_196 = F.interpolate(next_img, size=(196, 196), mode='bilinear', align_corners=False)

        with torch.no_grad():
            # [B, 384, 14x14]가 나옴
            PREV_G, PREV_F = self.encoder(prev_img_196)
            CURR_G, CURR_F = self.encoder(curr_img_196)
            NEXT_G, NEXT_F = self.encoder(next_img_196)

        PREV_G_RAW = PREV_G.clone()
        CURR_G_RAW = CURR_G.clone()
        NEXT_G_RAW = NEXT_G.clone()

        K, E_CURR_PREV, E_CURR_NEXT, E_CURR_PREV_INV, E_CURR_NEXT_INV = self.projection_head(PREV_F[-1], CURR_F[-1], NEXT_F[-1])

        d_min = torch.ones(B, 1, device=PREV_G.device) * self.d_min
        d_max = torch.ones(B, 1, device=PREV_G.device) * self.d_max

        CV_PREV = self.cost_volume_builder(CURR_G, PREV_G, E_CURR_PREV, d_min, d_max, K)
        CV_NEXT = self.cost_volume_builder(CURR_G, NEXT_G, E_CURR_NEXT, d_min, d_max, K)
        
        PREV_G = self.positionalEncoding2D(PREV_G)
        CURR_G = self.positionalEncoding2D(CURR_G)
        NEXT_G = self.positionalEncoding2D(NEXT_G)

        for decoder in self.decoders:
            tmp_curr = CURR_G
            
            curr_from_prev = decoder(tmp_curr, PREV_G, CV_PREV)
            curr_from_next = decoder(tmp_curr, NEXT_G, CV_NEXT)

            CURR_G = (curr_from_prev + curr_from_next) / 2.0
            PREV_G = decoder(PREV_G, tmp_curr, None)
            NEXT_G = decoder(NEXT_G, tmp_curr, None)

        # 4개 나옴
        # [B, 32, 224, 224] [B, 64, 112, 112] [B, 128, 56, 56] [B, 256, 28, 28]
        CURR_RGB_F = self.rgb_extractor(curr_img)

        CV_COMBINED = (CV_PREV + CV_NEXT) / 2.0

        CURR_UP_F, DISP = self.upsampler(CURR_G, CURR_F, CURR_RGB_F, CV_COMBINED)

        # DISP[0] = F.interpolate(DISP[0], size=(H, W), mode='nearest')
        # DISP[1] = F.interpolate(DISP[1], size=(H, W), mode='nearest')
        # DISP[2] = F.interpolate(DISP[2], size=(H, W), mode='nearest')

        # PREV_MATRIX, PREV_MATRIX_INV = self.get_MATRIX(B, K, E_CURR_PREV, E_CURR_PREV_INV)
        # NEXT_MATRIX, NEXT_MATRIX_INV = self.get_MATRIX(B, K, E_CURR_NEXT, E_CURR_NEXT_INV)

        # CURR_Z_28 = 1.0 / (self.d_min + (self.d_max - self.d_min) * DISP[0])
        # CURR_Z_56 = 1.0 / (self.d_min + (self.d_max - self.d_min) * DISP[1])
        # CURR_Z_112 = 1.0 / (self.d_min + (self.d_max - self.d_min) * DISP[2])
        # CURR_Z_224 = 1.0 / (self.d_min + (self.d_max - self.d_min) * DISP[3])

        def scale_K(K_mat, scale_factor):
            K_scaled = K_mat.clone()
            K_scaled[:, 0, :] *= scale_factor
            K_scaled[:, 1, :] *= scale_factor
            return K_scaled

        K_224 = K
        K_112 = scale_K(K, 112.0 / 224.0)
        K_56  = scale_K(K, 56.0 / 224.0)
        K_28  = scale_K(K, 28.0 / 224.0)
        K_14 = scale_K(K, 14.0 / 224.0)

        PREV_MAT_14, _ = self.get_MATRIX(B, K_14, E_CURR_PREV, E_CURR_PREV_INV)
        NEXT_MAT_14, _ = self.get_MATRIX(B, K_14, E_CURR_NEXT, E_CURR_NEXT_INV)

        PREV_MAT_28, _  = self.get_MATRIX(B, K_28, E_CURR_PREV, E_CURR_PREV_INV)
        NEXT_MAT_28, _  = self.get_MATRIX(B, K_28, E_CURR_NEXT, E_CURR_NEXT_INV)
        
        PREV_MAT_56, _  = self.get_MATRIX(B, K_56, E_CURR_PREV, E_CURR_PREV_INV)
        NEXT_MAT_56, _  = self.get_MATRIX(B, K_56, E_CURR_NEXT, E_CURR_NEXT_INV)
        
        PREV_MAT_112, _ = self.get_MATRIX(B, K_112, E_CURR_PREV, E_CURR_PREV_INV)
        NEXT_MAT_112, _ = self.get_MATRIX(B, K_112, E_CURR_NEXT, E_CURR_NEXT_INV)
        
        PREV_MAT_224, PREV_MAT_224_INV = self.get_MATRIX(B, K_224, E_CURR_PREV, E_CURR_PREV_INV)
        NEXT_MAT_224, NEXT_MAT_224_INV = self.get_MATRIX(B, K_224, E_CURR_NEXT, E_CURR_NEXT_INV)

        CURR_Z_28  = 1.0 / (self.d_min + (self.d_max - self.d_min) * DISP[0])
        CURR_Z_56  = 1.0 / (self.d_min + (self.d_max - self.d_min) * DISP[1])
        CURR_Z_112 = 1.0 / (self.d_min + (self.d_max - self.d_min) * DISP[2])
        CURR_Z_224 = 1.0 / (self.d_min + (self.d_max - self.d_min) * DISP[3])

        CURR_XYZ_28  = self.get_XYZ(B, CURR_Z_28, K_28, 28, 28)
        CURR_XYZ_56  = self.get_XYZ(B, CURR_Z_56, K_56, 56, 56)
        CURR_XYZ_112 = self.get_XYZ(B, CURR_Z_112, K_112, 112, 112)
        CURR_XYZ_224 = self.get_XYZ(B, CURR_Z_224, K_224, 224, 224)

        # CURR_XYZ_28 = self.get_XYZ(B, CURR_Z_28, K)
        # CURR_XYZ_56 = self.get_XYZ(B, CURR_Z_56, K)
        # CURR_XYZ_112 = self.get_XYZ(B, CURR_Z_112, K)
        # CURR_XYZ_224 = self.get_XYZ(B, CURR_Z_224, K)

        CURR_Z = CURR_Z_224

        if sfs:
            print(f"--- [Fixed Sample Monitoring] ---")
            print(f"True fx: {K[0, 0, 0].item():.2f}, True fy: {K[0, 1, 1].item():.2f}")
            print(f"K : \n{K}")
            print(f"E_CURR_PREV : \n{E_CURR_PREV}")
            print(f"E_CURR_NEXT : \n{E_CURR_NEXT}")
            print(f"Z min: {CURR_Z.min().item():.4f}, Z max: {CURR_Z.max().item():.4f}, 갭: {(CURR_Z.max() - CURR_Z.min()).item():.4f}")
            print(f"---------------------------------")

        # return {
        #     'DISP' : DISP,
        #     'XYZ' : [CURR_XYZ_28, CURR_XYZ_56, CURR_XYZ_112, CURR_XYZ_224], 
        #     'PREV_MATRIX' : PREV_MATRIX,
        #     'NEXT_MATRIX' : NEXT_MATRIX,
        #     'PREV_G_RAW' : PREV_G_RAW,
        #     'CURR_G_RAW' : CURR_G_RAW,
        #     'NEXT_G_RAW' : NEXT_G_RAW,
        # }
        return {
            'DISP' : DISP,
            'XYZ' : [CURR_XYZ_28, CURR_XYZ_56, CURR_XYZ_112, CURR_XYZ_224], 
            'PREV_MATRIX' : [PREV_MAT_28, PREV_MAT_56, PREV_MAT_112, PREV_MAT_224],
            'NEXT_MATRIX' : [NEXT_MAT_28, NEXT_MAT_56, NEXT_MAT_112, NEXT_MAT_224],
            'PREV_G_RAW' : PREV_G_RAW,
            'CURR_G_RAW' : CURR_G_RAW,
            'NEXT_G_RAW' : NEXT_G_RAW,
            'PREV_MATRIX_14' : PREV_MAT_14, # 이걸 내보내서
            'NEXT_MATRIX_14' : NEXT_MAT_14
        }
    
    def get_XYZ(self, B, Z, K, H, W):
        Z = Z.view(B, -1, 1)

        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)

        y, x = torch.meshgrid(torch.arange(H, device=Z.device), torch.arange(W, device=Z.device), indexing='ij')
        u_flat = (x.float() + 0.5).view(-1, 1)
        v_flat = (y.float() + 0.5).view(-1, 1)

        X = (u_flat - cx) * Z / fx
        Y = (v_flat - cy) * Z / fy
        XYZ = torch.cat([X, Y, Z], dim=-1)

        # X = (self.u - cx) * Z / fx
        # Y = (self.v - cy) * Z / fy
        # XYZ = torch.cat([X, Y, Z], dim=-1) # 최종 3D 좌표 [B, 50176, 3]

        return XYZ

    def get_MATRIX(self, B, K, E, E_INV):
        K44 = torch.eye(4, device=K.device).unsqueeze(0).repeat(B, 1, 1)
        K44[:, :3, :3] = K

        MATRIX = torch.bmm(K44, E)[:, :3, :]
        MATRIX_INV = torch.bmm(K44, E_INV)[:, :3, :]

        return MATRIX, MATRIX_INV