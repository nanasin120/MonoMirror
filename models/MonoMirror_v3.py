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

        self.proj_to_bias = nn.Linear(16, d_model)

        self.p2_norm = nn.LayerNorm(self.d_model)
        
        self.self_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_0 = nn.LayerNorm(self.d_model)

        self.cross_attention = MultiHead(self.d_model, self.h)
        self.layer_norm_1 = nn.LayerNorm(self.d_model)

        self.FFN = FeedForwardNetwork(self.d_model, self.d_model * 4)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)

    def forward(self, p1, p2, matrix_info=None):
        B = matrix_info.shape[0] 
        matrix_info = matrix_info.reshape(B, -1)

        matrix_bias = self.proj_to_bias(matrix_info).unsqueeze(1)

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
    def __init__(self, in_channel=32):
        super(DepthHead, self).__init__()

        self.MLP = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=1),
        )

        nn.init.normal_(self.MLP[-1].weight, std=1e-5)
        nn.init.zeros_(self.MLP[-1].bias)

    def forward(self, all_G):
        out = self.MLP(all_G) # [B, 1, 224, 224]

        disp_raw = torch.sigmoid(out) # 0 ~ 1
        min_disp = 0.1 # 100
        max_disp = 10.0 # 0.1
        
        # disp_raw가 0이면 0.01, 1이면 10.0
        # 0.1 ~ 10.0
        scaled_disp = min_disp + (max_disp - min_disp) * disp_raw 
        
        # disp_raw가 0이면 1/0.01 = 100, 1이면 1/10 = 0.1
        # 0.1 ~ 100.0
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
        F1_spatial = F1.permute(0, 2, 1).view(B, -1, 16, 16)
        F2_spatial = F2.permute(0, 2, 1).view(B, -1, 16, 16)
        Diff_spatial = F1_spatial - F2_spatial

        combined = torch.cat([F1_spatial, F2_spatial, Diff_spatial], dim=1) # [B, d_model * 2, 16, 16]
        extrinsic_raw = self.extrinsic_conv(combined)

        axis_angle = torch.tanh(extrinsic_raw[:, :3]) * 3.14159 / 3.0 # -3.14159 ~ 3.14159
        R = axis_angle_to_matrix(axis_angle) # [B, 3, 3]

        # translation_dir = F.normalize(extrinsic_raw[:, 3:6], p=2, dim=-1) # 이동 방향 [B, 3]
        # anchors = torch.tensor([0.05, 0.1, 0.2, 0.4, 0.8], dtype=torch.float32, device=F1.device) # 간격 [B, 5]
        
        # scale_logits = extrinsic_raw[:, 6:11] # [B, 5]
        # scale_probs = F.softmax(scale_logits, dim=-1) # [B, 5]
        
        # magnitude = torch.sum(scale_probs * anchors, dim=-1, keepdim=True) # [B, 1]
        
        # translation = translation_dir * magnitude # [B, 3]
        
        translation = torch.tanh(extrinsic_raw[:, 3:6]) * 0.5
        
        E = torch.eye(4, device=F1.device).unsqueeze(0).repeat(B, 1, 1) # [B, 4, 4]
        E[:, :3, :3] = R
        E[:, :3, 3] = translation

        return E
    
class FeatureUpsampler(nn.Module):
    def __init__(self, in_channels=768, out_channels=32): # 768 -> 256 -> 128 -> 64 -> 32
        super(FeatureUpsampler, self).__init__()

        self.in_channels = in_channels

        self.up1 = nn.Sequential(
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.GELU()
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),
        )
        self.fuse1 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=1), nn.GELU())
        
        self.up2 = nn.Sequential(
            nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU()
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),
        )
        self.fuse2 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1), nn.GELU())

        self.up3 = nn.Sequential(
            nn.Upsample(size=(112, 112), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU()
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Upsample(size=(112, 112), mode='bilinear', align_corners=False),
        )
        self.fuse3 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.GELU())

        self.up4 = nn.Sequential(
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_channels),
            nn.GELU()
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        )
        self.fuse4 = nn.Sequential(nn.Conv2d(96, out_channels, kernel_size=3, padding=1), nn.GELU())
    
    def forward(self, G, F, rgb_feats):
        # G는 디코더의 추출물 [B, 196, 768]
        # F는 인코더의 추출물 [4, B, 196, 768] 2, 5, 8, 11
        B = G.shape[0]
        c_28, c_56, c_112, c_224 = rgb_feats
        
        x = G.transpose(1, 2).view(B, self.in_channels, 16, 16)
        f1 = F[3].transpose(1, 2).view(B, self.in_channels, 16, 16)
        f2 = F[2].transpose(1, 2).view(B, self.in_channels, 16, 16)
        f3 = F[1].transpose(1, 2).view(B, self.in_channels, 16, 16)
        f4 = F[0].transpose(1, 2).view(B, self.in_channels, 16, 16)
        
        x = self.up1(x) # [B, 256, 28, 28]
        s1 = self.skip1(f1) # [B, 256, 28, 28]
        x = torch.cat([x, s1, c_28], dim=1) # [B, 512, 28, 28]
        out_28 = self.fuse1(x) # [B, 256, 28, 28]

        x = self.up2(out_28) # [B, 128, 56, 56]
        s2 = self.skip2(f2) # [B, 128, 56, 56]
        x = torch.cat([x, s2, c_56], dim=1) # [B, 256, 56, 56]
        out_56 = self.fuse2(x) # [B, 128, 56, 56]

        x = self.up3(out_56) # [B, 64, 112, 112]
        s3 = self.skip3(f3) # [B, 64, 112, 112]
        x = torch.cat([x, s3, c_112], dim=1) # [B, 128, 112, 112]
        out_112 = self.fuse3(x) # [B, 64, 112, 112]

        x = self.up4(out_112) # [B, 32, 224, 224]
        s4 = self.skip4(f4) # [B, 32, 224, 224]
        x = torch.cat([x, s4, c_224], dim=1) # [B, 64, 224, 224]
        out_224 = self.fuse4(x) # [B, 32, 224, 224]

        return [out_28, out_56, out_112, out_224]

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

class MonoMirror_v3(nn.Module):
    def __init__(self):
        super(MonoMirror_v3, self).__init__()

        self.encoder = DINOv2Encoder()
        
        self.patch_embedded_dim = 384

        self.positionalEncoding2D = PositionalEncoding2D(d_model=self.patch_embedded_dim, h_patches=16, w_patches=16)

        self.projection_head = ProjectionHead(d_model=self.patch_embedded_dim)

        self.decoder_layers = 8
        self.decoders = nn.ModuleList([
            Decoder(d_model=self.patch_embedded_dim, h=12) for _ in range(self.decoder_layers)
        ])

        self.rgb_extractor = RGBFeatureExtractor()

        self.upsampler = FeatureUpsampler(in_channels=self.patch_embedded_dim)

        self.depth_Head_28 = DepthHead(in_channel=256)
        self.depth_Head_56 = DepthHead(in_channel=128)
        self.depth_Head_112 = DepthHead(in_channel=64)
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
        B = prev_img.shape[0]

        with torch.no_grad():
            prev_G, prev_F = self.encoder(prev_img)
            curr_G, curr_F = self.encoder(curr_img)
            next_G, next_F = self.encoder(next_img)

        prev_rgb_feats = self.rgb_extractor(prev_img)
        curr_rgb_feats = self.rgb_extractor(curr_img)
        next_rgb_feats = self.rgb_extractor(next_img)

        K, E_CURR_PREV, E_CURR_NEXT, E_CURR_PREV_INV, E_CURR_NEXT_INV = self.projection_head(prev_F[-1], curr_F[-1], next_F[-1])
        
        prev_G = self.positionalEncoding2D(prev_G)
        curr_G = self.positionalEncoding2D(curr_G)
        next_G = self.positionalEncoding2D(next_G)

        for decoder in self.decoders:
            tmp_curr = curr_G
            
            curr_from_prev = decoder(tmp_curr, prev_G, E_CURR_PREV)
            curr_from_next = decoder(tmp_curr, next_G, E_CURR_NEXT)

            curr_G = (curr_from_prev + curr_from_next) / 2.0
            prev_G = decoder(prev_G, tmp_curr, E_CURR_PREV_INV)
            next_G = decoder(next_G, tmp_curr, E_CURR_NEXT_INV)

        prev_up_feats = self.upsampler(prev_G, prev_F, prev_rgb_feats) 
        curr_up_feats = self.upsampler(curr_G, curr_F, curr_rgb_feats)
        next_up_feats = self.upsampler(next_G, next_F, next_rgb_feats)

        PREV_MATRIX, PREV_MATRIX_INV = self.get_MATRIX(B, K, E_CURR_PREV, E_CURR_PREV_INV)
        NEXT_MATRIX, NEXT_MATRIX_INV = self.get_MATRIX(B, K, E_CURR_NEXT, E_CURR_NEXT_INV)

        PREV_F_DENSE = prev_up_feats[-1]
        CURR_F_DENSE = curr_up_feats[-1]
        NEXT_F_DENSE = next_up_feats[-1]

        PREV_Z, PREV_DISP = self.depth_Head_224(PREV_F_DENSE)
        CURR_Z, CURR_DISP = self.depth_Head_224(CURR_F_DENSE)
        NEXT_Z, NEXT_DISP = self.depth_Head_224(NEXT_F_DENSE)

        prev_F_frozen = F.interpolate(prev_F[-1].transpose(1, 2).view(B, -1, 16, 16), size=(224, 224), mode='bilinear', align_corners=False)
        curr_F_frozen = F.interpolate(curr_F[-1].transpose(1, 2).view(B, -1, 16, 16), size=(224, 224), mode='bilinear', align_corners=False)
        next_F_frozen = F.interpolate(next_F[-1].transpose(1, 2).view(B, -1, 16, 16), size=(224, 224), mode='bilinear', align_corners=False)

        PREV_XYZ = self.get_XYZ(B, PREV_Z, K)
        CURR_XYZ = self.get_XYZ(B, CURR_Z, K)
        NEXT_XYZ = self.get_XYZ(B, NEXT_Z, K)

        if sfs:
            print(f"--- [Fixed Sample Monitoring] ---")
            print(f"True fx: {K[0, 0, 0].item():.2f}, True fy: {K[0, 1, 1].item():.2f}")
            print(f"K : \n{K}")
            print(f"E_CURR_PREV : \n{E_CURR_PREV}")
            print(f"E_CURR_NEXT : \n{E_CURR_NEXT}")
            print(f"Z min: {CURR_Z.min().item():.4f}, Z max: {CURR_Z.max().item():.4f}, 갭: {(CURR_Z.max() - CURR_Z.min()).item():.4f}")
            print(f"---------------------------------")

        # 4. 출력 정리: 더 이상 복잡한 multi 배열은 없습니다!
        return {
            'Z' : [PREV_Z, CURR_Z, NEXT_Z],            
            'DISP' : [PREV_DISP, CURR_DISP, NEXT_DISP],
            'XYZ' : [PREV_XYZ, CURR_XYZ, NEXT_XYZ],                
            'F_FROZEN': [prev_F_frozen, curr_F_frozen, next_F_frozen],  # 🚨 여기 이름 변경!
            'MATRIX' : [PREV_MATRIX, NEXT_MATRIX],
            'MATRIX_INV' : [PREV_MATRIX_INV, NEXT_MATRIX_INV],
            'E' : [E_CURR_PREV, E_CURR_NEXT],
            'E_INV' : [E_CURR_PREV_INV, E_CURR_NEXT_INV],
            'K' : K
        }
    
    def get_XYZ(self, B, Z, K):
        Z = Z.view(B, -1, 1)

        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)

        X = (self.u - cx) * Z / fx
        Y = (self.v - cy) * Z / fy
        XYZ = torch.cat([X, Y, Z], dim=-1) # 최종 3D 좌표 [B, 50176, 3]

        return XYZ

    def get_MATRIX(self, B, K, E, E_INV):
        K44 = torch.eye(4, device=K.device).unsqueeze(0).repeat(B, 1, 1)
        K44[:, :3, :3] = K

        MATRIX = torch.bmm(K44, E)[:, :3, :]
        MATRIX_INV = torch.bmm(K44, E_INV)[:, :3, :]

        return MATRIX, MATRIX_INV