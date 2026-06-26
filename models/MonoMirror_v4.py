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
        nn.init.normal_(self.extrinsic_conv[-1].weight, mean=0.0, std=1e-5)
        nn.init.normal_(self.extrinsic_conv[-1].bias, mean=0.0, std=1e-5)

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

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256), nn.GELU()
        )
        self.skip1 = nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1))
        self.fuse1 = nn.Sequential(nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), nn.GELU()) 
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.GELU()
        )
        self.skip2 = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=1))
        self.fuse2 = nn.Sequential(nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), nn.GELU())

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU()
        )
        self.skip3 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1))
        self.fuse3 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.GELU())

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels), nn.GELU()
        )
        self.skip4 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=1))
        self.fuse4 = nn.Sequential(nn.Conv2d(32 + 32, out_channels, kernel_size=3, padding=1), nn.GELU())
    
    def forward(self, DINO_G, DINO_F):
        B = DINO_G.shape[0]

        x = DINO_G.transpose(1, 2).view(B, self.in_channels, 14, 14) # [B, 384, 14, 14]

        DINO_F_0, DINO_F_1, DINO_F_2, DINO_F_3 = [f.transpose(1, 2).view(B, self.in_channels, 14, 14) for f in DINO_F] # [B, 384, 14, 14]
        
        # 14 -> 28
        x = self.up1(x) # [B, 384, 14, 14] -> [B, 256, 28, 28]
        s1 = F.interpolate(self.skip1(DINO_F_3), size=(28, 28), mode='bilinear', align_corners=False) # [B, 384, 14, 14] -> [B, 256, 28, 28]
        x = torch.cat([x, s1], dim=1) # [B, 256 + 256 + 32, 28, 28]
        out_28 = self.fuse1(x) # [B, 256 + 256 + 32, 28, 28] -> [B, 256, 28, 28]

        # 28 -> 56
        x = self.up2(out_28) # [B, 256, 28, 28] -> [B, 128, 56, 56]
        s2 = F.interpolate(self.skip2(DINO_F_2), size=(56, 56), mode='bilinear', align_corners=False) # [B, 384, 14, 14] -> [B, 128, 56, 56]
        x = torch.cat([x, s2], dim=1) # [B, 128 + 128 + 16, 56, 56]
        out_56 = self.fuse2(x) 

        x = self.up3(out_56) 
        s3 = F.interpolate(self.skip3(DINO_F_1), size=(112, 112), mode='bilinear', align_corners=False) # [B, 384, 14, 14] -> [B, 64, 112, 112]
        x = torch.cat([x, s3], dim=1) 
        out_112 = self.fuse3(x) 

        # 112 -> 224
        x = self.up4(out_112) 
        s4 = F.interpolate(self.skip4(DINO_F_0), size=(224, 224), mode='bilinear', align_corners=False) # [B, 384, 14, 14] -> [B, 32, 224, 224]
        x = torch.cat([x, s4], dim=1) 
        out_224 = self.fuse4(x) 

        return [out_28, out_56, out_112, out_224]

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

        self.upsampler = FeatureUpsampler(in_channels=self.patch_embedded_dim)

        self.d_min = 0.4
        self.d_max = 15.0

        self.depth_Head_224 = DepthHead(in_channel=32)

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

        K, E_CURR_PREV, E_CURR_NEXT, E_CURR_PREV_INV, E_CURR_NEXT_INV = self.projection_head(PREV_F[-1], CURR_F[-1], NEXT_F[-1])
        
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

        CURR_UP_F = self.upsampler(CURR_G, CURR_F)

        CURR_Z, DISP = self.depth_Head_224(CURR_UP_F[-1])

        PREV_MATRIX, _ = self.get_MATRIX(B, K, E_CURR_PREV, E_CURR_PREV_INV)
        NEXT_MATRIX, _ = self.get_MATRIX(B, K, E_CURR_NEXT, E_CURR_NEXT_INV)

        CURR_XYZ = self.get_XYZ(B, CURR_Z, K, H, W)

        if sfs:
            print(f"--- [Fixed Sample Monitoring] ---")
            print(f"True fx: {K[0, 0, 0].item():.2f}, True fy: {K[0, 1, 1].item():.2f}")
            print(f"K : \n{K}")
            print(f"E_CURR_PREV : \n{E_CURR_PREV}")
            print(f"E_CURR_NEXT : \n{E_CURR_NEXT}")
            print(f"Z min: {CURR_Z.min().item():.4f}, Z max: {CURR_Z.max().item():.4f}, 갭: {(CURR_Z.max() - CURR_Z.min()).item():.4f}")
            print(f"---------------------------------")

        return {
            'DISP' : DISP,
            'XYZ' : CURR_XYZ, 
            'PREV_MATRIX' : PREV_MATRIX,
            'NEXT_MATRIX' : NEXT_MATRIX,
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