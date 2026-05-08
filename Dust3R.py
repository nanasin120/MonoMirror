import torch
import torch.nn as nn
import torch.nn.functional as F
from CroCo import CroCo
from blocks import PositionalEncoding2D, MultiHead, FeedForwardNetwork
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
    
class Head(nn.Module):
    def __init__(self, in_channel=32):
        super(Head, self).__init__()

        self.MLP = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=1),
        )

        nn.init.normal_(self.MLP[-1].weight, std=1e-4)
        nn.init.zeros_(self.MLP[-1].bias)

    def forward(self, all_G):
        out = self.MLP(all_G) # [B, 1, 224, 224]

        disp_raw = torch.sigmoid(out) # 0 ~ 1
        min_disp = 0.01 # 100
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

        self.intrinsic_mlp = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        )
        nn.init.normal_(self.intrinsic_mlp[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.intrinsic_mlp[-1].bias)
        with torch.no_grad(): self.intrinsic_mlp[-1].bias[:] = 150.0

        self.extrinsic_mlp = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Linear(256, 6)
        )
        nn.init.normal_(self.extrinsic_mlp[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.extrinsic_mlp[-1].bias)
        with torch.no_grad(): self.extrinsic_mlp[-1].bias[:] = 0.1

    def forward(self, F1, F2):
        B = F1.shape[0]

        combined = torch.cat([F1.mean(dim=1), F2.mean(dim=1)], dim=-1)
        intrinsic_raw = self.intrinsic_mlp(combined)
        extrinsic_raw = self.extrinsic_mlp(combined)

        axis_angle = torch.tanh(extrinsic_raw[:, :3]) * 3.14159 # -3.14159 ~ 3.14159
        translation = torch.tanh(extrinsic_raw[:, 3:]) * 1.0 # -1.0 ~ 1.0
        
        f = F.softplus(intrinsic_raw) + 50.0
        fx, fy = f[:, 0], f[:, 1]
        
        K = torch.zeros((B, 3, 3), device=F1.device)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = self.img_size / 2.0  # cx (정중앙 고정)
        K[:, 1, 2] = self.img_size / 2.0  # cy (정중앙 고정)
        K[:, 2, 2] = 1.0

        R = axis_angle_to_matrix(axis_angle) # [B, 3, 3]
        
        E = torch.eye(4, device=F1.device).unsqueeze(0).repeat(B, 1, 1) # [B, 4, 4]
        E[:, :3, :3] = R
        E[:, :3, 3] = translation

        return K, E
    
class FeatureUpsampler(nn.Module):
    def __init__(self, in_channels=768, out_channels=32): # 768 -> 256 -> 128 -> 64 -> 32
        super(FeatureUpsampler, self).__init__()
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(768, 64, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(768, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, G, F):
        # G는 디코더의 추출물 [B, 196, 768]
        # F는 인코더의 추출물 [4, B, 196, 768] 2, 5, 8, 11
        B = G.shape[0]
        
        x = G.transpose(1, 2).view(B, 768, 14, 14)
        f1 = F[3].transpose(1, 2).view(B, 768, 14, 14)
        f2 = F[2].transpose(1, 2).view(B, 768, 14, 14)
        f3 = F[1].transpose(1, 2).view(B, 768, 14, 14)
        f4 = F[0].transpose(1, 2).view(B, 768, 14, 14)
        
        x = self.up1(x) # [B, 256, 28, 28]
        s1 = self.skip1(f1) # [B, 256, 28, 28]
        x = torch.cat([x, s1], dim=1) # Concat: [B, 512, 28, 28]
        x = self.fuse1(x) # Fuse: [B, 256, 28, 28]

        x = self.up2(x) # [B, 128, 56, 56]
        s2 = self.skip2(f2) # [B, 128, 56, 56]
        x = torch.cat([x, s2], dim=1) # Concat: [B, 256, 56, 56]
        x = self.fuse2(x) # Fuse: [B, 128, 56, 56]

        x = self.up3(x) # [B, 64, 112, 112]
        s3 = self.skip3(f3) # [B, 64, 112, 112]
        x = torch.cat([x, s3], dim=1) # Concat: [B, 128, 112, 112]
        x = self.fuse3(x) # Fuse: [B, 64, 112, 112]

        x = self.up4(x) # [B, 32, 224, 224]
        s4 = self.skip4(f4) # [B, 32, 224, 224]
        x = torch.cat([x, s4], dim=1) # Concat: [B, 64, 224, 224]
        x = self.fuse4(x) # Fuse: [B, 32, 224, 224]

        return x

class Dust3R(nn.Module):
    def __init__(self):
        super(Dust3R, self).__init__()

        self.CroCo_Encoder = CroCo()
        
        self.patch_embedded_dim = 768

        self.projection_head = ProjectionHead()

        self.decoder_layers = 8
        self.decoders = nn.ModuleList([
            Decoder(d_model=self.patch_embedded_dim, h=12) for _ in range(self.decoder_layers)
        ])

        self.upsampler = FeatureUpsampler()

        self.Head = Head()

        H, W = 224, 224
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        u_flat = x.float().view(-1, 1)
        v_flat = y.float().view(-1, 1)

        self.register_buffer('u', u_flat)
        self.register_buffer('v', v_flat)

    def forward(self, image1, image2):
        # image1 : [B, 3, H, W] [B, 3, 224, 224]
        # image2 : [B, 3, H, W] [B, 3, 224, 224]

        F1, F2 = self.CroCo_Encoder(image1, image2) # [4, B, 196, 768]

        K, E = self.projection_head(F1[-1], F2[-1])
        E_INV = torch.inverse(E)

        G1, G2 = F1[-1], F2[-1]

        for decoder in self.decoders:
            G1, G2 = decoder(G1, G2, E), decoder(G2, G1, E_INV) # [B, 196, 768]

        G1_224x224 = self.upsampler(G1, F1) # [B, 32, 224, 224]
        G2_224x224 = self.upsampler(G2, F2) # [B, 32, 224, 224]

        B, C, H, W = G1_224x224.shape

        Z1, D1 = self.Head(G1_224x224) # [B, 1, 224, 224]
        Z2, D2 = self.Head(G2_224x224) # [B, 1, 224, 224]

        Z1 = Z1.view(B, -1, 1)
        Z2 = Z2.view(B, -1, 1)

        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)

        X1 = (self.u - cx) * Z1 / fx
        Y1 = (self.v - cy) * Z1 / fy
        XYZ1 = torch.cat([X1, Y1, Z1], dim=-1) # 최종 3D 좌표 [B, 50176, 3]

        # 이미지 2의 역투영 (X, Y 계산) - K는 공유한다고 가정
        X2 = (self.u - cx) * Z2 / fx
        Y2 = (self.v - cy) * Z2 / fy
        XYZ2 = torch.cat([X2, Y2, Z2], dim=-1) # 최종 3D 좌표 [B, 50176, 3]

        B = K.shape[0]
        K44 = torch.eye(4, device=K.device).unsqueeze(0).repeat(B, 1, 1)
        K44[:, :3, :3] = K

        MATRIX = torch.bmm(K44, E)[:, :3, :]
        MATRIX_INV = torch.bmm(K44, E_INV)[:, :3, :]

        print(f"True fx: {K[0, 0, 0].item():.2f}, True fy: {K[0, 1, 1].item():.2f}")
        print(f"K : {K}")
        print(f"E : {E}")
        print(f"Z min: {Z1.min().item():.4f}, Z max: {Z1.max().item():.4f}, 갭: {(Z1.max() - Z1.min()).item():.4f}")

        return XYZ1, D1, XYZ2, D2, MATRIX, MATRIX_INV