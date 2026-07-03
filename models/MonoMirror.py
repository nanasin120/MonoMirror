import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        translation = torch.tanh(extrinsic_raw[:, 3:6]) * 1.0
        
        E = torch.eye(4, device=F1.device).unsqueeze(0).repeat(B, 1, 1) # [B, 4, 4]
        E[:, :3, :3] = R
        E[:, :3, 3] = translation

        return E

class RGBFeatureHead(nn.Module):
    def __init__(self):
        super(RGBFeatureHead, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1), # [224, 224]
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1), # [112, 112]
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # [56, 56]
            nn.GELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # [28, 28]
            nn.GELU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # [14, 14]
            nn.GELU()
        )

    def forward(self, img):
        IMG_F_224 = self.conv1(img)
        IMG_F_112 = self.conv2(IMG_F_224)
        IMG_F_56 = self.conv3(IMG_F_112)
        IMG_F_28 = self.conv4(IMG_F_56)
        IMG_F_14 = self.conv5(IMG_F_28)

        return IMG_F_224, IMG_F_112, IMG_F_56, IMG_F_28, IMG_F_14

class FeatureUpsampler(nn.Module):
    def __init__(self, in_channels=768, out_channels=32):
        super(FeatureUpsampler, self).__init__()
        self.in_channels = in_channels

        self.fuse0 = nn.Sequential(nn.Conv2d(in_channels + 128, in_channels, kernel_size=3, padding=1), nn.GELU())

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256), nn.GELU()
        )
        self.skip1 = nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1))
        self.fuse1 = nn.Sequential(nn.Conv2d(256 + 256 + 64, 256, kernel_size=3, padding=1), nn.GELU()) 
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128), nn.GELU()
        )
        self.skip2 = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=1))
        self.fuse2 = nn.Sequential(nn.Conv2d(128 + 128 + 32, 128, kernel_size=3, padding=1), nn.GELU())

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU()
        )
        self.skip3 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1))
        self.fuse3 = nn.Sequential(nn.Conv2d(64 + 64 + 16, 64, kernel_size=3, padding=1), nn.GELU())

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels), nn.GELU()
        )
        self.skip4 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=1))
        self.fuse4 = nn.Sequential(nn.Conv2d(32 + 32 + 8, out_channels, kernel_size=3, padding=1), nn.GELU())
    
        self.gate0 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate4 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        for gate in [self.gate0, self.gate1, self.gate2, self.gate3, self.gate4]:
            final_conv = gate[-2] 
            nn.init.constant_(final_conv.weight, 0.0)
            nn.init.constant_(final_conv.bias, -2.0)

    def forward(self, DINO_G, DINO_F, IMG_F):
        B = DINO_G.shape[0]

        x = DINO_G.transpose(1, 2).view(B, self.in_channels, 14, 14) # [B, 384, 14, 14]

        DINO_F_0, DINO_F_1, DINO_F_2, DINO_F_3 = [f.transpose(1, 2).view(B, self.in_channels, 14, 14) for f in DINO_F] # [B, 384, 14, 14]
        IMG_F_224, IMG_F_112, IMG_F_56, IMG_F_28, IMG_F_14 = IMG_F

        gate0 = self.gate0(IMG_F_14)
        gate1 = self.gate1(IMG_F_28)
        gate2 = self.gate2(IMG_F_56)
        gate3 = self.gate3(IMG_F_112)
        gate4 = self.gate4(IMG_F_224)

        GATE_IMG_F_14 = IMG_F_14 * gate0
        GATE_IMG_F_28 = IMG_F_28 * gate1
        GATE_IMG_F_56 = IMG_F_56 * gate2
        GATE_IMG_F_112 = IMG_F_112 * gate3
        GATE_IMG_F_224 = IMG_F_224 * gate4
        
        x = torch.cat([x, GATE_IMG_F_14], dim=1)
        x = self.fuse0(x)

        # 14 -> 28
        x = self.up1(x) # [B, 384, 14, 14] -> [B, 256, 28, 28]
        s1 = F.interpolate(self.skip1(DINO_F_3), size=(28, 28), mode='bilinear', align_corners=False) # [B, 384, 14, 14] -> [B, 256, 28, 28]
        x = torch.cat([x, s1, GATE_IMG_F_28], dim=1) # [B, 256 + 256 + 32, 28, 28]
        out_28 = self.fuse1(x) # [B, 256 + 256 + 32, 28, 28] -> [B, 256, 28, 28]

        # 28 -> 56
        x = self.up2(out_28) # [B, 256, 28, 28] -> [B, 128, 56, 56]
        s2 = F.interpolate(self.skip2(DINO_F_2), size=(56, 56), mode='bilinear', align_corners=False) # [B, 384, 14, 14] -> [B, 128, 56, 56]
        x = torch.cat([x, s2, GATE_IMG_F_56], dim=1) # [B, 128 + 128 + 16, 56, 56]
        out_56 = self.fuse2(x) 

        x = self.up3(out_56) 
        s3 = F.interpolate(self.skip3(DINO_F_1), size=(112, 112), mode='bilinear', align_corners=False) # [B, 384, 14, 14] -> [B, 64, 112, 112]
        x = torch.cat([x, s3, GATE_IMG_F_112], dim=1) 
        out_112 = self.fuse3(x) 

        # 112 -> 224
        x = self.up4(out_112) 
        s4 = F.interpolate(self.skip4(DINO_F_0), size=(224, 224), mode='bilinear', align_corners=False) # [B, 384, 14, 14] -> [B, 32, 224, 224]
        x = torch.cat([x, s4, GATE_IMG_F_224], dim=1) 
        out_224 = self.fuse4(x) 

        return [out_28, out_56, out_112, out_224]

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
    def __init__(self):
        super().__init__()

        self.encoder = DINOv2Encoder()
        
        self.patch_embedded_dim = 384

        self.positionalEncoding2D = PositionalEncoding2D(d_model=self.patch_embedded_dim, h_patches=14, w_patches=14)

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

        #with torch.no_grad():
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

        IMG_F = self.rgbFeatureHead(curr_img)

        CURR_UP_F = self.upsampler(CURR_G, CURR_F, IMG_F)

        DISP_28 = F.interpolate(self.depth_Head_28(CURR_UP_F[0]), (H, W), mode='bilinear', align_corners=False)
        DISP_56 = F.interpolate(self.depth_Head_56(CURR_UP_F[1]), (H, W), mode='bilinear', align_corners=False)
        DISP_112 = F.interpolate(self.depth_Head_112(CURR_UP_F[2]), (H, W), mode='bilinear', align_corners=False)
        DISP_224 = self.depth_Head_224(CURR_UP_F[3])

        Z_28 = 1.0 / (DISP_28 + 1e-6)
        Z_56 = 1.0 / (DISP_56 + 1e-6)
        Z_112 = 1.0 / (DISP_112 + 1e-6)
        Z_224 = 1.0 / (DISP_224 + 1e-6)

        PREV_MATRIX, _ = self.get_MATRIX(B, K, E_CURR_PREV, E_CURR_PREV_INV)
        NEXT_MATRIX, _ = self.get_MATRIX(B, K, E_CURR_NEXT, E_CURR_NEXT_INV)

        XYZ_28 = self.get_XYZ(B, Z_28, K, H, W)
        XYZ_56 = self.get_XYZ(B, Z_56, K, H, W)
        XYZ_112 = self.get_XYZ(B, Z_112, K, H, W)
        XYZ_224 = self.get_XYZ(B, Z_224, K, H, W)

        if sfs:
            print(f"--- [Fixed Sample Monitoring] ---")
            print(f"True fx: {K[0, 0, 0].item():.2f}, True fy: {K[0, 1, 1].item():.2f}")
            print(f"K : \n{K}")
            print(f"E_CURR_PREV : \n{E_CURR_PREV}")
            print(f"E_CURR_NEXT : \n{E_CURR_NEXT}")
            print(f"Z min: {Z_224.min().item():.4f}, Z max: {Z_224.max().item():.4f}, 갭: {(Z_224.max() - Z_224.min()).item():.4f}")
            print(f"---------------------------------")

        return {
            'DISP' : [DISP_28, DISP_56, DISP_112, DISP_224],
            'XYZ' : [XYZ_28, XYZ_56, XYZ_112, XYZ_224], 
            'FEATURE' : [PREV_G.transpose(1, 2).view(B, self.patch_embedded_dim, 14, 14), CURR_G.transpose(1, 2).view(B, self.patch_embedded_dim, 14, 14), NEXT_G.transpose(1, 2).view(B, self.patch_embedded_dim, 14, 14)],
            'PREV_MATRIX' : PREV_MATRIX,
            'NEXT_MATRIX' : NEXT_MATRIX,
        }
    
    def get_XYZ(self, B, Z, K, H, W):
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