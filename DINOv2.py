import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2(nn.Module):
    def __init__(self, device='cuda'):
        super(DINOv2, self).__init__()
        print("Loading DINOv2 model...")

        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        self.device = device

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.patch_size = 14

    @torch.no_grad()
    def extract_and_resize(self, image):
        # image : [B, 3, H, W] [B, 3, 224, 224]
        B, C, H, W = image.shape

        out = self.model.forward_features(image)
        features = out['x_norm_patchtokens'] # [B, N, 384]

        h_feat = H // self.patch_size
        w_feat = W // self.patch_size

        # [B, 256, 384] -> [B, 384, 256] -> [B, 384, 16, 16] 으로 변환
        features = features.transpose(1, 2).reshape(B, 384, h_feat, w_feat)

        # [B, 384, 16, 16]을 우리가 원하는 [B, 384, 224, 224] 해상도로 부드럽게 뻥튀기!
        features_resized = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)

        return features_resized