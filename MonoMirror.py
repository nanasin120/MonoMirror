import torch
import torch.nn as nn
import torch.nn.functional as F
from Dust3R import Dust3R

class MonoMirror(nn.Module):
    def __init__(self):
        super(MonoMirror, self).__init__()

        self.dust3R = Dust3R()

    def forward(self, image1, image2):
        X1, C1, X2, C2, MATRIX, MATRIX_INV = self.dust3R(image1, image2)
        X2_transformed = self.transform_coordinates(X2, MATRIX_INV)
        weights = self.get_weight(X1, X2_transformed, C1)
        pass

    def transform_coordinates(self, X2, MATRIX_INV):
        X2_homogeneous_coordinates = torch.cat([X2, torch.ones_like(X2[..., :1])], dim=-1)
        X2_transformed = torch.matmul(X2_homogeneous_coordinates, MATRIX_INV.transpose(1, 2))
        X2_transformed = X2_transformed[..., :3]
        return X2_transformed
    
    def get_weight(self, X1, X2_transformed, C1):
        dist = torch.norm(X1 - X2_transformed, dim=-1) # 두 뷰에서 예측한거 거리가 멀면 잘 못 예측한거
        consistency_mask = torch.exp(-dist * 5.0) # 거리가 멀수록 신뢰도는 떨어짐, 5.0으로 조절
        refined_confidence = C1 * consistency_mask # 신뢰도 결합
        final_weights = torch.clamp(refined_confidence, min=0.1) # 0~1로 정규화하는데 최소값은 0.1로
        return final_weights