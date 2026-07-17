import torch
import torch.nn.functional as F
import open3d as o3d
import torchvision.utils as vutils
import os
import numpy as np

def get_XYZ(
        Z: torch.Tensor, # [B, 1, H, W]
        fx: torch.Tensor, # [B, 1]
        fy: torch.Tensor, # [B, 1]
        cx: torch.Tensor, # [B, 1]
        cy: torch.Tensor, # [B, 1]
        H: int, 
        W: int
    ) -> torch.Tensor: # [B, H * W, 3]
    """
        깊이를 3차원 좌표로 변환하는 함수    
    
        Args:
            Z: 깊이 맵 [B, 1, H, W]
            fx: 카메라의 x축 초점 거리 [B, 1]
            fy: 카메라의 y축 초점 거리 [B, 1]
            cx: 카메라의 x축 중심 좌표 [B, 1]
            cy: 카메라의 y축 중심 좌표 [B, 1]
        Returns:
            XYZ: 3D 좌표 [B, H * W, 3]
    """

    device = Z.device

    # 0 ~ H-1, 0 ~ W-1 범위의 좌표 생성
    y, x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )

    # 2차원 좌표를 1차원으로 펼치기
    u_flat = (x.float() + 0.5).view(-1, 1)
    v_flat = (y.float() + 0.5).view(-1, 1)

    # 계산 편하게 정리
    Z = Z.view(Z.shape[0], -1, 1)
    fx = fx.view(-1, 1, 1)
    fy = fy.view(-1, 1, 1)
    cx = cx.view(-1, 1, 1)
    cy = cy.view(-1, 1, 1)

    # 카메라 기준 3차원 좌표 계산
    X = (u_flat - cx) * Z / fx
    Y = (v_flat - cy) * Z / fy

    XYZ = torch.cat([X, Y, Z], dim=-1) # 최종 3D 좌표 [B, H * W, 3]

    return XYZ

def save_fixed_sample(model, dataset, epoch, save_path, device): # 항상 같은 이미지로만 하기 위해 새로 추출
    model.eval()
    with torch.no_grad():
        sample = dataset[0]

        prev_image_vis = sample['IMAGE_VIS'][0].unsqueeze(0).to(device)
        curr_image_vis = sample['IMAGE_VIS'][1].unsqueeze(0).to(device)
        next_image_vis = sample['IMAGE_VIS'][2].unsqueeze(0).to(device)

        _, _, H, W = curr_image_vis.shape

        prev_image_model = sample['IMAGE_MODEL'][0].unsqueeze(0).to(device)
        curr_image_model = sample['IMAGE_MODEL'][1].unsqueeze(0).to(device)
        next_image_model = sample['IMAGE_MODEL'][2].unsqueeze(0).to(device)

        fx = sample['F'][0].unsqueeze(0).to(device)
        fy = sample['F'][1].unsqueeze(0).to(device)
        K = [fx, fy]

        cx = sample['C'][0].unsqueeze(0).to(device)
        cy = sample['C'][1].unsqueeze(0).to(device)
        C = [cx, cy]

        OUTPUTS = model(prev_image_model, curr_image_model, next_image_model, K, C, True)
        
        DISP = OUTPUTS['DISP'][-1]
        DEPTH = 1 / (DISP + 1e-6)
        XYZ = get_XYZ(DEPTH, fx, fy, cx, cy, H, W)
        PREV_MATRIX = OUTPUTS['MATRIX_CURR_PREV'][0]
        NEXT_MATRIX = OUTPUTS['MATRIX_CURR_NEXT'][0]

        # RGB 재투영 (224x224)
        proj_img_p2c, mask_p2c = get_projected_image(curr_image_vis, prev_image_vis, XYZ, PREV_MATRIX)
        proj_img_n2c, mask_n2c = get_projected_image(curr_image_vis, next_image_vis, XYZ, NEXT_MATRIX)

        viz_d_curr = get_depth_viz(DISP, curr_image_vis)

        # 1행: 과거, 현재, 미래 (입력값)
        row1 = torch.cat([prev_image_vis[0], curr_image_vis[0], next_image_vis[0]], dim=2) 
        # 2행: 과거당겨옴, 현재깊이, 미래당겨옴 (핵심 결과)
        row2 = torch.cat([proj_img_p2c[0], viz_d_curr[0], proj_img_n2c[0]], dim=2)

        row3 = torch.cat([mask_p2c[0].repeat(3, 1, 1), torch.ones_like(viz_d_curr[0]), mask_n2c[0].repeat(3, 1, 1)], dim=2)

        combined = torch.cat([row1, row2, row3], dim=1)
        
        vutils.save_image(combined, os.path.join(save_path, f'vis_epoch_{epoch}.png'))
        print(f"saved V4 3x3 grid image: vis_epoch_{epoch}.png")

def get_depth_viz(depth_tensor, img_tensor):
    mask = (img_tensor.sum(dim=1, keepdim=True) > 0)
    valid_depths = depth_tensor[mask]
    
    if len(valid_depths) > 0:
        d_min, d_max = valid_depths.min(), valid_depths.max()
    else:
        d_min, d_max = torch.tensor(0.0), torch.tensor(1.0)
        
    norm = (depth_tensor - d_min) / (d_max - d_min + 1e-8)
    norm = norm * mask.float()
    return norm.repeat(1, 3, 1, 1) # 3채널 복사

def axis_angle_to_matrix(rot_vec):
    batch_size = rot_vec.shape[0]
    angle = torch.norm(rot_vec, dim=-1, keepdim=True)
    axis = rot_vec / (angle + 1e-7)
    
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)
    
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    
    # Skew-symmetric matrix 생성
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1)
    ], dim=1)
    
    I = torch.eye(3, device=rot_vec.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # 회전 행렬 R 계산
    R = I + sin_a * K + (1 - cos_a) * torch.bmm(K, K)
    return R

def get_projected_points(
        XYZ: torch.Tensor,  # [B, H * W, 3]
        MATRIX_to_from: torch.Tensor,  # [B, 3, 4]
    ) -> torch.Tensor:  # projected_points
    """
        XYZ(img_to)의 좌표계를 MATRIX연산을 통해 img_from의 좌표계로 바꾸는 함수
    
        Args:
            XYZ: 3D 포인트 클라우드 [B, H * W, 3]
            MATRIX_to_from: 카메라 프로젝션 행렬 [B, 3, 4]
        Returns:
            projected_points: 투영된 2D 포인트 [B, H * W, 3]
    """
    B = XYZ.shape[0]

    ones = torch.ones((B, XYZ.shape[1], 1), device=XYZ.device)
    X_homo = torch.cat([XYZ, ones], dim=-1) # 동차 좌표계로 변경

    projected_points = torch.matmul(X_homo, MATRIX_to_from.transpose(1, 2)) # [B, H * W, 3]

    return projected_points

def get_projected_image(
        img_to: torch.Tensor,     # [B, C, H, W]
        img_from: torch.Tensor,     # [B, C, H, W]
        XYZ: torch.Tensor,      # [B, H * W, 3]
        MATRIX_to_from: torch.Tensor,   # [B, 3, 4]
    ) -> tuple[torch.Tensor, torch.Tensor]: # [projected_img, valid_mask]
    """
        img_from을 img_to로 투영하는 함수

        Args:
            img_to: 목적지 이미지
            img_from: 원본 이미지
            XYZ: 3D 포인트 클라우드
            MATRIX_to_from: 카메라 프로젝션 행렬

        Returns:
            projected_img: 합성된 이미지 [B, 3, H, W]
            valid_mask: 유효 픽셀 마스크 [B, 1, H, W]
    """

    B, _, img_H, img_W = img_to.shape

    projected_points = get_projected_points(XYZ, MATRIX_to_from) # [B, H * W, 3]

    # 3차원 좌표계를 2차원 좌표계로 변환하는 과정
    raw_z = projected_points[..., 2]
    z = raw_z.clamp(min=1e-3)
    u = projected_points[..., 0] / z
    v = projected_points[..., 1] / z

    # -1 ~ 1 범위로 정규화
    grid_u = (u / img_W) * 2.0 - 1.0
    grid_v = (v / img_H) * 2.0 - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1).view(B, img_H, img_W, 2)

    # from의 이미지의 픽셀을 to의 이미지 좌표계로 샘플링
    projected_img = F.grid_sample(img_from, grid, mode='bilinear', padding_mode='border', align_corners=False)

    # 이미지 밖으로 나간 픽셀 마스킹 처리
    valid_z = (raw_z > 0).float()
    valid_u = ((grid_u >= -1.0) & (grid_u <= 1.0)).float()
    valid_v = ((grid_v >= -1.0) & (grid_v <= 1.0)).float()
    valid_uv = valid_u * valid_v

    valid_mask = (valid_z * valid_uv).view(B, 1, img_H, img_W)

    return projected_img, valid_mask