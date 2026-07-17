import torch
import torch.nn.functional as F
import open3d as o3d
import torchvision.utils as vutils
import os
import numpy as np

def get_XYZ(Z, fx, fy, cx, cy, H, W):
    device = Z.device
    y, x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )

    u_flat = (x.float()).view(-1, 1)
    v_flat = (y.float()).view(-1, 1)

    Z = Z.view(Z.shape[0], -1, 1)

    fx = fx.view(-1, 1, 1)
    fy = fy.view(-1, 1, 1)
    cx = cx.view(-1, 1, 1)
    cy = cy.view(-1, 1, 1)

    X = (u_flat - cx) * Z / fx
    Y = (v_flat - cy) * Z / fy

    XYZ = torch.cat([X, Y, Z], dim=-1) # 최종 3D 좌표 [B, 50176, 3]

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
        proj_img_p2c, mask_p2c = get_projected_image(curr_image_vis, prev_image_vis, XYZ, PREV_MATRIX, H, W)
        proj_img_n2c, mask_n2c = get_projected_image(curr_image_vis, next_image_vis, XYZ, NEXT_MATRIX, H, W)

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

def get_projected_points(X, MATRIX):
    B = X.shape[0]
    X_dense = X
    ones = torch.ones((B, X_dense.shape[1], 1), device=X.device)
    X_homo = torch.cat([X_dense, ones], dim=-1)

    projected_points = torch.matmul(X_homo, MATRIX.transpose(1, 2)) # [B, H * W, 4]
    return projected_points

def get_projected_image(img1, img2, X, MATRIX, cam_H=224, cam_W=224):
    B, _, img_H, img_W = img1.shape
    projected_points = get_projected_points(X, MATRIX)

    raw_z = projected_points[..., 2]

    z = raw_z.clamp(min=1e-3)
    u = projected_points[..., 0] / z
    v = projected_points[..., 1] / z

    grid_u = (u / (cam_W-1)) * 2.0 - 1.0
    grid_v = (v / (cam_H-1)) * 2.0 - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1).view(B, img_H, img_W, 2)

    projected_img = F.grid_sample(img2, grid, mode='bilinear', padding_mode='border', align_corners=False)

    valid_z = (raw_z > 0).float()
    valid_u = ((grid_u >= -1.0) & (grid_u <= 1.0)).float()
    valid_v = ((grid_v >= -1.0) & (grid_v <= 1.0)).float()
    
    valid_mask = (valid_z).view(B, 1, img_H, img_W)

    return projected_img, valid_mask