import torch
import torch.nn.functional as F
import open3d as o3d
import torchvision.utils as vutils
import os
import numpy as np

def save_fixed_sample_v4(model, dataset, epoch, save_path, device):
    model.eval()
    with torch.no_grad():
        # 고정된 첫 번째 데이터 가져오기
        sample = dataset[0]

        prev_image_vis = sample['prev_image_vis'].unsqueeze(0).to(device)
        curr_image_vis = sample['curr_image_vis'].unsqueeze(0).to(device)
        next_image_vis = sample['next_image_vis'].unsqueeze(0).to(device)

        prev_image_model = sample['prev_image_model'].unsqueeze(0).to(device)
        curr_image_model = sample['curr_image_model'].unsqueeze(0).to(device)
        next_image_model = sample['next_image_model'].unsqueeze(0).to(device)

        # 모델 추론 (sfs 플래그는 필요에 따라 넣으세요)
        OUTPUTS = model(prev_image_model, curr_image_model, next_image_model, True)
        
        # V4에 맞게 깔끔해진 출력값 추출
        CURR_XYZ = OUTPUTS['CURR_XYZ']
        CURR_DISP = OUTPUTS['CURR_DISP']
        PREV_MATRIX = OUTPUTS['PREV_MATRIX']
        NEXT_MATRIX = OUTPUTS['NEXT_MATRIX']

        # RGB 재투영 (224x224)
        proj_img_p2c, mask_p2c = get_projected_image(curr_image_vis, prev_image_vis, CURR_XYZ, PREV_MATRIX)
        proj_img_n2c, mask_n2c = get_projected_image(curr_image_vis, next_image_vis, CURR_XYZ, NEXT_MATRIX)

        # [꿀팁] Error Map 계산 (원본과 당겨온 사진의 차이, 0에 가까울수록 검은색)
        error_p2c = torch.abs(curr_image_vis - proj_img_p2c) * mask_p2c
        error_n2c = torch.abs(curr_image_vis - proj_img_n2c) * mask_n2c

        # 오직 1장뿐인 소중하고 완벽한 현재 깊이 맵!
        viz_d_curr = get_depth_viz(CURR_DISP, curr_image_vis)

        # 3x3 그리드 조립
        # 1행: 과거, 현재, 미래 (입력값)
        row1 = torch.cat([prev_image_vis[0], curr_image_vis[0], next_image_vis[0]], dim=2) 
        # 2행: 과거당겨옴, 현재깊이, 미래당겨옴 (핵심 결과)
        row2 = torch.cat([proj_img_p2c[0], viz_d_curr[0], proj_img_n2c[0]], dim=2)
        # 3행: 과거에러, 현재원본, 미래에러 (Loss 상태 모니터링)
        row3 = torch.cat([error_p2c[0], curr_image_vis[0], error_n2c[0]], dim=2)

        combined = torch.cat([row1, row2, row3], dim=1)
        
        vutils.save_image(combined, os.path.join(save_path, f'vis_epoch_{epoch}.png'))
        print(f"saved V4 3x3 grid image: vis_epoch_{epoch}.png")

def save_fixed_sample(model, dataset, epoch, save_path, device, version):
    model.eval()
    with torch.no_grad():
        # 고정된 첫 번째 데이터 가져오기
        sample = dataset[0]

        prev_image_vis = sample['prev_image_vis'].unsqueeze(0).to(device)
        curr_image_vis = sample['curr_image_vis'].unsqueeze(0).to(device)
        next_image_vis = sample['next_image_vis'].unsqueeze(0).to(device)

        prev_image_model = sample['prev_image_model'].unsqueeze(0).to(device)
        curr_image_model = sample['curr_image_model'].unsqueeze(0).to(device)
        next_image_model = sample['next_image_model'].unsqueeze(0).to(device)

        # 모델 추론
        OUTPUTS = model(prev_image_model, curr_image_model, next_image_model, True)
        
        # ==========================================================
        # [핵심 수정] version 분기 삭제 및 D -> DISP 이름 변경
        # 더 이상 4단계 multi 변수가 없으므로 깔끔하게 바로 꺼냅니다.
        # ==========================================================
        CURR_XYZ = OUTPUTS['XYZ'][1]
        
        PREV_DISP = OUTPUTS['DISP'][0]
        CURR_DISP = OUTPUTS['DISP'][1]
        NEXT_DISP = OUTPUTS['DISP'][2]
        
        PREV_MATRIX, NEXT_MATRIX = OUTPUTS['MATRIX'][0], OUTPUTS['MATRIX'][1]

        # 재투영 이미지 생성
        projected_img_p2c, _ = get_projected_image(curr_image_vis, prev_image_vis, CURR_XYZ, PREV_MATRIX)
        projected_img_n2c, _ = get_projected_image(curr_image_vis, next_image_vis, CURR_XYZ, NEXT_MATRIX)

        # 깊이(시차) 시각화 텐서 생성
        viz_d_prev = get_depth_viz(PREV_DISP, prev_image_vis)
        viz_d_curr = get_depth_viz(CURR_DISP, curr_image_vis)
        viz_d_next = get_depth_viz(NEXT_DISP, next_image_vis)

        # 3x3 그리드 생성
        row1 = torch.cat([prev_image_vis[0], curr_image_vis[0], next_image_vis[0]], dim=2) 
        row2 = torch.cat([viz_d_prev[0], viz_d_curr[0], viz_d_next[0]], dim=2)
        row3 = torch.cat([projected_img_p2c[0], curr_image_vis[0], projected_img_n2c[0]], dim=2)

        combined = torch.cat([row1, row2, row3], dim=1)
        
        vutils.save_image(combined, os.path.join(save_path, f'vis_epoch_{epoch}.png'))
        print(f"saved 3x3 grid image: vis_epoch_{epoch}.png")

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

    # 0.5를 더하는 로직을 추가했기에 (W-1)이 아닌 W로 나눔
    grid_u = (u / cam_W) * 2.0 - 1.0
    grid_v = (v / cam_H) * 2.0 - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1).view(B, img_H, img_W, 2)

    projected_img = F.grid_sample(img2, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    valid_z = (raw_z > 0).float()
    valid_u = ((grid_u >= -1.0) & (grid_u <= 1.0)).float()
    valid_v = ((grid_v >= -1.0) & (grid_v <= 1.0)).float()
    
    valid_mask = (valid_z * valid_u * valid_v).view(B, 1, img_H, img_W)

    return projected_img, valid_mask

def visualize_points(X, image, z_scale=1.0):
    X = X.detach().cpu().numpy().reshape(-1, 3)
    image = image.detach().cpu()
    colors = image.permute(1, 2, 0).reshape(-1, 3).numpy()

    valid_mask = (0.1 < X[:, 2]) & (X[:, 2] < 0.7) # 깊이값 이상치 제거
    
    X = X[valid_mask]
    colors = colors[valid_mask]
    
    X[:, 2] = X[:, 2] * z_scale
    X[:, 1] = X[:, 1] * -1.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def visualize_points_(X, image, z_scale=1.0):
    X = X.detach().cpu().numpy().reshape(-1, 3)
    image = image.detach().cpu()
    colors = image.permute(1, 2, 0).reshape(-1, 3).numpy()

    # 1. 깊이 필터링
    valid_mask = (0.1 < X[:, 2]) & (X[:, 2] < 1.0)
    X = X[valid_mask]
    colors = colors[valid_mask]
    
    # 2. 축 조정
    X[:, 2] = X[:, 2] * z_scale
    X[:, 1] = X[:, 1] * -1.0

    # 3. [강제 평탄화] 가장 낮은 Z(사실상 깊이) 영역을 찾아 수평으로 맞춤
    # 점군 중 y값이 가장 작은(바닥인) 점들의 평균을 잡아 수평 벡터로 삼음
    bottom_points = X[X[:, 1] > np.percentile(X[:, 1], 90)] # 책상 바닥 샘플링
    
    # 책상의 기울기를 수동으로 보정 (0.1은 튜닝 가능)
    # y축(높이)이 컵의 중심을 향하도록 강제 회전
    angle = np.radians(-25) # 이 값을 조절하며 책상이 평평해지는 지점을 찾으세요!
    R = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    X = X @ R.T

    # 4. 시각화
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def load_croco_weights_to_dust3r(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = checkpoint['model_state_dict']

    model_dict = model.state_dict()

    matched_dict = {}

    for k, v in pretrained_dict.items():
        dust3r_key = f"CroCo_Encoder.{k}"

        if dust3r_key in model_dict:
            if v.shape == model_dict[dust3r_key].shape: matched_dict[dust3r_key] = v
        else:
            pass

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)