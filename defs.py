import torch
import torch.nn.functional as F
import open3d as o3d
import torchvision.utils as vutils
import os

def save_fixed_sample(model, dataset, epoch, save_path, device):
    model.eval()
    with torch.no_grad():
        # 고정된 첫 번째 데이터 가져오기
        sample = dataset[0]
        curr_img = sample['current_image'].unsqueeze(0).to(device) # [1, 3, 224, 224]
        next_img = sample['next_image'].unsqueeze(0).to(device) # [1, 3, 224, 224]

        # 모델 추론
        XYZ1, C1, D1, XYZ2, C2, D2, MATRIX, MATRIX_INV = model(curr_img, next_img)

        # 재투영 이미지 생성
        projected_img, _ = get_projected_image(curr_img, next_img, XYZ1, MATRIX)

        depth_resized = D1.view(1, 1, 224, 224) # [1, 1, 14, 14]로 변환

        fg_mask = (curr_img[0].sum(dim=0, keepdim=True) > 0) # [1, 224, 224]
        valid_depths = depth_resized[0][fg_mask]
        
        # 시각화를 위해 0~1 사이로 정규화 (가까운 곳은 밝게, 먼 곳은 어둡게)
        depth_min = valid_depths.min()
        depth_max = valid_depths.max()
        depth_norm = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)
        depth_norm = depth_norm * fg_mask.float()

        print(f"Disp min: {depth_min.item():.4f}, Disp max: {depth_max.item():.4f}, 갭: {(depth_max - depth_min).item():.4f}")
        
        # 3채널로 복사 (이미지 결합을 위해)
        depth_viz = depth_norm.repeat(1, 3, 1, 1)

        # 시각화를 위해 두 이미지 결합 (가로로 붙이기)
        # [1, 3, 224, 224] -> [3, 224, 448]
        combined = torch.cat([curr_img[0], projected_img[0], depth_viz[0]], dim=2)
        
        # 이미지 저장 (0~1 범위 클리핑 및 저장)
        vutils.save_image(combined, os.path.join(save_path, f'vis_epoch_{epoch}.png'))

        print('saved image')

def axis_angle_to_matrix(rot_vec):
    batch_size = rot_vec.shape[0]
    angle = torch.norm(rot_vec + 1e-7, dim=-1, keepdim=True)
    axis = rot_vec / angle
    
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

def get_projected_image(img1, img2, X, MATRIX):
    B, _, H, W = img1.shape
    X_dense = X
    ones = torch.ones((B, X_dense.shape[1], 1), device=X.device)
    X_homo = torch.cat([X_dense, ones], dim=-1)

    projected_points = torch.matmul(X_homo, MATRIX.transpose(1, 2)) # [B, H * W, 4]

    raw_z = projected_points[..., 2]
    z = raw_z.clamp(min=1e-3)
    u = projected_points[..., 0] / z
    v = projected_points[..., 1] / z

    # print(f"U range: {u.min().item():.2f} ~ {u.max().item():.2f}")
    # print(f"V range: {v.min().item():.2f} ~ {v.max().item():.2f}")

    # fx = MATRIX[0, 0, 0].item()
    # fy = MATRIX[0, 1, 1].item()
    # print(f"Focal Length: fx={fx:.2f}, fy={fy:.2f}")

    # # 2. 투영 전 3D 포인트의 분포 확인
    # x_mean, x_std = X_homo[..., 0].mean().item(), X_homo[..., 0].std().item()
    # z_mean, z_std = X_homo[..., 2].mean().item(), X_homo[..., 2].std().item()
    # print(f"3D Point X mean: {x_mean:.4f}, std: {x_std:.4f}")
    # print(f"3D Point Z mean: {z_mean:.4f}, std: {z_std:.4f}")

    grid_u = (u / (W - 1)) * 2.0 - 1.0
    grid_v = (v / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1).view(B, H, W, 2)

    projected_img = F.grid_sample(img2, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    valid_z = (raw_z > 0).float()
    valid_u = ((grid_u >= -1.0) & (grid_u <= 1.0)).float()
    valid_v = ((grid_v >= -1.0) & (grid_v <= 1.0)).float()
    
    valid_mask = (valid_z * valid_u * valid_v).view(B, 1, H, W)

    return projected_img, valid_mask

def visualize_points(X, image, z_scale=10.0):
    X = X.detach().cpu().numpy()
    X[:, 2] = X[:, 2] * z_scale
    X[:, 1] = X[:, 1] * -1.0
    image = image.detach().cpu()
    colors = image.permute(1, 2, 0).reshape(-1, 3).numpy()

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