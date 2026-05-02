import torch
import torch.nn.functional as F
import open3d as o3d

def axis_angle_to_matrix(rot_vec):
    batch_size = rot_vec.shape[0]
    angle = torch.norm(rot_vec, dim=-1, keepdim=True) + 1e-7
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

    X_map = X.permute(0, 2, 1).view(B, 3, 14, 14)
    X_dense = F.interpolate(X_map, size=(H, W), mode='bilinear', align_corners=False)
    X_dense = X_dense.permute(0, 2, 3, 1).reshape(B, -1, 3) # [B, H * W, 3]

    ones = torch.ones((B, X_dense.shape[1], 1), device=X.device)
    X_homo = torch.cat([X_dense, ones], dim=-1)

    projected_points = torch.matmul(X_homo, MATRIX.transpose(1, 2)) # [B, H * W, 4]
    z = projected_points[..., 2].clamp(min=1e-3)

    u = projected_points[..., 0] / z
    v = projected_points[..., 1] / z

    print(f"U range: {u.min().item():.2f} ~ {u.max().item():.2f}")
    print(f"V range: {v.min().item():.2f} ~ {v.max().item():.2f}")

    fx = MATRIX[0, 0, 0].item()
    fy = MATRIX[0, 1, 1].item()
    print(f"Focal Length: fx={fx:.2f}, fy={fy:.2f}")

    # 2. 투영 전 3D 포인트의 분포 확인
    x_mean, x_std = X_homo[..., 0].mean().item(), X_homo[..., 0].std().item()
    z_mean, z_std = X_homo[..., 2].mean().item(), X_homo[..., 2].std().item()
    print(f"3D Point X mean: {x_mean:.4f}, std: {x_std:.4f}")
    print(f"3D Point Z mean: {z_mean:.4f}, std: {z_std:.4f}")

    grid_u = (u / (W - 1)) * 2.0 - 1.0
    grid_v = (v / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1).view(B, H, W, 2)

    spread_loss_u = F.relu(0.4 - grid_u.std())
    spread_loss_v = F.relu(0.4 - grid_v.std())
    spread_loss_2d = spread_loss_u + spread_loss_v

    projected_img = F.grid_sample(img2, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return projected_img, spread_loss_2d

def visualize_points(X):
    X = X.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
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