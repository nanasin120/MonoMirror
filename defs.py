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
        prev_image = sample['prev_image'].unsqueeze(0).to(device)# [1, 3, 224, 224]
        curr_image = sample['curr_image'].unsqueeze(0).to(device)
        next_image = sample['next_image'].unsqueeze(0).to(device)

        # 모델 추론
        OUTPUTS = model(prev_image, curr_image, next_image, True)
        
        XYZ = OUTPUTS['XYZ']
        D = OUTPUTS['D']
        MATRIX = OUTPUTS['MATRIX']
        MATRIX_INV = OUTPUTS['MATRIX_INV']

        PREV_XYZ, CURR_XYZ, NEXT_XYZ = XYZ[0], XYZ[1], XYZ[2]
        PREV_D, CURR_D, NEXT_D = D[0], D[1], D[2]
        PREV_MATRIX, NEXT_MATRIX = MATRIX[0], MATRIX[1]
        PREV_MATRIX_INV, NEXT_MATRIX_INV = MATRIX_INV[0], MATRIX_INV[1]

        # 재투영 이미지 생성
        projected_img_p2c, _ = get_projected_image(curr_image, prev_image, CURR_XYZ, PREV_MATRIX)
        projected_img_n2c, _ = get_projected_image(curr_image, next_image, CURR_XYZ, NEXT_MATRIX)

        depth_resized_p = PREV_D.view(1, 1, 224, 224) # [1, 1, 14, 14]로 변환
        depth_resized_n = NEXT_D.view(1, 1, 224, 224) # [1, 1, 14, 14]로 변환

        viz_d_prev = get_depth_viz(PREV_D, prev_image)
        viz_d_curr = get_depth_viz(CURR_D, curr_image)
        viz_d_next = get_depth_viz(NEXT_D, next_image)

        row1 = torch.cat([prev_image[0], curr_image[0], next_image[0]], dim=2) 
        row2 = torch.cat([viz_d_prev[0], viz_d_curr[0], viz_d_next[0]], dim=2)
        row3 = torch.cat([projected_img_p2c[0], curr_image[0], projected_img_n2c[0]], dim=2)

        combined = torch.cat([row1, row2, row3], dim=1)
        
        vutils.save_image(combined, os.path.join(save_path, f'vis_epoch_{epoch:03d}.png'))
        print(f"saved 3x3 grid image: vis_epoch_{epoch:03d}.png")

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


def get_projected_image(img1, img2, X, MATRIX):
    B, _, H, W = img1.shape
    projected_points = get_projected_points(X, MATRIX)

    raw_z = projected_points[..., 2]
    z = raw_z.clamp(min=1e-3)
    u = projected_points[..., 0] / z
    v = projected_points[..., 1] / z

    grid_u = (u / (W - 1)) * 2.0 - 1.0
    grid_v = (v / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1).view(B, H, W, 2)

    projected_img = F.grid_sample(img2, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    valid_z = (raw_z > 0).float()
    valid_u = ((grid_u >= -1.0) & (grid_u <= 1.0)).float()
    valid_v = ((grid_v >= -1.0) & (grid_v <= 1.0)).float()
    
    valid_mask = (valid_z * valid_u * valid_v).view(B, 1, H, W)

    return projected_img, valid_mask

def visualize_points(X, image, z_scale=1.0):
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