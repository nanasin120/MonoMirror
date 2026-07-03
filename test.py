import torch
from models.MonoMirror_v3 import MonoMirror_v3
from dataset.ImageDataset import ImageDataset
import open3d as o3d
from defs import visualize_points

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'./dataset/laptop_dataset'
feat_dir = r'./dataset/dino_features'
full_dataset = ImageDataset(img_dir=img_dir, feat_dir=feat_dir, frame_interval=3)

model = MonoMirror_v3().to(DEVICE)
model.load_state_dict(torch.load(r"./save/model_save/best_model_epoch.pth", weights_only=True))
model.eval()

prev_image_vis = full_dataset[0]['prev_image_vis'].unsqueeze(0).to(DEVICE)
curr_image_vis = full_dataset[0]['curr_image_vis'].unsqueeze(0).to(DEVICE)
next_image_vis = full_dataset[0]['next_image_vis'].unsqueeze(0).to(DEVICE)

prev_image_model = full_dataset[0]['prev_image_model'].unsqueeze(0).to(DEVICE)
curr_image_model = full_dataset[0]['curr_image_model'].unsqueeze(0).to(DEVICE)
next_image_model = full_dataset[0]['next_image_model'].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    OUTPUTS = model(prev_image_model, curr_image_model, next_image_model)

PREV_XYZ, CURR_XYZ, NEXT_XYZ = OUTPUTS['XYZ'][0], OUTPUTS['XYZ'][1], OUTPUTS['XYZ'][2]
E_PREV_TO_CURR = OUTPUTS['E_INV'][0]
E_NEXT_TO_CURR = OUTPUTS['E_INV'][1]

print(CURR_XYZ[..., 0].min(), CURR_XYZ[..., 0].max())
print(CURR_XYZ[..., 1].min(), CURR_XYZ[..., 1].max())
print(CURR_XYZ[..., 2].min(), CURR_XYZ[..., 2].max())

def apply_transform(XYZ, E):
    B, N, _ = XYZ.shape
    ones = torch.ones((B, N, 1), device=XYZ.device)
    XYZ_homo = torch.cat([XYZ, ones], dim=-1) # [X, Y, Z, 1]로 변환
    XYZ_transformed = torch.matmul(XYZ_homo, E.transpose(1, 2)) # 행렬 곱셈!
    return XYZ_transformed[..., :3] # 다시 1을 떼고 3D 좌표만 반환

# 5. PREV와 NEXT를 CURR 시점으로 순간이동 시키기!
PREV_XYZ_ALIGNED = apply_transform(PREV_XYZ, E_PREV_TO_CURR)
NEXT_XYZ_ALIGNED = apply_transform(NEXT_XYZ, E_NEXT_TO_CURR)

# 6. 3개의 점군과 3장의 색상을 하나로 합치기 (torch.cat)
COMBINED_XYZ = torch.cat([PREV_XYZ_ALIGNED, CURR_XYZ, NEXT_XYZ_ALIGNED], dim=1)

prev_color = prev_image_vis.view(1, 3, -1).transpose(1, 2)
curr_color = curr_image_vis.view(1, 3, -1).transpose(1, 2)
next_color = next_image_vis.view(1, 3, -1).transpose(1, 2)
COMBINED_COLOR = torch.cat([prev_color, curr_color, next_color], dim=1)

def visualize_combined_points(xyz_tensor, color_tensor):
    X = xyz_tensor.squeeze(0).detach().cpu().numpy()
    colors = color_tensor.squeeze(0).detach().cpu().numpy()

    # 조각칼: 거대한 V자 꼬리(배경) 자르기 (0.2m ~ 0.6m 사이만 남김)
    valid_mask = (0.2 < X[:, 2]) & (X[:, 2] < 0.6)
    X = X[valid_mask]
    colors = colors[valid_mask]

    # 축 방향 조정
    X[:, 1] = X[:, 1] * -1.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 마법의 지우개: 통계적 노이즈 제거 (허공에 튀어 나간 불량 점들 깎아내기)
    # 주변에 이웃 점이 부족한 외톨이 픽셀들을 싹 지워줍니다.
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    clean_pcd = pcd.select_by_index(ind)

    print(f"합쳐진 점의 개수: {len(X)}개 -> 노이즈 제거 후: {len(ind)}개")
    o3d.visualization.draw_geometries([clean_pcd])

# 대망의 실행!
visualize_combined_points(COMBINED_XYZ, COMBINED_COLOR)