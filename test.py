import torch
from Dust3R import Dust3R
from ImageDataset import ImageDataset
from defs import get_projected_image, load_croco_weights_to_dust3r, visualize_points
from Loss import Minimum_Reprojection_Loss, Smooth_Loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'cup_dataset'
full_dataset = ImageDataset(img_dir=img_dir, frame_interval=3)

model = Dust3R().to(DEVICE)
# load_croco_weights_to_dust3r(model, r'croco_epoch_150.pth')
model.load_state_dict(torch.load(r"model_save\best_model_epoch.pth", weights_only=True))

prev_image = full_dataset[0]['prev_image'].unsqueeze(0).to(DEVICE)
curr_image = full_dataset[0]['curr_image'].unsqueeze(0).to(DEVICE)
next_image = full_dataset[0]['next_image'].unsqueeze(0).to(DEVICE)

OUTPUTS = model(prev_image, curr_image, next_image)
XYZ = OUTPUTS['XYZ']
PREV_XYZ, CURR_XYZ, NEXT_XYZ = XYZ[0], XYZ[1], XYZ[2]

print(CURR_XYZ[..., 0].min(), CURR_XYZ[..., 0].max())
print(CURR_XYZ[..., 1].min(), CURR_XYZ[..., 1].max())
print(CURR_XYZ[..., 2].min(), CURR_XYZ[..., 2].max())

H, W = 224, 224
y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

u_norm = (x.float() / W) - 0.5
v_norm = (y.float() / H) - 0.5
z_val = CURR_XYZ[0, ..., 2].view(224, 224) # 모델이 예측한 Z

# 이제 이 u_norm, v_norm을 XYZ에 사용하면 
# 전체적인 3D 공간의 스케일이 컵의 실제 비율과 비슷해집니다.
XYZ = torch.stack([u_norm.to(DEVICE), v_norm.to(DEVICE), z_val], dim=-1).view(-1, 3)

visualize_points(XYZ, curr_image[0])