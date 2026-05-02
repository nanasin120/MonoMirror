import torch
from Dust3R import Dust3R
from ImageDataset import ImageDataset
from defs import get_projected_image, load_croco_weights_to_dust3r, visualize_points
from Loss import Minimum_Reprojection_Loss, Smooth_Loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'cup'
full_dataset = ImageDataset(img_dir=img_dir, frame_interval=1)

model = Dust3R().to(device)
criterion_reprojection = Minimum_Reprojection_Loss().to(device)
criterion_smooth = Smooth_Loss().to(device)

load_croco_weights_to_dust3r(model, r'croco_epoch_150.pth')

current_image = full_dataset[0]['current_image'].unsqueeze(0).to(device)
next_image = full_dataset[0]['next_image'].unsqueeze(0).to(device)

X1, C1, X2, C2, MATRIX, MATRIX_INV = model(current_image, next_image)

B = 1

projected_img1 = get_projected_image(current_image, next_image, X1, MATRIX)
projected_img2 = get_projected_image(next_image, current_image, X2, MATRIX_INV)

loss_reproj_1 = criterion_reprojection(current_image, next_image, projected_img1, C1)
loss_reproj_2 = criterion_reprojection(next_image, current_image, projected_img2, C2)

loss_smoothloss_1 = criterion_smooth(X1.permute(0, 2, 1).reshape(B, 3, 14, 14), current_image)
loss_smoothloss_2 = criterion_smooth(X2.permute(0, 2, 1).reshape(B, 3, 14, 14), next_image)

loss_reproj = (loss_reproj_1 + loss_reproj_2) * 0.5
loss_smoothloss = (loss_smoothloss_1 + loss_smoothloss_2) * 0.5

total_loss = loss_reproj + loss_smoothloss * 0.001

visualize_points(X1[0])