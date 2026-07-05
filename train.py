import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.MonoMirror import MonoMirror
from dataset.ImageDataset import DTU_Dataset
from defs import get_projected_image, get_XYZ, save_fixed_sample
from loss.Loss import Edge_Aware_Smooth_Loss, RGB_Reprojection_Loss
import os
import time

model_save_path = r'./save/model_save'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
img_save_path = r'./save/image_save'
if not os.path.exists(img_save_path): os.makedirs(img_save_path)

BATCH = 4
START_EPOCH = 0
END_EPOCH = 500
ADDITIONAL_EPOCH = END_EPOCH-START_EPOCH
LEARNING_RATE = 1e-4
IMAGE_SAVE_INTERVEL = 5
WEIGHT_SAVE_INTERVEL = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_dir = r'C:\Users\MSI\Desktop\DTU\scan65'
# dataset_dir = r'/content/data_local'

full_dataset = DTU_Dataset(dataset_dir, frame_interval=1, H=224, W=224)

dataloader = DataLoader(
    dataset=full_dataset,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=True
)

model = MonoMirror(H=224, W=224).to(DEVICE)

criterion_edge_smooth = Edge_Aware_Smooth_Loss().to(DEVICE)
criterion_rgb_reprojection = RGB_Reprojection_Loss().to(DEVICE)

backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
        
    if "encoder" in name:
        backbone_params.append(param)
    else:
        head_params.append(param)

optim_groups = [{'params': head_params, 'lr': 1e-4}]

if len(backbone_params) > 0:
    optim_groups.append({'params': backbone_params, 'lr': 1e-5})

optimizer = optim.AdamW(optim_groups, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADDITIONAL_EPOCH, eta_min=1e-6)

def train():
    print('TRAIN START')
    best_avg_loss = float('inf')

    for epoch in range(START_EPOCH, END_EPOCH + 1):
        model.train()
        
        train_loss = 0.0
        train_rgb_loss = 0.0
        train_edge_smooth_loss = 0.0

        epoch_start_time = time.time()

        batch_start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            prev_image_model = batch['IMAGE_MODEL'][0].to(DEVICE)
            curr_image_model = batch['IMAGE_MODEL'][1].to(DEVICE)
            next_image_model = batch['IMAGE_MODEL'][2].to(DEVICE)
            
            prev_image_vis = batch['IMAGE_VIS'][0].to(DEVICE)
            curr_image_vis = batch['IMAGE_VIS'][1].to(DEVICE)
            next_image_vis = batch['IMAGE_VIS'][2].to(DEVICE)

            curr_fx = batch['CURR_F'][0].to(DEVICE)
            curr_fy = batch['CURR_F'][1].to(DEVICE)

            curr_K = [curr_fx, curr_fy]

            _, _, H, W = curr_image_vis.shape

            OUTPUTS = model(prev_image_model, curr_image_model, next_image_model, curr_K, False)

            PREV_MATRIX = OUTPUTS['MATRIX_CURR_PREV'][0]
            NEXT_MATRIX = OUTPUTS['MATRIX_CURR_NEXT'][0]
            DISP = OUTPUTS['DISP']

            loss_rgb_reproj = 0.0
            loss_edge_smoothloss = 0.0

            # -------------------------------------------------------------------
            # RGB 재투영과 smooth loss
            # -------------------------------------------------------------------
            for i in range(4):
                DISPARITY = DISP[i]
                DISPARITY = F.interpolate(DISPARITY, size=(H, W), mode='bilinear', align_corners=False)
                DEPTH = 1 / (DISPARITY + 1e-6)
                XYZ = get_XYZ(DEPTH, curr_fx, curr_fy, H, W)

                proj_rgb_prev, valid_mask_prev = get_projected_image(curr_image_vis, prev_image_vis, XYZ, PREV_MATRIX, H, W)
                proj_rgb_next, valid_mask_next = get_projected_image(curr_image_vis, next_image_vis, XYZ, NEXT_MATRIX, H, W)

                loss_rgb_reproj += criterion_rgb_reprojection(curr_image_vis, prev_image_vis, next_image_vis, proj_rgb_prev, valid_mask_prev, proj_rgb_next, valid_mask_next)
                loss_edge_smoothloss += criterion_edge_smooth(DISPARITY, curr_image_vis)

            # 가중치 설정
            weight_rgb = 1.0
            weight_smooth = 0.001

            total_loss = ((loss_rgb_reproj * weight_rgb) + (loss_edge_smoothloss * weight_smooth)) / 4

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f"Epoch [{epoch}/{END_EPOCH}] Batch [{batch_idx}/{len(dataloader)}] Loss_total : {total_loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}")
                batch_start_time = time.time()

            train_loss += total_loss.item()
            train_edge_smooth_loss += loss_edge_smoothloss.item()
            train_rgb_loss += loss_rgb_reproj.item()

        avg_train_loss = train_loss / len(dataloader)
        avg_train_smooth_loss = train_edge_smooth_loss / len(dataloader)
        avg_rgb_loss = train_rgb_loss / len(dataloader)

        epoch_end_time = time.time()
        scheduler.step()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Train RGB reproj Loss : {avg_rgb_loss:4f} Train Smooth Loss : {avg_train_smooth_loss:.4f} Time : {epoch_end_time-epoch_start_time:.4f}')

        if epoch % WEIGHT_SAVE_INTERVEL == 0:
            save_path = os.path.join(model_save_path, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

            print(f'Saved : {model_save_path}')

        if epoch % IMAGE_SAVE_INTERVEL == 0:
            save_fixed_sample(model, full_dataset, epoch, img_save_path, DEVICE)

        if avg_train_loss < best_avg_loss:
            best_avg_loss = avg_train_loss
            save_path = os.path.join(model_save_path, f'best_model_epoch.pth')
            torch.save(model.state_dict(), save_path)

            print(f'New Best Model Saved! Loss : {best_avg_loss:.4f}') 

if __name__ == "__main__":
    train()