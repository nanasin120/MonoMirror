import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.MonoMirror import MonoMirror
from dataset.ImageDataset import ImageDataset
from defs import get_projected_image, save_fixed_sample_v4
from loss.Loss import Edge_Aware_Smooth_Loss, RGB_Reprojection_Loss, Surface_Normal_Consistency_Loss, new_Piecewise_Planar_Loss, Feature_Reprojection_Loss
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
full_dataset = ImageDataset(img_dir=dataset_dir, frame_interval=1)

dataloader = DataLoader(
    dataset=full_dataset,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=True
)

model = MonoMirror().to(DEVICE)

criterion_edge_smooth = Edge_Aware_Smooth_Loss().to(DEVICE)
criterion_rgb_reprojection = RGB_Reprojection_Loss().to(DEVICE)
criterion_surface_normal_consistency_loss = Surface_Normal_Consistency_Loss().to(DEVICE)
criterion_planar = new_Piecewise_Planar_Loss().to(DEVICE)
criterion_feature_reprojection = Feature_Reprojection_Loss().to(DEVICE)

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
        train_feature_loss = 0.0
        train_edge_smooth_loss = 0.0
        train_planner = 0.0

        epoch_start_time = time.time()

        batch_start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            prev_image_vis = batch['prev_image_vis'].to(DEVICE)
            curr_image_vis = batch['curr_image_vis'].to(DEVICE)
            next_image_vis = batch['next_image_vis'].to(DEVICE)

            prev_image_model = batch['prev_image_model'].to(DEVICE)
            curr_image_model = batch['curr_image_model'].to(DEVICE)
            next_image_model = batch['next_image_model'].to(DEVICE)

            OUTPUTS = model(prev_image_model, curr_image_model, next_image_model, False)

            PREV_MATRIX = OUTPUTS['PREV_MATRIX']
            NEXT_MATRIX = OUTPUTS['NEXT_MATRIX']
            FEATURE = OUTPUTS['FEATURE']

            loss_rgb_reproj = 0.0
            loss_surface = 0.0
            loss_edge_smoothloss = 0.0
            loss_planar = 0.0
            loss_feature_reproj = 0.0

            # -------------------------------------------------------------------
            # RGB 재투영과 smooth loss
            # -------------------------------------------------------------------
            for i in range(4):
                proj_rgb_prev, valid_mask_prev = get_projected_image(curr_image_vis, prev_image_vis, OUTPUTS['XYZ'][i], PREV_MATRIX)
                proj_rgb_next, valid_mask_next = get_projected_image(curr_image_vis, next_image_vis, OUTPUTS['XYZ'][i], NEXT_MATRIX)

                final_mask_prev = valid_mask_prev
                final_mask_next = valid_mask_next

                loss_rgb_reproj += criterion_rgb_reprojection(curr_image_vis, prev_image_vis, next_image_vis, proj_rgb_prev, final_mask_prev, proj_rgb_next, final_mask_next)
                loss_surface += criterion_surface_normal_consistency_loss(OUTPUTS['XYZ'][i], curr_image_vis)
                loss_edge_smoothloss += criterion_edge_smooth(OUTPUTS['DISP'][i], curr_image_vis)
                loss_planar += criterion_planar(OUTPUTS['DISP'][i], curr_image_vis)
                loss_feature_reproj += criterion_feature_reprojection(FEATURE[1], FEATURE[0], final_mask_prev, FEATURE[2], final_mask_next)

            # 가중치 설정
            weight_rgb = 1.0
            weight_smooth = 0.01
            weight_surface = 0.05
            weight_planar = 0.05
            weight_feature = 1.0

            total_loss = ((loss_rgb_reproj * weight_rgb) + (loss_edge_smoothloss * weight_smooth) + (loss_surface * weight_surface) + (loss_planar * weight_planar) + (loss_feature_reproj * weight_feature)) / 4

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
            train_planner += loss_planar.item()
            train_feature_loss += loss_feature_reproj.item()

        avg_train_loss = train_loss / len(dataloader)
        avg_train_smooth_loss = train_edge_smooth_loss / len(dataloader)
        avg_rgb_loss = train_rgb_loss / len(dataloader)
        avg_planner_loss = train_planner / len(dataloader)
        avg_feature_loss = train_feature_loss / len(dataloader)

        epoch_end_time = time.time()
        scheduler.step()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Train RGB reproj Loss : {avg_rgb_loss:4f} Train Smooth Loss : {avg_train_smooth_loss:.4f} Train Planner Loss : {avg_planner_loss:4f} Train Feature Loss : {avg_feature_loss:4f} Time : {epoch_end_time-epoch_start_time:.4f}')

        if epoch % WEIGHT_SAVE_INTERVEL == 0:
            save_path = os.path.join(model_save_path, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

            print(f'Saved : {model_save_path}')

        if epoch % IMAGE_SAVE_INTERVEL == 0:
            save_fixed_sample_v4(model, full_dataset, epoch, img_save_path, DEVICE)

        if avg_train_loss < best_avg_loss:
            best_avg_loss = avg_train_loss
            save_path = os.path.join(model_save_path, f'best_model_epoch.pth')
            torch.save(model.state_dict(), save_path)

            print(f'New Best Model Saved! Loss : {best_avg_loss:.4f}') 

if __name__ == "__main__":
    train()