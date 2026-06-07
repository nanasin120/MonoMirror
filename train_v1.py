import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.MonoMirror_v1_5 import MonoMirror_v1_5
from data.ImageDataset import ImageDataset
from defs import get_projected_image, load_croco_weights_to_dust3r, save_fixed_sample
from utils.Loss import Edge_Aware_Smooth_Loss, Feature_Reprojection_Loss, RGB_Reprojection_Loss, Pose_Consistency_Loss
import os
import time
from collections import defaultdict
from torch.optim import Optimizer

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        """
        optimizer: 사용하는 옵티마이저 (AdamW)
        k: 덮어씌우는 빈도
        alpha: 덮어씌우는 정도
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        for group in self.param_groups:
            group["counter"] = 0
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha # 기존 + (현재 - 기존) * alpha
            fast.data.copy_(slow) # 덮어씌우기

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()

model_save_path = r'./save/model_save'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
img_save_path = r'./save/image_save'
if not os.path.exists(img_save_path): os.makedirs(img_save_path)

BATCH = 2
START_EPOCH = 0
END_EPOCH = 500
ADDITIONAL_EPOCH = END_EPOCH-START_EPOCH
LEARNING_RATE = 1e-4 # 1e-4에서 좀 낮춤
IMAGE_SAVE_INTERVEL = 5
WEIGHT_SAVE_INTERVEL = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'./dataset/cup_dataset'
feat_dir = r'./dataset/dino_features'
full_dataset = ImageDataset(img_dir=img_dir, feat_dir=feat_dir, frame_interval=3)

dataloader = DataLoader(
    dataset=full_dataset,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=True
)

model = MonoMirror_v1_5().to(DEVICE)
# model.load_state_dict(torch.load(r'save\model_save\model_epoch_80.pth', weights_only=True))
load_croco_weights_to_dust3r(model, r'./weights/croco_epoch_150.pth')

criterion_edge_smooth = Edge_Aware_Smooth_Loss().to(DEVICE)
criterion_feature_reprojection = Feature_Reprojection_Loss().to(DEVICE)
criterion_rgb_reprojection = RGB_Reprojection_Loss().to(DEVICE)
criterion_pose_consistency_loss = Pose_Consistency_Loss().to(DEVICE)

# backbone_params = []
# head_params = []

# for name, param in model.named_parameters():
#     if "CroCo_Encoder" in name: 
#         backbone_params.append(param)
#     else:
#         head_params.append(param)

# base_optimizer = optim.AdamW([
#     {'params': backbone_params, 'lr': 1e-5}, 
#     {'params': head_params, 'lr': 1e-4}      
# ], weight_decay=1e-4)

base_optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, T_max=ADDITIONAL_EPOCH, eta_min=1e-6)

def train():
    print('TRAIN START')
    best_avg_loss = float('inf')

    for epoch in range(START_EPOCH, END_EPOCH + 1):
        model.train()
        train_loss = 0.0
        train_smooth_loss = 0.0
        train_reproj_loss = 0.0
        train_rgb_loss = 0.0
        train_consist_loss = 0.0
        epoch_start_time = time.time()

        batch_start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            prev_image_vis = batch['prev_image_vis'].to(DEVICE)
            curr_image_vis = batch['curr_image_vis'].to(DEVICE)
            next_image_vis = batch['next_image_vis'].to(DEVICE)

            prev_image_model = batch['prev_image_model'].to(DEVICE)
            curr_image_model = batch['curr_image_model'].to(DEVICE)
            next_image_model = batch['next_image_model'].to(DEVICE)

            prev_feature = batch['prev_feature'].to(DEVICE)
            curr_feature = batch['curr_feature'].to(DEVICE)
            next_feature = batch['next_feature'].to(DEVICE)

            OUTPUTS = model(prev_image_model, curr_image_model, next_image_model)

            CURR_XYZ = OUTPUTS['XYZ'][1]
            
            PREV_D = OUTPUTS['D'][0]
            CURR_D = OUTPUTS['D'][1]
            NEXT_D = OUTPUTS['D'][2]
            
            PREV_MATRIX, NEXT_MATRIX = OUTPUTS['MATRIX'][0], OUTPUTS['MATRIX'][1]

            # -----------------------------------------
            # [1] 특징(Feature) 재투영
            # -----------------------------------------
            projected_img_p2c, valid_mask_p2c = get_projected_image(curr_feature, prev_feature, CURR_XYZ, PREV_MATRIX)
            projected_img_n2c, valid_mask_n2c = get_projected_image(curr_feature, next_feature, CURR_XYZ, NEXT_MATRIX)

            loss_reproj = criterion_feature_reprojection(curr_feature, projected_img_p2c, valid_mask_p2c, projected_img_n2c, valid_mask_n2c)

            # -----------------------------------------
            # [2] RGB 재투영 오차
            # -----------------------------------------
            proj_img_prev, mask_img_prev = get_projected_image(curr_image_vis, prev_image_vis, CURR_XYZ, PREV_MATRIX)
            proj_img_next, mask_img_next = get_projected_image(curr_image_vis, next_image_vis, CURR_XYZ, NEXT_MATRIX)
            
            loss_rgb_reproj = criterion_rgb_reprojection(curr_image_vis, prev_image_vis, next_image_vis, proj_img_prev, mask_img_prev, proj_img_next, mask_img_next)

            # -----------------------------------------
            # [3] 스무스(Smooth) 오차
            # -----------------------------------------
            loss_smoothloss_1 = criterion_edge_smooth(PREV_D, prev_image_vis)
            loss_smoothloss_2 = criterion_edge_smooth(CURR_D, curr_image_vis)
            loss_smoothloss_3 = criterion_edge_smooth(NEXT_D, next_image_vis)
            loss_smoothloss = (loss_smoothloss_1 + loss_smoothloss_2 + loss_smoothloss_3) / 3.0

            # -----------------------------------------
            # [4] 포즈 일관성(Consistency) 오차
            # -----------------------------------------
            loss_consist_prev = criterion_pose_consistency_loss(OUTPUTS['E'][0], OUTPUTS['E_INV'][0])
            loss_consist_next = criterion_pose_consistency_loss(OUTPUTS['E'][1], OUTPUTS['E_INV'][1])
            loss_consist = (loss_consist_prev + loss_consist_next) / 2.0

            weight_reproj = 2.0
            weight_rgb = 2.0
            weight_smooth = 0.05
            weight_consist = 1.0

            # if epoch > 30:
            #     weight_reproj = 3.0
            #     weight_smooth = 0.05
            
            total_loss = (loss_reproj * weight_reproj) + (loss_rgb_reproj * weight_rgb) + (loss_smoothloss * weight_smooth) + (loss_consist * weight_consist)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f"Epoch [{epoch}/{END_EPOCH}] Batch [{batch_idx}/{len(dataloader)}] Loss_total : {total_loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}")
                batch_start_time = time.time()

            train_loss += total_loss.item()
            train_smooth_loss += loss_smoothloss.item()
            train_reproj_loss += loss_reproj.item()
            train_rgb_loss += loss_rgb_reproj.item()
            train_consist_loss += loss_consist.item()

        avg_train_loss = train_loss / len(dataloader)
        avg_train_smooth_loss = train_smooth_loss / len(dataloader)
        avg_reproj_loss = train_reproj_loss / len(dataloader)
        avg_rgb_loss = train_rgb_loss / len(dataloader)
        avg_consist_loss = train_consist_loss / len(dataloader)

        epoch_end_time = time.time()
        scheduler.step()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Train Reproj Loss : {avg_reproj_loss:.4f} Train RGB Loss : {avg_rgb_loss:4f} Train Smooth Loss : {avg_train_smooth_loss:.4f} Train Consist Loss : {avg_consist_loss:4f} Time : {epoch_end_time-epoch_start_time:.4f}')

        if epoch % WEIGHT_SAVE_INTERVEL == 0:
            save_path = os.path.join(model_save_path, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

            print(f'Saved : {model_save_path}')

        if epoch % IMAGE_SAVE_INTERVEL == 0:
            save_fixed_sample(model, full_dataset, epoch, img_save_path, DEVICE, 1)

        if avg_train_loss < best_avg_loss:
            best_avg_loss = avg_train_loss
            save_path = os.path.join(model_save_path, f'best_model_epoch.pth')
            torch.save(model.state_dict(), save_path)

            print(f'New Best Model Saved! Loss : {best_avg_loss:.4f}') 

if __name__ == "__main__":
    train()