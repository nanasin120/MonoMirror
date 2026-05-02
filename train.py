import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from Dust3R import Dust3R
from ImageDataset import ImageDataset
from defs import get_projected_image, load_croco_weights_to_dust3r
from Loss import Minimum_Reprojection_Loss, Smooth_Loss
import os
import time
import torchvision.utils as vutils

def save_fixed_sample(model, dataset, epoch, save_path, device):
    model.eval()
    with torch.no_grad():
        # 고정된 첫 번째 데이터 가져오기
        sample = dataset[0]
        curr_img = sample['current_image'].unsqueeze(0).to(device) # [1, 3, 224, 224]
        next_img = sample['next_image'].unsqueeze(0).to(device) # [1, 3, 224, 224]

        # 모델 추론
        X1, _, _, _, MATRIX, _ = model(curr_img, next_img)

        # 재투영 이미지 생성
        projected_img, _ = get_projected_image(curr_img, next_img, X1, MATRIX)

        depth = X1[:, :, 2].view(1, 1, 14, 14) # [1, 1, 14, 14]로 변환

        # 3. 14x14를 224x224로 확대 (원본 이미지와 크기 맞춤)
        depth_resized = F.interpolate(depth, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 시각화를 위해 0~1 사이로 정규화 (가까운 곳은 밝게, 먼 곳은 어둡게)
        depth_min = depth_resized.min()
        depth_max = depth_resized.max()
        depth_norm = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)
        
        # 3채널로 복사 (이미지 결합을 위해)
        depth_viz = depth_norm.repeat(1, 3, 1, 1)

        # 시각화를 위해 두 이미지 결합 (가로로 붙이기)
        # [1, 3, 224, 224] -> [3, 224, 448]
        combined = torch.cat([curr_img[0], projected_img[0], depth_viz[0]], dim=2)
        
        # 이미지 저장 (0~1 범위 클리핑 및 저장)
        vutils.save_image(combined, os.path.join(save_path, f'vis_epoch_{epoch}.png'))

        print('saved image')

model_save_path = r'./model_save'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
img_save_path = r'./image_save'
if not os.path.exists(img_save_path): os.makedirs(img_save_path)

BATCH = 8
EPOCH = 1000
LEARNING_RATE = 1e-5
IMAGE_SAVE_INTERVEL = 10
WEIGHT_SAVE_INTERVEL = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'cola_cleaned'
full_dataset = ImageDataset(img_dir=img_dir, frame_interval=1)

dataloader = DataLoader(
    dataset=full_dataset,
    batch_size=BATCH,
    shuffle=True
)

model = Dust3R().to(DEVICE)
load_croco_weights_to_dust3r(model, r'croco_epoch_150.pth')

criterion_reprojection = Minimum_Reprojection_Loss().to(DEVICE)
criterion_smooth = Smooth_Loss().to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)

def train():
    print('TRAIN START')
    best_avg_loss = float('inf')

    for epoch in range(0, EPOCH + 1):
        model.train()
        train_loss = 0.0
        train_reproj_loss = 0.0
        train_smooth_loss = 0.0
        epoch_start_time = time.time()

        batch_start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            current_image = batch['current_image'].to(DEVICE)
            next_image = batch['next_image'].to(DEVICE)

            X1, C1, X2, C2, MATRIX, MATRIX_INV = model(current_image, next_image)

            B = X1.shape[0]

            projected_img1, spread_loss_2d_1 = get_projected_image(current_image, next_image, X1, MATRIX)
            projected_img2, spread_loss_2d_2 = get_projected_image(next_image, current_image, X2, MATRIX_INV)

            loss_reproj_1 = criterion_reprojection(current_image, next_image, projected_img1, C1)
            loss_reproj_2 = criterion_reprojection(next_image, current_image, projected_img2, C2)

            loss_smoothloss_1 = criterion_smooth(X1.permute(0, 2, 1).reshape(B, 3, 14, 14), current_image)
            loss_smoothloss_2 = criterion_smooth(X2.permute(0, 2, 1).reshape(B, 3, 14, 14), next_image)

            loss_reproj = (loss_reproj_1 + loss_reproj_2) * 0.5
            loss_smoothloss = (loss_smoothloss_1 + loss_smoothloss_2) * 0.5
            spread_loss_2d = (spread_loss_2d_1 + spread_loss_2d_2) * 0.5

            total_loss = loss_reproj + (loss_smoothloss * 0.001) + (spread_loss_2d * 0.5)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f'Epoch [{epoch}/{EPOCH}] Batch [{batch_idx}/{len(dataloader)}] Loss_total : {total_loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}')
                batch_start_time = time.time()

            train_loss += total_loss.item()
            train_reproj_loss += loss_reproj.item()
            train_smooth_loss += loss_smoothloss.item() * 0.001

        avg_train_loss = train_loss / len(dataloader)
        avg_train_reproj_loss = train_reproj_loss / len(dataloader)
        avg_train_smooth_loss = train_smooth_loss / len(dataloader)

        epoch_end_time = time.time()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Train Reproj Loss : {avg_train_reproj_loss:.4f} Train Smooth Loss : {avg_train_smooth_loss:.4f} Time : {epoch_end_time-epoch_start_time:.4f}')

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

        scheduler.step()

if __name__ == "__main__":
    train()