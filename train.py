import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from Dust3R import Dust3R
from ImageDataset import ImageDataset
from defs import get_projected_image, load_croco_weights_to_dust3r, save_fixed_sample
from Loss import Minimum_Reprojection_Loss, Smooth_Loss, Edge_Aware_Smooth_Loss
import os
import time

model_save_path = r'./model_save'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
img_save_path = r'./image_save'
if not os.path.exists(img_save_path): os.makedirs(img_save_path)

BATCH = 4
EPOCH = 500
LEARNING_RATE = 1e-5
IMAGE_SAVE_INTERVEL = 25
WEIGHT_SAVE_INTERVEL = 501
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'cup'
full_dataset = ImageDataset(img_dir=img_dir, frame_interval=10)

dataloader = DataLoader(
    dataset=full_dataset,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=True
)

model = Dust3R().to(DEVICE)
load_croco_weights_to_dust3r(model, r'croco_epoch_150.pth')

criterion_reprojection = Minimum_Reprojection_Loss().to(DEVICE)
criterion_smooth = Smooth_Loss().to(DEVICE)
criterion_edge_smooth = Edge_Aware_Smooth_Loss().to(DEVICE)

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

            XYZ1, C1, D1, XYZ2, C2, D2, MATRIX, MATRIX_INV = model(current_image, next_image)

            B = XYZ1.shape[0]

            D1 = D1.permute(0, 2, 1).reshape(B, 1, 224, 224)
            D2 = D2.permute(0, 2, 1).reshape(B, 1, 224, 224)

            projected_img1, valid_mask1 = get_projected_image(current_image, next_image, XYZ1, MATRIX)
            projected_img2, valid_mask2 = get_projected_image(next_image, current_image, XYZ2, MATRIX_INV)

            loss_reproj_1 = criterion_reprojection(current_image, next_image, projected_img1, C1, valid_mask1)
            loss_reproj_2 = criterion_reprojection(next_image, current_image, projected_img2, C2, valid_mask2)

            # loss_smoothloss_1 = criterion_smooth(Z1, current_image)
            # loss_smoothloss_2 = criterion_smooth(Z2, next_image)

            loss_smoothloss_1 = criterion_edge_smooth(D1, current_image)
            loss_smoothloss_2 = criterion_edge_smooth(D2, next_image)

            loss_reproj = (loss_reproj_1 + loss_reproj_2) * 0.5
            loss_smoothloss = (loss_smoothloss_1 + loss_smoothloss_2) * 0.5

            total_loss = loss_reproj + (loss_smoothloss * 0.01)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f'Epoch [{epoch}/{EPOCH}] Batch [{batch_idx}/{len(dataloader)}] Loss_total : {total_loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}')
                batch_start_time = time.time()

            train_loss += total_loss.item()
            train_reproj_loss += loss_reproj.item()
            train_smooth_loss += loss_smoothloss.item() * 0.01

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