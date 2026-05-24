import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from Dust3R import Dust3R
from ImageDataset import ImageDataset
from defs import get_projected_image, load_croco_weights_to_dust3r, save_fixed_sample, get_projected_points
from Loss import Minimum_Reprojection_Loss, Smooth_Loss, Edge_Aware_Smooth_Loss, pointmap_Loss, Disparity_Loss, U3Frame_Loss
import os
import time

model_save_path = r'./model_save'
if not os.path.exists(model_save_path): os.makedirs(model_save_path)
img_save_path = r'./image_save'
if not os.path.exists(img_save_path): os.makedirs(img_save_path)

BATCH = 4
START_EPOCH = 0
END_EPOCH = 1000
ADDITIONAL_EPOCH = END_EPOCH-START_EPOCH
LEARNING_RATE = 5e-5 # 1e-4에서 좀 낮춤
IMAGE_SAVE_INTERVEL = 5
WEIGHT_SAVE_INTERVEL = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = r'cup_dataset'
full_dataset = ImageDataset(img_dir=img_dir, frame_interval=3)

dataloader = DataLoader(
    dataset=full_dataset,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=True
)

model = Dust3R().to(DEVICE)
# model.load_state_dict(torch.load(r'model_save\best_model_epoch.pth', weights_only=True))
load_croco_weights_to_dust3r(model, r'croco_epoch_150.pth')

criterion_reprojection = Minimum_Reprojection_Loss().to(DEVICE)
criterion_smooth = Smooth_Loss().to(DEVICE)
criterion_edge_smooth = Edge_Aware_Smooth_Loss().to(DEVICE)
criterion_pointmap_loss = pointmap_Loss().to(DEVICE)
criterion_disparity_loss = Disparity_Loss().to(DEVICE)
criterion_u3frame_loss = U3Frame_Loss().to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(dataloader),
    epochs=ADDITIONAL_EPOCH,
    pct_start=0.1, # 10%동안 warm up
    div_factor=25.0, # max_lr / 25로 시작
    final_div_factor=1000.0 # 마지막 학습률은 0에 가깝게 
)

def train():
    print('TRAIN START')
    best_avg_loss = float('inf')

    for epoch in range(START_EPOCH, END_EPOCH + 1):
        model.train()
        train_loss = 0.0
        train_smooth_loss = 0.0
        train_reproj_loss = 0.0
        epoch_start_time = time.time()

        batch_start_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            prev_image = batch['prev_image'].to(DEVICE)
            curr_image = batch['curr_image'].to(DEVICE)
            next_image = batch['next_image'].to(DEVICE)

            OUTPUTS = model(prev_image, curr_image, next_image)

            XYZ = OUTPUTS['XYZ']
            D = OUTPUTS['D']
            MATRIX = OUTPUTS['MATRIX']
            MATRIX_INV = OUTPUTS['MATRIX_INV']

            PREV_XYZ, CURR_XYZ, NEXT_XYZ = XYZ[0], XYZ[1], XYZ[2]
            PREV_D, CURR_D, NEXT_D = D[0], D[1], D[2]
            PREV_MATRIX, NEXT_MATRIX = MATRIX[0], MATRIX[1]
            PREV_MATRIX_INV, NEXT_MATRIX_INV = MATRIX_INV[0], MATRIX_INV[1]

            projected_img_p2c, valid_mask_p2c = get_projected_image(curr_image, prev_image, CURR_XYZ, PREV_MATRIX)

            projected_img_n2c, valid_mask_n2c = get_projected_image(curr_image, next_image, CURR_XYZ, NEXT_MATRIX)

            loss_3frame = criterion_u3frame_loss(prev_image, curr_image, next_image, projected_img_p2c, valid_mask_p2c, projected_img_n2c, valid_mask_n2c)
            #loss_reproj_1 = criterion_reprojection(curr_image, prev_image, projected_img_p2c, valid_mask_p2c)
            #loss_reproj_2 = criterion_reprojection(curr_image, next_image, projected_img_n2c, valid_mask_n2c)

            loss_smoothloss_1 = criterion_edge_smooth(PREV_D, prev_image)
            loss_smoothloss_2 = criterion_edge_smooth(CURR_D, curr_image)
            loss_smoothloss_3 = criterion_edge_smooth(NEXT_D, next_image)

            loss_reproj = loss_3frame
            loss_smoothloss = (loss_smoothloss_1 + loss_smoothloss_2 + loss_smoothloss_3) / 3.0

            total_loss = (loss_reproj * 1.0) + (loss_smoothloss * 0.005)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if batch_idx % 10 == 0:
                batch_end_time = time.time()
                print(f'Epoch [{epoch}/{END_EPOCH}] Batch [{batch_idx}/{len(dataloader)}] Loss_total : {total_loss.item():.4f} Time : {batch_end_time-batch_start_time:.4f}')
                batch_start_time = time.time()

            train_loss += total_loss.item()
            train_smooth_loss += loss_smoothloss.item() * 0.001
            train_reproj_loss += loss_reproj.item() * 1.0

        avg_train_loss = train_loss / len(dataloader)
        avg_train_smooth_loss = train_smooth_loss / len(dataloader)
        avg_reproj_loss = train_reproj_loss / len(dataloader)

        epoch_end_time = time.time()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Train Reproj Loss : {avg_reproj_loss:.4f} Train Smooth Loss : {avg_train_smooth_loss:.4f} Time : {epoch_end_time-epoch_start_time:.4f}')

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