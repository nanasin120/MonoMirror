import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.MonoMirror_v4 import MonoMirror_v4
from data.ImageDataset import ImageDataset
from defs import get_projected_image, save_fixed_sample_v4
from utils.Loss import Edge_Aware_Smooth_Loss, Feature_Reprojection_Loss, RGB_Reprojection_Loss, Surface_Normal_Consistency_Loss, new_Piecewise_Planar_Loss
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

img_dir = r'./dataset/laptop_dataset'
feat_dir = r'./dataset/dino_features'
# img_dir = r'/content/data_local'
# feat_dir = r'/content/feature_local'
full_dataset = ImageDataset(img_dir=img_dir, feat_dir=feat_dir, frame_interval=3)

dataloader = DataLoader(
    dataset=full_dataset,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=True
)

model = MonoMirror_v4().to(DEVICE)

criterion_edge_smooth = Edge_Aware_Smooth_Loss().to(DEVICE)
criterion_feature_reprojection = Feature_Reprojection_Loss().to(DEVICE)
criterion_rgb_reprojection = RGB_Reprojection_Loss().to(DEVICE)
criterion_surface_normal_consistency_loss = Surface_Normal_Consistency_Loss().to(DEVICE)
criterion_piece_planar_loss = new_Piecewise_Planar_Loss().to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADDITIONAL_EPOCH, eta_min=1e-6)

def train():
    print('TRAIN START')
    best_avg_loss = float('inf')

    for epoch in range(START_EPOCH, END_EPOCH + 1):
        model.train()
        
        train_loss = 0.0
        train_smooth_loss = 0.0
        train_reproj_loss = 0.0
        train_rgb_loss = 0.0
        train_surface_loss = 0.0
        train_piece_loss = 0.0

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

            DISP = OUTPUTS['DISP']
            XYZ = OUTPUTS['XYZ']
            PREV_MATRIX = OUTPUTS['PREV_MATRIX']
            NEXT_MATRIX = OUTPUTS['NEXT_MATRIX']
            PREV_G = OUTPUTS['PREV_G_RAW']
            CURR_G = OUTPUTS['CURR_G_RAW']
            NEXT_G = OUTPUTS['NEXT_G_RAW']

            CURR_DISP = DISP[3]
            CURR_XYZ = XYZ[3]

            B, _, C = CURR_G.shape

            PREV_G_2D = PREV_G.transpose(1, 2).view(B, C, 14, 14)
            CURR_G_2D = CURR_G.transpose(1, 2).view(B, C, 14, 14)
            NEXT_G_2D = NEXT_G.transpose(1, 2).view(B, C, 14, 14)

            CURR_XYZ_2D = CURR_XYZ.transpose(1, 2).view(B, 3, 224, 224)

            CURR_XYZ_14 = F.interpolate(CURR_XYZ_2D, size=(14, 14), mode='bilinear', align_corners=False)
            CURR_XYZ_14_FLAT = CURR_XYZ_14.view(B, 3, -1).transpose(1, 2)
            
            # # -------------------------------------------------------------------
            # # 특징 재투영
            # # -------------------------------------------------------------------
            # proj_feat_p2c, mask_feat_p2c = get_projected_image(CURR_G_2D, PREV_G_2D, CURR_XYZ_14_FLAT, PREV_MATRIX)
            # proj_feat_n2c, mask_feat_n2c = get_projected_image(CURR_G_2D, NEXT_G_2D, CURR_XYZ_14_FLAT, NEXT_MATRIX)

            # loss_reproj = criterion_feature_reprojection(CURR_G_2D, proj_feat_p2c, mask_feat_p2c, proj_feat_n2c, mask_feat_n2c)

            # # -------------------------------------------------------------------
            # # RGB 재투영
            # # -------------------------------------------------------------------
            # proj_rgb_prev_28, mask_rgb_prev_28 = get_projected_image(curr_image_vis, prev_image_vis, XYZ[0], PREV_MATRIX)
            # proj_rgb_next_28, mask_rgb_next_28 = get_projected_image(curr_image_vis, next_image_vis, XYZ[0], NEXT_MATRIX)
            # loss_rgb_reproj_28 = criterion_rgb_reprojection(curr_image_vis, prev_image_vis, next_image_vis, proj_rgb_prev_28, mask_rgb_prev_28, proj_rgb_next_28, mask_rgb_next_28)

            # proj_rgb_prev_56, mask_rgb_prev_56 = get_projected_image(curr_image_vis, prev_image_vis, XYZ[1], PREV_MATRIX)
            # proj_rgb_next_56, mask_rgb_next_56 = get_projected_image(curr_image_vis, next_image_vis, XYZ[1], NEXT_MATRIX)
            # loss_rgb_reproj_56 = criterion_rgb_reprojection(curr_image_vis, prev_image_vis, next_image_vis, proj_rgb_prev_56, mask_rgb_prev_56, proj_rgb_next_56, mask_rgb_next_56)

            # proj_rgb_prev_112, mask_rgb_prev_112 = get_projected_image(curr_image_vis, prev_image_vis, XYZ[2], PREV_MATRIX)
            # proj_rgb_next_112, mask_rgb_next_112 = get_projected_image(curr_image_vis, next_image_vis, XYZ[2], NEXT_MATRIX)
            # loss_rgb_reproj_112 = criterion_rgb_reprojection(curr_image_vis, prev_image_vis, next_image_vis, proj_rgb_prev_112, mask_rgb_prev_112, proj_rgb_next_112, mask_rgb_next_112)

            # proj_rgb_prev_224, mask_rgb_prev_224 = get_projected_image(curr_image_vis, prev_image_vis, XYZ[3], PREV_MATRIX)
            # proj_rgb_next_224, mask_rgb_next_224 = get_projected_image(curr_image_vis, next_image_vis, XYZ[3], NEXT_MATRIX)
            # loss_rgb_reproj_224 = criterion_rgb_reprojection(curr_image_vis, prev_image_vis, next_image_vis, proj_rgb_prev_224, mask_rgb_prev_224, proj_rgb_next_224, mask_rgb_next_224)

            # loss_rgb_reproj = (loss_rgb_reproj_224 * 1.0 + loss_rgb_reproj_112 * 0.1 + loss_rgb_reproj_56 * 0.001 + loss_rgb_reproj_28 * 0.0001)

            # -------------------------------------------------------------------
            # [추가] RGB 사진 멀티스케일 축소 (mode='area' 사용)
            # -------------------------------------------------------------------
            c_img_28 = F.interpolate(curr_image_vis, size=(28, 28), mode='area')
            p_img_28 = F.interpolate(prev_image_vis, size=(28, 28), mode='area')
            n_img_28 = F.interpolate(next_image_vis, size=(28, 28), mode='area')

            c_img_56 = F.interpolate(curr_image_vis, size=(56, 56), mode='area')
            p_img_56 = F.interpolate(prev_image_vis, size=(56, 56), mode='area')
            n_img_56 = F.interpolate(next_image_vis, size=(56, 56), mode='area')

            c_img_112 = F.interpolate(curr_image_vis, size=(112, 112), mode='area')
            p_img_112 = F.interpolate(prev_image_vis, size=(112, 112), mode='area')
            n_img_112 = F.interpolate(next_image_vis, size=(112, 112), mode='area')

            # -------------------------------------------------------------------
            # 특징 재투영 (기존 코드 그대로 유지 - 224 행렬을 쓰지만 14x14로 줄여서 쓰므로 정상 작동)
            # -------------------------------------------------------------------
            proj_feat_p2c, mask_feat_p2c = get_projected_image(CURR_G_2D, PREV_G_2D, CURR_XYZ_14_FLAT, OUTPUTS['PREV_MATRIX_14'])
            proj_feat_n2c, mask_feat_n2c = get_projected_image(CURR_G_2D, NEXT_G_2D, CURR_XYZ_14_FLAT, OUTPUTS['NEXT_MATRIX_14'])
            loss_reproj = criterion_feature_reprojection(CURR_G_2D, proj_feat_p2c, mask_feat_p2c, proj_feat_n2c, mask_feat_n2c)

            # -------------------------------------------------------------------
            # RGB 멀티스케일 재투영 (각 스케일에 맞는 사진, XYZ, MATRIX 투입!)
            # -------------------------------------------------------------------
            # 28x28 (가장 단단한 뼈대)
            proj_rgb_prev_28, mask_rgb_prev_28 = get_projected_image(c_img_28, p_img_28, XYZ[0], PREV_MATRIX[0])
            proj_rgb_next_28, mask_rgb_next_28 = get_projected_image(c_img_28, n_img_28, XYZ[0], NEXT_MATRIX[0])
            loss_rgb_reproj_28 = criterion_rgb_reprojection(c_img_28, p_img_28, n_img_28, proj_rgb_prev_28, mask_rgb_prev_28, proj_rgb_next_28, mask_rgb_next_28)

            # 56x56
            proj_rgb_prev_56, mask_rgb_prev_56 = get_projected_image(c_img_56, p_img_56, XYZ[1], PREV_MATRIX[1])
            proj_rgb_next_56, mask_rgb_next_56 = get_projected_image(c_img_56, n_img_56, XYZ[1], NEXT_MATRIX[1])
            loss_rgb_reproj_56 = criterion_rgb_reprojection(c_img_56, p_img_56, n_img_56, proj_rgb_prev_56, mask_rgb_prev_56, proj_rgb_next_56, mask_rgb_next_56)

            # 112x112
            proj_rgb_prev_112, mask_rgb_prev_112 = get_projected_image(c_img_112, p_img_112, XYZ[2], PREV_MATRIX[2])
            proj_rgb_next_112, mask_rgb_next_112 = get_projected_image(c_img_112, n_img_112, XYZ[2], NEXT_MATRIX[2])
            loss_rgb_reproj_112 = criterion_rgb_reprojection(c_img_112, p_img_112, n_img_112, proj_rgb_prev_112, mask_rgb_prev_112, proj_rgb_next_112, mask_rgb_next_112)

            # 224x224 (가장 날카로운 엣지)
            proj_rgb_prev_224, mask_rgb_prev_224 = get_projected_image(curr_image_vis, prev_image_vis, XYZ[3], PREV_MATRIX[3])
            proj_rgb_next_224, mask_rgb_next_224 = get_projected_image(curr_image_vis, next_image_vis, XYZ[3], NEXT_MATRIX[3])
            loss_rgb_reproj_224 = criterion_rgb_reprojection(curr_image_vis, prev_image_vis, next_image_vis, proj_rgb_prev_224, mask_rgb_prev_224, proj_rgb_next_224, mask_rgb_next_224)

            # 총합 로스 (저해상도일수록 뼈대를 잡는 역할이므로 가중치를 동일하거나 비슷하게 부여!)
            loss_rgb_reproj = (loss_rgb_reproj_224 * 1.0 + loss_rgb_reproj_112 * 1.0 + loss_rgb_reproj_56 * 1.0 + loss_rgb_reproj_28 * 1.0)

            # -------------------------------------------------------------------
            # edge loss
            # -------------------------------------------------------------------
            loss_smoothloss = criterion_edge_smooth(CURR_DISP, curr_image_vis)

            # -------------------------------------------------------------------
            # surface loss
            # -------------------------------------------------------------------
            loss_surface = criterion_surface_normal_consistency_loss(CURR_XYZ, curr_image_vis)

            # -------------------------------------------------------------------
            # piece loss
            # -------------------------------------------------------------------
            loss_piece = criterion_piece_planar_loss(CURR_DISP, curr_image_vis)

            # 가중치 설정
            weight_rgb = 1.0
            weight_reproj = 0.01
            weight_smooth = 0.003
            weight_surface = 0.005
            weight_piece = 0.003
            
            total_loss = (loss_reproj * weight_reproj) + (loss_rgb_reproj * weight_rgb) + (loss_smoothloss * weight_smooth) + (loss_surface * weight_surface) + (loss_piece * weight_piece)

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
            train_surface_loss += loss_surface.item()
            train_piece_loss += loss_piece.item()

        avg_train_loss = train_loss / len(dataloader)
        avg_train_smooth_loss = train_smooth_loss / len(dataloader)
        avg_reproj_loss = train_reproj_loss / len(dataloader)
        avg_rgb_loss = train_rgb_loss / len(dataloader)
        avg_surface_loss = train_surface_loss / len(dataloader)
        avg_piece_loss = train_piece_loss / len(dataloader)

        epoch_end_time = time.time()
        scheduler.step()
        print(f'==> Epoch {epoch} 완료 Train Loss : {avg_train_loss:.4f} Train feature reproj Loss : {avg_reproj_loss:.4f} Train RGB reproj Loss : {avg_rgb_loss:4f} Train Smooth Loss : {avg_train_smooth_loss:.4f} Train Surface Loss : {avg_surface_loss:4f} Train Piece Loss : {avg_piece_loss:4f} Time : {epoch_end_time-epoch_start_time:.4f}')

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