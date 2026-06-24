import torch
import torch.nn as nn
import torch.nn.functional as F

class Pose_Consistency_Loss(nn.Module): # 이동량용 손실함수, 둘이 항등 행렬 나와야함
    def __init__(self):
        super(Pose_Consistency_Loss, self).__init__()

    def forward(self, E_fwd, E_bwd):
        """
        E_fwd: [B, 4, 4]
        E_bwd: [B, 4, 4]
        """

        B = E_fwd.size(0)
        
        I_target = torch.eye(4, device=E_fwd.device).unsqueeze(0).repeat(B, 1, 1)
        
        E_cycle = torch.bmm(E_fwd, E_bwd)
        
        consistency_loss = torch.abs(E_cycle - I_target).mean()
        
        return consistency_loss

class Mask_Loss(nn.Module): # 마스크가 너무 많으면 손실
    def __init__(self):
        super(Mask_Loss, self).__init__()

    def forward(self, valid_mask_p2c, valid_mask_n2c):
        valid_mask_any = valid_mask_p2c.bool() | valid_mask_n2c.bool()

        valid_ratio = valid_mask_any.sum() / valid_mask_any.numel()

        mask_penalty = torch.relu(0.85 - valid_ratio) * 10.0

        return mask_penalty

class Disparity_3Frame_Loss(nn.Module):
    def __init__(self):
        super(Disparity_3Frame_Loss, self).__init__()

    def forward(self, prev_d_warped, curr_d, next_d_warped):
        pass

class Disparity_Loss(nn.Module):
    def __init__(self):
        super(Disparity_Loss, self).__init__()

    def forward(self, d1, d2, valid_mask=None):
        B = d1.shape[0]

        diff = torch.abs(d1.view(B, -1, 1) - d2.view(B, -1, 1))

        if valid_mask is not None:
            valid_mask = valid_mask.view(B, -1, 1)
            diff = diff * valid_mask
            loss = diff.sum() / (valid_mask.sum() + 1e-7)
        else:
            loss = diff.mean()

        return loss

class pointmap_Loss(nn.Module):
    def __init__(self):
        super(pointmap_Loss, self).__init__()

    def forward(self, p1, p2, valid_mask=None):
        # p1, p2 [B, H * W, 3]
        B = p1.shape[0]

        mean_z1 = p1[:, :, 2:3].mean(dim=1, keepdim=True) + 1e-7 
        mean_z2 = p2[:, :, 2:3].mean(dim=1, keepdim=True) + 1e-7 

        p1_norm = p1 / mean_z1
        p2_norm = p2 / mean_z2

        diff = torch.abs(p1_norm - p2_norm)

        if valid_mask is not None:
            valid_mask = valid_mask.view(B, -1, 1)
            diff = diff * valid_mask
            loss = diff.sum() / (valid_mask.sum() * 3 + 1e-7)
        else:
            loss = diff.mean()

        return loss

class Edge_Aware_Smooth_Loss(nn.Module): # 원본 이미지를 참조하는 Smooth Loss
    def __init__(self):
        super(Edge_Aware_Smooth_Loss, self).__init__()

    def forward(self, disp, img):
        """
        disp: 네트워크가 예측한 시차 (Disparity) [B, 1, H, W]
        img: 원본 입력 이미지 [B, 3, H, W]
        """
        mean_disp = disp.mean(dim=(1, 2, 3), keepdim=True)
        disp = disp / (mean_disp + 1e-7)

        # 깊이(시차)의 변화량(Gradient) 계산
        # [B, 1, H, W-1], [B, 1, H-1, W]
        disp_dx = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        disp_dy = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        # 원본 이미지의 색상 변화량(Gradient) 계산
        # [B, 1, H, W-1], [B, 1, H-1, W]
        img_dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(1, keepdim=True)
        img_dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(1, keepdim=True)

        # 이미지 색상이 변하면 깊이 평활화를 꺼버림 (exp(-색상변화))
        # 색상 변화가 클수록 가중치가 0에 가까워져서 Smooth Loss가 무시됨
        # [B, 1, H, W-1], [B, 1, H-1, W]
        weight_x = torch.exp(-img_dx * 100.0)
        weight_y = torch.exp(-img_dy * 100.0)

        pad_mask = (img.sum(dim=1, keepdim=True) > 0).float()
        mask_x = pad_mask[:, :, :, :-1] * pad_mask[:, :, :, 1:]
        mask_y = pad_mask[:, :, :-1, :] * pad_mask[:, :, 1:, :]

        # 최종 Loss 계산
        # [B, 1, H, W-1], [B, 1, H-1, W]
        smoothness_x = disp_dx * weight_x * mask_x
        smoothness_y = disp_dy * weight_y * mask_y

        return (smoothness_x.sum() + smoothness_y.sum()) / (mask_x.sum() + mask_y.sum() + 1e-8)

class SSIM(nn.Module): # 두 이미지가 얼마나 비슷한가
    def __init__(self, window_size = 3, C1 = 0.01 ** 2, C2 = 0.03 ** 2):
        super(SSIM, self).__init__()
        self.window_size = window_size # 윈도우 사이즈 
        self.C1 = C1 # 밝기는 1%정도 차이나도 인간은 거의 똑같다 느낌
        self.C2 = C2 # 분산은 3%정도 차이나도 인간은 거의 똑같다 느낌 

    def forward(self, x, y): 
        # x는 예측 이미지, y는 정답 이미지
        # 밝기, 대비, 구조 세가지 요소로 비교

        p = self.window_size // 2 # 패딩 크기

        # pad를 통해 이미지 상하좌우에 p만큼 확장
        # 거울처럼 이미지 끝이 1, 2, 3, 4라면 1, 2, 3, 4, 3, 2, 1 이렇게 확장함
        # 이를 통해 avg이후에도 크기가 안줄음
        x = F.pad(x, (p, p, p, p), mode='replicate') 
        y = F.pad(y, (p, p, p, p), mode='replicate')

        # mu는 밝기 
        # avg_pool2d는 이미지에 윈도우를 두고 그 안의 모든 수의 평균을 구함
        mu_x = F.avg_pool2d(x, self.window_size, 1)
        mu_y = F.avg_pool2d(y, self.window_size, 1)

        # sigma는 대비 [분산]
        # 분산은 평균으로부터 얼마나 멀리 퍼져있는가의 척도
        # 분산은 제곱의 평균 - 평균의 제곱
        # 분산이 낮으면 평균에 몰려있음 -> 색 차이가 거의 없음 -> 대비가 낮음
        # 분산이 높으면 평균에 멀리 떨어져있음 -> 색 차이 많음 -> 대비가 높음
        # 대비의 목적은 물체의 경계선, 즉 선명함
        sigma_x = F.avg_pool2d(x ** 2, self.window_size, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, self.window_size, 1) - mu_y ** 2

        # sigma_xy는 구조 [공분산]
        # 공분산은 두 이미지 사이에서 픽셀들이 변하는 모양새가 서로 닮았나
        # 공분산이 양이면 x가 밝아지는 지점에서 y도 밝아짐, 패턴이 일치
        # 공분산이 음이면 x가 밝아지는 지점에서 y는 어두워짐, 패턴 반전
        # 공분산이 0에 가까우면 x가 변할때 y는 아무 상관없이 변함, 패턴 불일치
        # 구조는 윤곽선, 질감 무늬 등 여러가지가 종합된것
        sigma_xy = F.avg_pool2d(x * y, self.window_size, 1) - mu_x * mu_y

        # n은 (평균 밝기가 얼마나 일치하나) * (구조적으로 얼마나 유사하나), 둘이 얼마나 닮았나
        # 곱하기로 연결되었기에 둘다 높으면 확 높아짐
        # self.C1과 self.C2는 0이 되는것을 막기위한 대비책
        n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)

        # n을 0~1로 정규화 해주기 위함이 목적, 두 이미지가 가질 수 있는 최대치
        # 두 이미지가 완전히 같으면 평균이 같을 것이고 2A^2 = A^2 + A^2이다.
        # 두 이미지가 완전히 같으면 분산이 같을 것이고 공분산도 같을 것이고
        # 두 이미지가 완전히 같으면 분산과 공분산이 같으니 2 * A = A + A이다.
        d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # 밝기 일치 * 구조적 일치를 0~1로 정규화한 값
        # 현재는 이미지 크기만큼 ex) [B, 3, H, W]의 값이 반환됨
        return n / d
    
        # 평군을 내고 1에 빼면 스칼라 값이 반환됨
        # 같으면 손실이 0, 다르면 1 이렇게 반환됨
        return 1 - (n / d).mean()

class photometric_error(nn.Module): # 두 이미지가 얼마나 비슷한가, SSIM + L1 Loss
    def __init__(self):
        super(photometric_error, self).__init__()
        self.a = 0.85 # SSIM비율, 논문에 따름
        self.ssim = SSIM()

    def forward(self, image_A, image_B):
        # A는 모델이 만든 사진
        # B는 카메라로 찍은 사진 

        ssim = self.ssim(image_B, image_A) # [B, 3, H, W] 
        
        # 1 - ssim이게 오차임, 이게 0이면 좋은거임
        # /2를 해주는 이유는 압축을 더 안전하게 하기 위해
        # 0 ~ 1은 혹시 모를 안전장치
        # 두 사진이 얼마나 닮았나
        ssim = torch.clamp((1 - ssim) / 2, 0, 1).mean(1, keepdim=True) # [B, 1, H, W]

        # 색상 픽셀끼리 빼고 절대값 씌움
        # L1 loss 픽셀:픽셀 이걸로 확인, 절대값임 제곱 아님
        l1 = torch.abs(image_A - image_B).mean(1, keepdim=True) # [B, 1, H, W]

        # 황금 비율로 섞어서 반환
        # 현재는 스칼라가 아닌 지도를 반환
        pe = self.a * ssim + (1 - self.a) * l1 # [B, 1, H, W]

        return pe

class Minimum_Reprojection_Loss(nn.Module):
    def __init__(self):
        super(Minimum_Reprojection_Loss, self).__init__()
        self.pe = photometric_error()

    def forward(self, target_image, source_image, projected_image, valid_mask):
        # target_image : [B, 3, H, W]
        # projected_image : [B, 3, H, W]

        B, _, H, W = target_image.shape

        bg_mask = (target_image.sum(dim=1, keepdim=True) > 0).float()

        projected_pe = self.pe(target_image, projected_image) # [B, 1, H, W]
        source_pe = self.pe(target_image, source_image) # [B, 1, H, W]

        # mask = (projected_pe < source_pe).float() # [B, 1, H, W]
        # mask = mask * bg_mask * valid_mask

        # min_pe = torch.minimum(projected_pe, source_pe) # auto masking

        weight_loss = projected_pe * valid_mask * bg_mask # [B, 1, H, W]
        
        return weight_loss.sum() / ((valid_mask * bg_mask).sum() + 1e-8)

class U3Frame_Loss(nn.Module): # 이전, 현재, 이후 를 이용한 재투영 오차 + 원본 오차
    def __init__(self):
        super(U3Frame_Loss, self).__init__()
        self.pe = photometric_error()

    def forward(self, prev_img, curr_img, next_img, proj_p2c, mask_p2c, proj_n2c, mask_n2c):
        bg_mask = (curr_img.sum(dim=1, keepdim=True) > 0).float() # 배경 검정색인거 mask

        pe_p2c = self.pe(curr_img, proj_p2c) # prev -> curr가 curr_img와 얼마나 닮았나
        pe_n2c = self.pe(curr_img, proj_n2c) # next -> curr가 curr_img와 얼마나 닮았나

        # 재투영할때 나온 mask 적용
        # 9999.0을 적용하는 이유는 minimum에서 예외 처리하기 위해
        pe_p2c[~mask_p2c.bool()] = 9999.0 
        pe_n2c[~mask_n2c.bool()] = 9999.0

        # minimum은 같은 위치값중 더 작은값만을 이용한 텐서를 만들어냄
        # p2c와 n2c의 오차를 합친 pe 완성
        min_pe_temporal = torch.minimum(pe_p2c, pe_n2c)
        # min_pe_temporal = (pe_p2c + pe_n2c) / 2.0 # 1장 과적합용. 지금은 E가 같게 나와서 안됨

        # auto masking 용
        # 앞차가 계속 나오는 상황을 생각하면 됨
        pe_source_p = self.pe(curr_img, prev_img)
        pe_source_n = self.pe(curr_img, next_img)

        # 원본 오차 합치기
        min_pe_source = torch.minimum(pe_source_p, pe_source_n)

        # 광도 오차와 원본 오차를 합친 최종 오차 완성
        final_min_pe = torch.minimum(min_pe_temporal, min_pe_source)

        # 이전에 재투영 할떄 나온 mask 합치기
        valid_mask_any = mask_p2c.bool() | mask_n2c.bool()

        # 재투영 mask + 배경 mask 합치기
        total_mask = valid_mask_any.float()

        # 최종 Loss = (광도 오차 + 원본 오차) * (재투영 mask + 배경 mask)
        loss = final_min_pe * total_mask
        mean_loss = loss.sum() / (total_mask.sum() + 1e-8)

        valid_ratio = total_mask.sum() / total_mask.numel()
        mask_penalty = torch.relu(0.5 - valid_ratio) * 10.0 # 50% 넘기면 손실

        return mean_loss + mask_penalty

class Smooth_Loss(nn.Module): # 깊이값을 부드럽게 만들어 주는 Loss
    def __init__(self):
        super(Smooth_Loss, self).__init__()

    def forward(self, X, image):
        B, C, H, W = X.shape

        mask = (image.sum(dim=1, keepdim=True) > 0).float()

        # disp의 기울기 (x, y 방향)
        mean_X = torch.abs(X).mean(dim=(1, 2, 3), keepdim=True) # [B, 1, 1, 1] 모든 거리의 평균
        X = X / (mean_X + 1e-7) # 평균으로 나눔 -> 정규화 -> 평균이 1이 됨 -> 일관된 smooth loss 적용 가능

        # [0 ~ N-1] - [1 ~ N] 바로 옆 픽셀의 거리와의 차이
        X_dx = torch.abs(X[:, :, :, :-1] - X[:, :, :, 1:]) # 너비 깊이 [B, 1, H, W]
        X_dy = torch.abs(X[:, :, :-1, :] - X[:, :, 1:, :]) # 높이 깊이 [B, 1, H, W]

        # image의 기울기 (x, y 방향)
        # [0 ~ N-1] - [1 ~ N] 바로 옆 픽셀과의 차이
        # 기울기가 크다면 변화량이 큰것 -> 윤곽선이 있는것
        image_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(1, keepdim=True) # [B, 1, H, W]
        image_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(1, keepdim=True) # [B, 1, H, W]

        # 가중치
        # 변화량이 작으면 무한대 -> 윤곽선이 없다면 가중치 커짐
        # 변화량이 크면 0에 가까워짐 -> 윤곽선이 있다면 가중치 작아짐
        weights_x = torch.exp(-image_dx * 50.0)
        weights_y = torch.exp(-image_dy * 50.0)

        mask_x = mask[:, :, :, :-1] * mask[:, :, :, 1:] # 가로로 인접한 두 칸이 모두 물체인 경우
        mask_y = mask[:, :, :-1, :] * mask[:, :, 1:, :] # 세로로 인접한 두 칸이 모두 물체인 경우

        # 거리차이 * 가중치
        # 윤곽선이 없는데 거리차이가 많다고 예측하면 손실 커짐
        # 윤곽선이 없는데 거리차이가 적다고 예측하면 손실 작아짐
        smoothness_x = (X_dx * weights_x) * mask_x
        smoothness_y = (X_dy * weights_y) * mask_y
        
        return (smoothness_x.sum() + smoothness_y.sum()) / (mask_x.sum() + mask_y.sum() + 1e-8)
    
class Feature_Reprojection_Loss(nn.Module): # 재투영한 특징값 Loss
    def __init__(self):
        super(Feature_Reprojection_Loss, self).__init__()

    def forward(self, curr_feature, projected_img_p2c, valid_mask_p2c, projected_img_n2c, valid_mask_n2c):
        cos_sim_p = F.cosine_similarity(curr_feature, projected_img_p2c, dim=1).unsqueeze(1)
        cos_sim_n = F.cosine_similarity(curr_feature, projected_img_n2c, dim=1).unsqueeze(1)
        
        feat_loss_p = 1.0 - cos_sim_p
        feat_loss_n = 1.0 - cos_sim_n
        
        # 마스크 밖의 픽셀은 오차를 무한대(9999)로 설정하여 선택되지 않게 함
        feat_loss_p[~valid_mask_p2c.bool()] = 9999.0
        feat_loss_n[~valid_mask_n2c.bool()] = 9999.0
        
        # 가려진 부분(Occlusion) 자동 무시
        min_feat_loss = torch.minimum(feat_loss_p, feat_loss_n)
        
        # 합집합 마스크 생성 후 최종 평균 산출
        valid_mask_feat_any = (valid_mask_p2c.bool() | valid_mask_n2c.bool()).float()
        loss_reproj = (min_feat_loss * valid_mask_feat_any).sum() / (valid_mask_feat_any.sum() + 1e-8)

        return loss_reproj

class RGB_Reprojection_Loss(nn.Module): # 재투영한 특징값 Loss
    def __init__(self):
        super(RGB_Reprojection_Loss, self).__init__()
        self.pe = photometric_error()

    def forward(self, curr_image_vis, prev_image_vis, next_image_vis, proj_img_prev, mask_img_prev, proj_img_next, mask_img_next):
        # [B, 3, H, W] -> 3채널 오차의 평균을 내어 [B, 1, H, W]로 변환 (아직 공간 H, W 평균은 내면 안 됨!)
        rgb_loss_p = self.pe(curr_image_vis, proj_img_prev)
        rgb_loss_n = self.pe(curr_image_vis, proj_img_next)
        
        rgb_loss_p[~mask_img_prev.bool()] = float('inf')
        rgb_loss_n[~mask_img_next.bool()] = float('inf')
        
        min_rgb_loss = torch.minimum(rgb_loss_p, rgb_loss_n)

        source_loss_p = self.pe(curr_image_vis, prev_image_vis)
        source_loss_n = self.pe(curr_image_vis, next_image_vis)
        
        min_source_loss = torch.minimum(source_loss_p, source_loss_n)

        final_loss = torch.minimum(min_rgb_loss, min_source_loss)
        
        valid_mask_rgb_any = (mask_img_prev.bool() | mask_img_next.bool()).float()
        loss_rgb_reproj = (final_loss * valid_mask_rgb_any).sum() / (valid_mask_rgb_any.sum() + 1e-8)
        
        return loss_rgb_reproj

class Surface_Normal_Consistency_Loss(nn.Module): # 표면 벡터를 이용한 Loss
    def __init__(self):
        super(Surface_Normal_Consistency_Loss, self).__init__()

    def forward(self, XYZ, image):
        # XYZ: 네트워크가 예측한 3D 좌표 [B, 50176, 3]
        # image: 원본 입력 이미지 [B, 3, 224, 224]
        
        B = XYZ.shape[0]
        H = W = 224
        
        # [B, 50176, 3] -> [B, 3, 224, 224]
        XYZ_img = XYZ.view(B, H, W, 3).permute(0, 3, 1, 2)
        
        # X축, Y축 방향의 3D 표면 벡터 (바로 옆 픽셀과의 거리 차이)
        V_x = XYZ_img[:, :, :, 1:] - XYZ_img[:, :, :, :-1] # 가로 벡터 [B, 3, H, W-1]
        V_y = XYZ_img[:, :, 1:, :] - XYZ_img[:, :, :-1, :] # 세로 벡터 [B, 3, H-1, W]
        
        # 크기를 224로 똑같이 맞추기 위해 끝에 빈 공간을 복사해서 붙여줌 (Padding)
        V_x = F.pad(V_x, (0, 1, 0, 0), mode='replicate') # [B, 3, H, W]
        V_y = F.pad(V_y, (0, 0, 0, 1), mode='replicate') # [B, 3, H, W]

        Normal = torch.cross(V_x, V_y, dim=1) # [B, 3, H, W]
        Normal = F.normalize(Normal, p=2, dim=1) # 이쑤시개 길이를 1로 일정하게 맞춤

        cos_sim_x = F.cosine_similarity(Normal[:, :, :, :-1], Normal[:, :, :, 1:], dim=1).unsqueeze(1)
        cos_sim_y = F.cosine_similarity(Normal[:, :, :-1, :], Normal[:, :, 1:, :], dim=1).unsqueeze(1)

        N_dx = 1.0 - cos_sim_x
        N_dy = 1.0 - cos_sim_y
        
        # 테두리 확인용
        img_dx = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(dim=1, keepdim=True)
        img_dy = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1, keepdim=True)
        
        # 테두리에는 벌점을 주지 않음
        weight_x = torch.exp(-img_dx * 100.0)
        weight_y = torch.exp(-img_dy * 100.0)
        
        # 최종 Loss 계산
        loss_x = (N_dx * weight_x).mean()
        loss_y = (N_dy * weight_y).mean()
        
        return loss_x + loss_y

class Piecewise_Planar_Loss(nn.Module):
    def __init__(self):
        super(Piecewise_Planar_Loss, self).__init__()

    def forward(self, XYZ, image):
        # XYZ: 네트워크가 예측한 3D 좌표 [B, 50176, 3]
        # image: 원본 입력 이미지 [B, 3, 224, 224]

        B = XYZ.shape[0]
        H = W = 224
        
        # [B, 3, 224, 224]
        XYZ_img = XYZ.view(B, H, W, 3).permute(0, 3, 1, 2)
        
        # X축, Y축 방향의 3D 표면 벡터 (바로 옆 픽셀과의 거리 차이)
        V_x = XYZ_img[:, :, :, 1:] - XYZ_img[:, :, :, :-1]
        V_y = XYZ_img[:, :, 1:, :] - XYZ_img[:, :, :-1, :]
        V_x = F.pad(V_x, (0, 1, 0, 0), mode='replicate')
        V_y = F.pad(V_y, (0, 0, 0, 1), mode='replicate')
        
        # 외적하고 정규화
        Normal = torch.cross(V_x, V_y, dim=1)
        Normal = F.normalize(Normal, p=2, dim=1) # [B, 3, 224, 224]
        
        # X축, Y축 방향의 3D 표면 벡터 (바로 옆옆 픽셀과의 거리 차이)
        V_x2 = XYZ_img[:, :, :, 2:] - XYZ_img[:, :, :, :-2]
        V_y2 = XYZ_img[:, :, 2:, :] - XYZ_img[:, :, :-2, :]
        V_x2 = F.pad(V_x2, (0, 2, 0, 0), mode='replicate')
        V_y2 = F.pad(V_y2, (0, 0, 0, 2), mode='replicate')
        
        # dim=1(X, Y, Z 채널) 방향으로 곱하고 더해서 내적 계산
        dot_x = torch.abs(torch.sum(Normal * V_x2, dim=1, keepdim=True))
        dot_y = torch.abs(torch.sum(Normal * V_y2, dim=1, keepdim=True))
        
        # 테두리 확인용
        img_dx = torch.abs(image[:, :, :, :-2] - image[:, :, :, 2:]).mean(dim=1, keepdim=True)
        img_dy = torch.abs(image[:, :, :-2, :] - image[:, :, 2:, :]).mean(dim=1, keepdim=True)
        img_dx = F.pad(img_dx, (0, 2, 0, 0), mode='replicate')
        img_dy = F.pad(img_dy, (0, 0, 0, 2), mode='replicate')
        
        weight_x = torch.exp(-img_dx * 50.0)
        weight_y = torch.exp(-img_dy * 50.0)
        
        loss_x = (dot_x * weight_x).mean()
        loss_y = (dot_y * weight_y).mean()
        
        return loss_x + loss_y

class new_Piecewise_Planar_Loss(nn.Module):
    def __init__(self):
        super(new_Piecewise_Planar_Loss, self).__init__()

    def forward(self, disp, image):
        # ---------------------------------------------------------
        # 방어막 1: Disparity 평균 정규화 (하얗게 타버리는 꼼수 차단!)
        # ---------------------------------------------------------
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7) # 0으로 나누는 것 방지

        # 이제 disp 대신 norm_disp를 가지고 미분을 시작합니다.
        V_x = norm_disp[:, :, :, 1:] - norm_disp[:, :, :, :-1]
        V_y = norm_disp[:, :, 1:, :] - norm_disp[:, :, :-1, :]
        
        V_x2 = V_x[:, :, :, 1:] - V_x[:, :, :, :-1]
        V_y2 = V_y[:, :, 1:, :] - V_y[:, :, :-1, :]

        V_x2 = F.pad(V_x2, (0, 2, 0, 0), mode='replicate')
        V_y2 = F.pad(V_y2, (0, 0, 0, 2), mode='replicate')
        
        # ---------------------------------------------------------
        # 방어막 2: Charbonnier Penalty (V자 미분 폭주 방지!)
        # torch.abs() 대신 사용합니다. 1e-6을 더해 밑바닥을 부드럽게 깎아줍니다.
        # ---------------------------------------------------------
        dot_x = torch.sqrt(V_x2 ** 2 + 1e-6)
        dot_y = torch.sqrt(V_y2 ** 2 + 1e-6)
        
        # 테두리 확인용 (image는 값이 크지 않아 abs를 써도 터지지 않습니다)
        img_dx = torch.abs(image[:, :, :, :-2] - image[:, :, :, 2:]).mean(dim=1, keepdim=True)
        img_dy = torch.abs(image[:, :, :-2, :] - image[:, :, 2:, :]).mean(dim=1, keepdim=True)
        img_dx = F.pad(img_dx, (0, 2, 0, 0), mode='replicate')
        img_dy = F.pad(img_dy, (0, 0, 0, 2), mode='replicate')
        
        weight_x = torch.exp(-img_dx * 100.0)
        weight_y = torch.exp(-img_dy * 100.0)
        
        loss_x = (dot_x * weight_x).mean()
        loss_y = (dot_y * weight_y).mean()
        
        return loss_x + loss_y







