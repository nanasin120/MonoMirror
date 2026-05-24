# MonoMirror
Self-Supervised 3D Reconstruction from Monocular Vidio

단일 카메라 영상에서 자기지도 학습을 통해 3D 포인트 클라우드와 기하학적 구조를 추출하는 딥러닝 파이프라인 (졸업 프로젝트로 진행할 예정)

자기지도 학습을 하는 가장 큰 이유는 학습에 사용하는 컴퓨터가 개인용 게이밍 노트북이기 때문. 많은 데이터 이런건 학습 자체가 힘들어짐.

## Project Overview
Lider 센서나 다중 카메라 비디오, 정답 데이터 없이 오직 연속된 단일 카메라 비디오 만을 이용하여 3D 공간 복원

## Architecture & Pipeline
- Backbone Network: DUSt3R
- Encoder: CroCo (self made, Pre-trained Weights with same Monocular Video)
- Input Structure: 3-Frame Architecture (Prev, Curr, Next)
  - predict depth based on Curr frame, calculate Loss by Wraping Prev and Next Frame to Pixel
- Loss Functions:
  - Minimum Reprojection Loss (Weight: 0.8): 가려짐 및 시야각 이탈 문제를 해결하기 위해 과거/미래 프레임중 오차가 더 작은 쪽으로 학습
  - Edge-Aware Smooth Loss (Weight: 0.05): Depth가 부드러운 평면을 유지하되, 물체의 경계선 입체감은 보존
- Optimizer & Scheduler: AdamW(LR: 1e-4) + CosineAnnealingLR (Min LR: 1e-6)

## Current Progress: Pipeline Sanity Check (Overfitting Test)
본격적인 전체 데이터셋 학습에 앞서, 단 3장의 이미지를 사용하여 파이프라인의 수학적 결함이 없는지 검증하는 과적합 테스트
- Test Setup: 3 Frames, Batch Size 4, 4,400 Epochs
- Key Result & Observations:
<img width="672" height="672" alt="image" src="https://github.com/user-attachments/assets/b90cd1ad-111f-48a9-adba-d7bb606b865c" />
<img width="956" height="557" alt="image" src="https://github.com/user-attachments/assets/7c224dde-9718-46ad-93f2-610db9c3a8f2" />
<img width="953" height="557" alt="image" src="https://github.com/user-attachments/assets/0b89c50d-e1ea-4aac-89bc-1dd634dae478" />
<img width="953" height="554" alt="image" src="https://github.com/user-attachments/assets/6599156f-58f8-4cc7-b4ab-4383facb3a53" />

## 2026-05-24 진행 상태
데이터셋을 이용해 과적합중
- Test Setup : interval=3, Batch Size=4, 240 Epochs
- Loss : smoothloss=0.001, U3frame=1.0
- Key Result & Observations:
<img width="672" height="672" alt="image" src="https://github.com/user-attachments/assets/0fe4635a-4f0c-4699-bceb-30a36a073aa6" />
<img width="1906" height="1105" alt="image" src="https://github.com/user-attachments/assets/f546e7a7-5127-4f06-bba7-bd9d7e31da10" />

## 2026-05-24 1844 진행 상태
아키텍처, 손실함수, 학습 방식 변화 발생
### 아키텍처
아키텍처의 경우 디코더에 PositionalEncoding2D 추가, K값 고정, Upsampling + DepthHead 다시 추가
### 손실 함수
모델이 자꾸 픽셀을 전부 밖으로 던져버리는 꼼수를 찾음 그래서
```
        valid_ratio = total_mask.sum() / total_mask.numel()
        mask_penalty = torch.relu(0.5 - valid_ratio) * 10.0 # 50% 넘기면 손실

        return mean_loss + mask_penalty
```
U3FrameLoss에 다음과 같은 손실을 넣어서 이미지를 재투영 할때 픽셀이 많이 밖으로 나가면 손실을 얻게 만듬
### 학습
에포크는 1000, LEARNING_RATE는 5e-5, schedular는 OneCycleLR로 변경
<img width="672" height="672" alt="ezgif com-animated-gif-maker (2)" src="https://github.com/user-attachments/assets/47898643-af6d-4c2c-b4b9-21a4260fd36d" />

순서대로 0, 20, 40, 60, 80, 100. 100번에서 터진것처럼 보이지만

<img width="672" height="672" alt="image" src="https://github.com/user-attachments/assets/81def350-3c84-4ae8-81cd-9fda4d92d9cd" />

140에서 다시 돌아오기 시작. 그래서 현재는 계속 학습 진행중
### 결과
<img width="672" height="672" alt="ezgif com-animated-gif-maker" src="https://github.com/user-attachments/assets/c5678219-7ae5-4a7f-92b6-6bc0e694b181" />

0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 535 순서. 535에서 멈춘 이유는 더이상 변하는 것이 없는것 같아서

Epoch [535/1000] Batch [0/9] Loss_total : 0.0621 Time : 0.5008
```
==> Epoch 535 완료 Train Loss : 0.0754 Train Reproj Loss : 0.0753 Train Smooth Loss : 0.0000 Time : 5.3203
--- [Fixed Sample Monitoring] ---
True fx: 160.00, True fy: 160.00
K : 
tensor([[[160.,   0., 112.],
         [  0., 160., 112.],
         [  0.,   0.,   1.]]], device='cuda:0')
E_CURR_PREV : 
tensor([[[ 0.9762,  0.2050,  0.0706, -0.0461],
         [-0.1998,  0.9770, -0.0744,  0.0467],
         [-0.0842,  0.0585,  0.9947,  0.0500],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
E_CURR_NEXT : 
tensor([[[ 0.9742,  0.2174,  0.0600, -0.0460],
         [-0.2119,  0.9735, -0.0861,  0.0463],
         [-0.0771,  0.0711,  0.9945,  0.0500],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
Z min: 0.2000, Z max: 19.6051, 갭: 19.4051
```
위와 같이 최종적으로 결과가 나옴
### 분석
먼저 535의 깊이들을 보면 신기하게도 뚜껑 부분은 깊이가 추출이 안됨. 투명한 부분은 잘 안된다는게 이런것 같음

그리고 curr_next와 curr_prev의 E가 거의 같음. 아마도 손실함수에서 사용하는 min_pe_temporal = torch.minimum(pe_p2c, pe_n2c) 이 부분이 문제인것 같음 



---
# 겪은 문제들
## 1. translation 학습 안됨
- 증상 : 카메라의 이동이 0.1로 고정되어 학습이 되지 않음
- 원인 : 가중치를 0.0으로 초기화 하는 바람에 기울기가 0이 되어 학습이 진행되지 않음
- 해결 : 가중치를 std=1e-5로 초기화
## 2. depth는 그대로인데 loss는 떨어짐
- 증상 : depth는 gap없이 0.999로 고정되어있는데 Loss는 계속해서 떨어짐
- 원인 : 초점거리를 최소로 줄여 강제로 맞춤
- 해결 : torch.tanh(intrinsic_raw) * 100.0 + 200.0를 이용해 200에서 시작해 100 ~ 300사이의 값을 갖게 함
## 3. Minimum_Reprojection_Loss 버그
- 증상 : Minimum_Reprojection_Loss가 제대로 작동을 안함
- 원인 : mask 적용에서 잘못 덮어씌움. [pe_n2c[~mask_p2c.bool()]]
- 해결 : [pe_n2c[~mask_p2c.bool()]]를 pe_n2c[~mask_n2c.bool()]로 수정
## 4. 연산량이 너무 커짐
- 증상 : upsampling과 depthHead로 이어지는 과정에서 연산량이 너무나도 커짐
- 원인 : 수많은 conv2d, skip connection 등이 주요 이유
- 해결 : Instant-NGP에서 영감받은 ImplictDepthHead로 깊이 추정 아키텍처를 교체
## 5. 카메라 외부 파라미터가 똑같이 나옴 [2025-05-24]
- 증상 : prev와 next의 카메라 외부 파라미터가 똑같이 나옴. 이는 서로가 똑같이 이동하고 회전했다는것인데 말이 되지 않음
- 원인 : 데이터 부족으로 추측함
- 해결 : F1, F2만 concat했던것과 달리 서로의 차이 Diff = F1 - F2도 concat에 포함해서 학습
