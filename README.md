# MonoMirror
Self-Supervised 3D Reconstruction from Monocular Vidio

단일 카메라 영상에서 자기지도 학습을 통해 3D 포인트 클라우드와 기하학적 구조를 추출하는 딥러닝 파이프라인 (졸업 프로젝트로 진행할 예정)

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
