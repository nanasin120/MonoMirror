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

