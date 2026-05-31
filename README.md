# MonoMirror
Self-Supervised 3D Reconstruction from Monocular Vidio

단일 카메라 영상에서 자기지도 학습을 통해 3D 포인트 클라우드와 기하학적 구조를 추출하는 딥러닝 파이프라인 (졸업 프로젝트로 진행할 예정)

자기지도 학습을 하는 가장 큰 이유는 학습에 사용하는 컴퓨터가 개인용 게이밍 노트북이기 때문. 많은 데이터 이런건 학습 자체가 힘들어짐.

## Environment & Hardware

본 프로젝트는 다음 사양의 로컬 환경에서 개발 및 테스트되었습니다.

| Component | Specification |
| :--- | :--- |
| **Laptop Model** | MSI Alpha 17 C7VG |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop (8GB VRAM) |
| **CUDA** | 13.1 |
| **Driver** | 591.74 |
| **OS** | Windows 11 |

## Project Overview
Lider 센서나 다중 카메라 비디오, 정답 데이터 없이 오직 연속된 단일 카메라 비디오 만을 이용하여 3D 공간 복원

## Architecture & Pipeline [IMAGE](./docs/ARCHTICTURE.md)
- Backbone Network: DUSt3R
- Encoder: CroCo (self made, Pre-trained Weights with same Monocular Video)
- Input Structure: 3-Frame Architecture (Prev, Curr, Next)
  - predict depth based on Curr frame, calculate Loss by Wraping Prev and Next Frame to Pixel
- Loss Functions:
  - Minimum Reprojection Loss (Weight: 0.8): 가려짐 및 시야각 이탈 문제를 해결하기 위해 과거/미래 프레임중 오차가 더 작은 쪽으로 학습
  - Edge-Aware Smooth Loss (Weight: 0.05): Depth가 부드러운 평면을 유지하되, 물체의 경계선 입체감은 보존
- Optimizer & Scheduler: AdamW(LR: 1e-4) + CosineAnnealingLR (Min LR: 1e-6)

## 진행 상태 [바로가기](./docs/DEVLOG.md)
진행 상태 기록

## 문제 해결 기록 [바로가기](./docs/TROUBLESHOODING.md)
문제 해결 기록