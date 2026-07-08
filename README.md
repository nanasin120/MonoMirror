# MoniMirror: Self-Supervised 3D Reconstruction from Monocular Video

단일 카메라 비디오 입력만을 활용하여 정답(GT) 데이터 없이 자기지도 학습(Self-Supervised Learning) 기반으로 3D 기하학적 구조 및 포인트 클라우드를 복원하는 딥러닝 파이프라인입니다.

---

## 1. 주요 특징 (Key Features)

- **비지도/자기지도 학습 기반:** LiDAR 센서나 다중 동기화 카메라, 별도의 3D 정답 데이터 없이 오직 연속된 단일 카메라 비디오 프레임만을 이용해 3D 공간을 복원합니다.
- **ResNet-18 백본 활용:** 대표적인 CNN 아키텍처인 ResNet-18을 인코더로 채택하여 가볍고 빠른 추론 속도(High FPS)를 보장합니다.
- **3-프레임 아키텍처:** 연속된 3개의 프레임(Prev, Curr, Next) 간의 기하학적 일관성을 학습 메커니즘으로 활용합니다.

---

## 2. 개발 환경 및 하드웨어 (Environment & Hardware)

본 프로젝트는 아래의 하드웨어 및 소프트웨어 환경에서 테스트 및 개발되었습니다. (Google Colab 환경에서도 학습 및 구동이 가능합니다.)

| Component | Specification |
| :--- | :--- |
| **Laptop Model** | MSI Alpha 17 C7VG |
| **GPU** | NVIDIA GeForce RTX 4070 Laptop (8GB VRAM) |
| **OS** | Windows 11 |
| **CUDA / Driver** | CUDA 13.1 / Driver 591.74 |

---

## 3. 모델 아키텍처 및 파이프라인 (Architecture & Pipeline)

### 학습 아키텍처 (Input Structure)
- **3-Frame Mechanism:** 현재 프레임($\text{Frame}_{t}$)을 기준으로 Depth를 예측한 후, 이전 프레임($\text{Frame}_{t-1}$) 및 다음 프레임($\text{Frame}_{t+1}$)으로의 Inverse Warping을 통해 픽셀 단위 재투영 오차를 계산합니다.

### 손실 함수 (Loss Functions) & 최적화 (Optimization)
- **Loss Composition**
  - Minimum Reprojection Loss (Weight: 1.0)
  - Edge-Aware Smoothness Loss (Weight: 0.001)
- **Optimizer & Scheduler:** `AdamW` (Initial LR: 1e-4) + `CosineAnnealingLR` (Min LR: 1e-6)

---

## 4. 시작하기 (Getting Started)

### 4-1. Installation
*(추후 업데이트 예정)*

### 4-2. Dataset Preparation
*(추후 업데이트 예정)*

---

## 5. 평가 지표 및 결과 (Evaluation & Results)

### 5-1. 정량적 평가 지표 (Evaluation Metrics)
단일 영상 기반 깊이 예측(Depth Estimation) 및 3D 복원 성능을 평가하기 위해 다음 평가지표를 활용합니다. *(지표 모두 낮을수록 우수)*

| Method | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ |
| :--- | :---: | :---: | :---: |
| **MoniMirror (Ours)** | 0.000 | 0.000 | 0.000 |

### 5-2. 결과 (Results)
최신 학습 상태 및 시각화 결과물입니다.

**설명:**
- 

---
