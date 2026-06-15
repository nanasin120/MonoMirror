# Devlopment Log v3 (진행 과정)

MonoMirror 아키텍처 개발 및 학습 파이프라인 구축 과정 기록

가장 기본으로 돌아갈것.

## 목표

**책상 위 물체를 360도로 찍은 영상을 모델에 넣고 과적합 과정을 거쳐 3D Point Cloud로 뽑아내는 모델**

## 파이프 라인

1. 데이터 전처리
2. 깊이(Z)와 이동과 회전(E) 예측
3. 모든 3D Point 조립

## 현재의 문제

### 1. 데이터의 문제 (책상 위 투명한 컵)

현재 사용하는 데이터는 투명한 컵. 물론 안에 내용물이 어느정도 있기는 하지만 뒤의 내용이 비쳐지는건 마찬가지

- 문제점 : 투명 컵이나 비닐 같은 것들은 각도에 따라 빛의 위치나 모양이 바뀜.
- 해결책 : 빛 반사가 없는 불 투명한 데이터를 사용하거나, 새로운 방법을 이용하거나

2026-06-14 2114 하여 물체를 플라스틱 컵이 아닌 노트북으로 변경하였다.

### 2. 데이터 전처리 문제

현재 데이터의 전처리는 세로를 224크기로 만든뒤 CentorCrop(224)로 잘라내는 방식

- 문제점 : 배경이 다 날아가 버리니 이동을 예측하는 부분에서 많은 문제가 생겨버림
- 해결책 : 다시 Resize((224, 224))를 사용하거나 Padding으로 다시 돌아가거나 하는 방식으로 배경 살리기

2026-06-14 2114 현재는 고민중이다.

### 3. 초점 문제

현재 초점을 213으로 강제해놓은 상태

- 문제점 : 결국 스마트폰마다 초점은 다른데 이걸 이렇게 고정시키는게 문제임
- 실제 스펙을 찾거나 파이썬 OpenCV를 통해 실제 K를 알아내기

2026-06-14 2114 현재는 고민중이다.

---

<details>
  <summary>
    2026-06-14 2114 진행상태
  </summary>
  
  일단 크게 보면 세가지가 변형되었다.

  첫번째는 학습 환경, 두번째는 데이터, 세번쨰는 MonoMirror의 구조이다.

  학습 환경은 이전의 노트북이 아닌 구글 코랩에서 돌리고 있다. 잘 돌아간다.

  데이터는 이전의 플라스틱 컵이 아닌 노트북으로 변경하였다.

  MonoMirror는 Encoder를 DINOv2로 변경하였다. 이에 따라 다른 세부적인 것들도 변경되었다.

  <img width="672" height="672" alt="ezgif com-animated-gif-maker" src="https://github.com/user-attachments/assets/2ebe99da-ea8d-45e3-8c77-ab8e82dc1a7f" />

  ```
  ==> Epoch 500 완료 Train Loss : 0.4129 Train Reproj Loss : 0.3450 Train RGB Loss : 0.067034 Train Smooth Loss : 0.0247 Train Consist Loss : 0.058113 Time : 7.1830
  Saved : ./save/model_save
  --- [Fixed Sample Monitoring] ---
  True fx: 213.00, True fy: 213.00
  K : 
  tensor([[[213.,   0., 112.],
           [  0., 213., 112.],
           [  0.,   0.,   1.]]], device='cuda:0')
  E_CURR_PREV : 
  tensor([[[ 0.9971,  0.0539, -0.0533,  0.0548],
           [-0.0438,  0.9837,  0.1745, -0.0662],
           [ 0.0618, -0.1717,  0.9832,  0.0093],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  E_CURR_NEXT : 
  tensor([[[ 0.9980,  0.0571, -0.0267,  0.0293],
           [-0.0530,  0.9893,  0.1357, -0.0722],
           [ 0.0341, -0.1340,  0.9904,  0.0021],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  Z min: 0.1221, Z max: 9.9885, 갭: 9.8664
  ---------------------------------
  ```

  그리고 위에가 현재의 결과이다. 데이터를 노트북으로 바꾼것 하나만으로 이런 결과가 나올것이라곤 예상하지 않았다.

  하지만 현재도 그리 좋은 상태가 아니다. 나는 아직 배가 고프다.
  
</details>

<details>
  <summary>
    2026-06-14 2248 진행상태
  </summary>
  
  초점 거리를 213.0에서 315.0으로 변경했다.

  <img width="672" height="672" alt="ezgif com-animated-gif-maker" src="https://github.com/user-attachments/assets/7d0f94d5-21d2-4845-9c20-e301e0cc72ce" />

  ```
  ==> Epoch 500 완료 Train Loss : 0.4131 Train Reproj Loss : 0.3433 Train RGB Loss : 0.069020 Train Smooth Loss : 0.0251 Train Consist Loss : 0.044181 Time : 7.2417
  Saved : ./save/model_save
  --- [Fixed Sample Monitoring] ---
  True fx: 315.00, True fy: 315.00
  K : 
  tensor([[[315.,   0., 112.],
           [  0., 315., 112.],
           [  0.,   0.,   1.]]], device='cuda:0')
  E_CURR_PREV : 
  tensor([[[ 0.9988,  0.0310, -0.0383,  0.0354],
           [-0.0266,  0.9935,  0.1105, -0.0400],
           [ 0.0415, -0.1094,  0.9931,  0.0092],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  E_CURR_NEXT : 
  tensor([[[ 0.9840,  0.0571, -0.1687,  0.0577],
           [-0.0682,  0.9958, -0.0605, -0.0088],
           [ 0.1645,  0.0710,  0.9838, -0.0026],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  Z min: 0.1256, Z max: 9.9973, 갭: 9.8717
  ---------------------------------
  ```

  보면 그리 좋지는 않다. 일단 디테일들은 어느정도 살아있는게 보이지만 깊이가 깔끔하지가 않다.

  게다가 맨 마지막것을 보면 깊이가 마치 물결처럼 나와있는것이 보인다. 

</details>

<details>
  <summary>
    2026-06-15 0010 진행상태
  </summary>
  
  스케쥴러를 CosineAnnealingWarmRestarts로 변경해 50 에포크마다 학습량을 되돌렸다. 이것이 loss의 정체를 부셔줄거라 믿었다.

  depth의 범위를 0.2 ~ 2.5로 바꿨다.

  <img width="672" height="672" alt="ezgif com-animated-gif-maker" src="https://github.com/user-attachments/assets/5ae26e9b-66cc-43ab-8e0d-b5ffc6ea2d3b" />
  
  ```
  ==> Epoch 500 완료 Train Loss : 0.4234 Train Reproj Loss : 0.3486 Train RGB Loss : 0.074097 Train Smooth Loss : 0.0195 Train Consist Loss : 0.052744 Time : 7.2686
  Saved : ./save/model_save
  --- [Fixed Sample Monitoring] ---
  True fx: 315.00, True fy: 315.00
  K : 
  tensor([[[315.,   0., 112.],
           [  0., 315., 112.],
           [  0.,   0.,   1.]]], device='cuda:0')
  E_CURR_PREV : 
  tensor([[[ 0.9972,  0.0545, -0.0508,  0.0581],
           [-0.0470,  0.9892,  0.1388, -0.0725],
           [ 0.0578, -0.1360,  0.9890,  0.0125],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  E_CURR_NEXT : 
  tensor([[[ 0.9761,  0.0715, -0.2052,  0.1000],
           [-0.0877,  0.9936, -0.0706, -0.0087],
           [ 0.1989,  0.0869,  0.9762, -0.0038],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  Z min: 0.2058, Z max: 2.4998, 갭: 2.2940
  ---------------------------------
  ```

  저 물결은 사라지지를 않고 loss또한 0.4에서 내려가지를 않는다. 디테일은 잘 잡히는거 같은데 저런 부분이 아직 부족하다.

</details>

<details>
  <summary>
    2026-06-15 1450 진행상태
  </summary>

  edge_aware_smooth_loss의 가중치를 0.01에서 0.1로 올렸다. 물결을 없애주길 빌었다.

  <img width="672" height="672" alt="ezgif com-animated-gif-maker" src="https://github.com/user-attachments/assets/e38f0127-a182-49ad-bf16-b7f83b355650" />
  
  ```
  ==> Epoch 500 완료 Train Loss : 0.4213 Train Reproj Loss : 0.3450 Train RGB Loss : 0.073908 Train Smooth Loss : 0.0168 Train Consist Loss : 0.065460 Time : 7.4473
  Saved : ./save/model_save
  --- [Fixed Sample Monitoring] ---
  True fx: 315.00, True fy: 315.00
  K : 
  tensor([[[315.,   0., 112.],
           [  0., 315., 112.],
           [  0.,   0.,   1.]]], device='cuda:0')
  E_CURR_PREV : 
  tensor([[[ 0.9982,  0.0191, -0.0561,  0.0687],
           [-0.0106,  0.9887,  0.1493, -0.0802],
           [ 0.0583, -0.1485,  0.9872,  0.0031],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  E_CURR_NEXT : 
  tensor([[[ 0.9940,  0.1090,  0.0066,  0.0205],
           [-0.1089,  0.9838,  0.1422, -0.0889],
           [ 0.0090, -0.1420,  0.9898,  0.0159],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  Z min: 0.2214, Z max: 2.4984, 갭: 2.2769
  ---------------------------------
  ```

  물결이 안사라진다. 고민고민해보니까 이게 DINOv2가 결국에는 패치 단위로 보는데 그럼 패치별로 또 경계가 생기지 않을까 싶다.

</details>

<details>
  <summary>
    2026-06-16 0001 진행상태
  </summary>

  여러가지의 변화점이 있다. 먼저 LookAhead를 없에고 CosineAnnealingWarmRestarts에서 CosineAnnealingLR로 교체했다. 너무 덕지덕지 붙여놨다는 느낌이 들었다.

  smooth loss를 0.001로 낮추었다. 아예 모든 깊이가 같아지는 문제가 발생해 낮추는 쪽으로 방향을 옮겼다.

  모델에서의 가장 큰 차이점은 DINOv2에 들어가는 이미지의 크기를 224에서 196으로 줄여서 넣은 것이다.

  이전에 224로 넣으니 224 / 14 = 16으로 G와 F가 나오게 되었다. 이게 문제인 이유가 나중에 upsampling에서 값을 올릴때 2배가 아닌 1.75배가 되는 것이다.

  그래서 DINOv2 Encoder에 들어가는 이미지는 따로 크기를 줄여줬고 FeatureUpsampler의 주요 연산 방식들을 변경했다. 

  <img width="672" height="672" alt="ezgif com-animated-gif-maker" src="https://github.com/user-attachments/assets/ed2c8a65-818e-4aaa-99b6-453eec758138" />
  
  ```
  ==> Epoch 500 완료 Train Loss : 0.0631 Train Reproj Loss : 0.0002 Train RGB Loss : 0.061419 Train Smooth Loss : 0.0266 Train Consist Loss : 0.001524 Time : 4.9829
  Saved : ./save/model_save
  --- [Fixed Sample Monitoring] ---
  True fx: 315.00, True fy: 315.00
  K : 
  tensor([[[315.,   0., 112.],
           [  0., 315., 112.],
           [  0.,   0.,   1.]]], device='cuda:0')
  E_CURR_PREV : 
  tensor([[[ 9.9983e-01, -4.3648e-03, -1.8040e-02,  4.3801e-02],
           [ 4.3475e-03,  9.9999e-01, -1.0007e-03,  2.1982e-02],
           [ 1.8044e-02,  9.2207e-04,  9.9984e-01,  2.3840e-02],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]],
         device='cuda:0')
  E_CURR_NEXT : 
  tensor([[[ 9.9976e-01, -3.8339e-05, -2.1906e-02,  5.0753e-02],
           [ 4.8929e-04,  9.9979e-01,  2.0582e-02, -4.6664e-02],
           [ 2.1900e-02, -2.0587e-02,  9.9955e-01, -4.6505e-02],
           [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]],
         device='cuda:0')
  Z min: 0.2046, Z max: 1.9106, 갭: 1.7059
  ---------------------------------
  ```

  물결은 사라졌다. 하지만 노트북의 화면에서 깊이가 달라지는것이 눈에 보인다. 깊이를 깔끔하게 추정하지를 못하고 있다.

</details>
