# Devlopment Log v4 (진행 과정)

MonoMirror 아키텍처 개발 및 학습 파이프라인 구축 과정 기록

<details>
  <summary>
    2026-06-22 2142 진행상태
  </summary>

  새로운 방식을 적용하고 첫번째

  weight_rgb = 1.0
  weight_reproj = 0.0
  weight_smooth = 0.001
  weight_surface = 0.001
  weight_piece = 0.001

  <img width="672" height="672" alt="ezgif com-animated-gif-maker" src="https://github.com/user-attachments/assets/dd62dc25-2121-48e2-9830-d1d413449446" />

  ```
--- [Fixed Sample Monitoring] ---
True fx: 315.00, True fy: 315.00
K : 
tensor([[[315.,   0., 112.],
         [  0., 315., 112.],
         [  0.,   0.,   1.]]], device='cuda:0')
E_CURR_PREV : 
tensor([[[ 0.5720, -0.7410, -0.3519,  0.0984],
         [ 0.7425,  0.6500, -0.1619,  0.0932],
         [ 0.3487, -0.1686,  0.9219,  0.0999],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
E_CURR_NEXT : 
tensor([[[ 0.4596, -0.8257, -0.3271,  0.0995],
         [ 0.7970,  0.5460, -0.2583,  0.0980],
         [ 0.3918, -0.1420,  0.9090,  0.1000],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
Z min: 0.2000, Z max: 2.4219, 갭: 2.2219
---------------------------------
  ```

  깊이는 평평하게 잡히는거 같은데 보면

  <img width="672" height="672" alt="vis_epoch_500" src="https://github.com/user-attachments/assets/a951b6a4-3da7-4c37-bf84-f5acea1399a3" />

  윤곽선이 분리가 안되었다. 아직 문제가 많다.

</details>


<details>
  <summary>
    2026-06-22 2332 진행상태
  </summary>

  <img width="672" height="672" alt="ezgif com-animated-gif-maker" src="https://github.com/user-attachments/assets/edbf56a7-202a-4e22-bab8-b3675dac5bb1" />

  ```
  ==> Epoch 500 완료 Train Loss : 0.1104 Train feature reproj Loss : 0.5646 Train RGB reproj Loss : 0.106836 Train Smooth Loss : 0.0042 Train Surface Loss : 0.068859 Train Piece Loss : 0.005517 Time : 3.6761
  Saved : ./save/model_save
  --- [Fixed Sample Monitoring] ---
  True fx: 315.00, True fy: 315.00
  K : 
  tensor([[[315.,   0., 112.],
           [  0., 315., 112.],
           [  0.,   0.,   1.]]], device='cuda:0')
  E_CURR_PREV : 
  tensor([[[ 0.6467, -0.6612, -0.3801,  0.0950],
           [ 0.7480,  0.4523,  0.4858, -0.0956],
           [-0.1493, -0.5985,  0.7871,  0.1000],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  E_CURR_NEXT : 
  tensor([[[ 0.6761, -0.6322, -0.3783,  0.0925],
           [ 0.7320,  0.5182,  0.4423, -0.0977],
           [-0.0836, -0.5760,  0.8132,  0.1000],
           [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
  Z min: 0.2008, Z max: 2.4718, 갭: 2.2711
  ---------------------------------
  ```

  <img width="672" height="672" alt="vis_epoch_500" src="https://github.com/user-attachments/assets/e33086de-b1e3-4c21-9286-df800052eb1e" />

  보면 책상과 바닥의 깊이가 비슷하게 나온다. 그냥 보이는것만 부드럽게 나오고 깊이감은 잡히지 않는것 같다.

</details>
