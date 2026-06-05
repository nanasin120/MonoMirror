# 01. Edge Aware Smooth Loss

## 1. 목표
DepthMap의 전체적인 구조를 부드럽게 유지하되(Smooth), 이미지상에서 색상 변화가 큰 영역(Edge)에서는 Depth의 단절을 허용하여 물체의 윤곽선을 보존하는 것입니다.

## 2. 수식

```math
\mathcal{L}_{smooth} = |\nabla d_x| \cdot e^{-|\nabla I_x|} + |\nabla d_y| \cdot e^{-|\nabla I_y|}
```

$\nabla d$: 깊이 맵의 변화량 (인접 픽셀 간 차이)

$\nabla I$: 원본 이미지의 색상 변화량 (인접 픽셀 간 차이)

$e^{-|\nabla I|}$: 이미지의 에지 강도가 높을수록 가중치를 $0$에 가깝게 만들어, 에지 부근에서는 스무스 로스의 적용을 무시함.

## 3. 코드

### 입력값

```
def forward(self, disp, img):
```

disp와 img를 입력받습니다.

disp는 disparity로 시차를 나타냅니다. [B, 1, H, W]

img는 원본 이미지로 외곽선을 추출할때 이용할 것입니다. [B, 3, H, W]

### $\nabla d$: 깊이 맵 변화량 (Smooth)

```
# [B, 1, H, W-1], [B, 1, H-1, W]
disp_dx = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
disp_dy = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
```

disp의 [-1]는 W로 너비입니다. 즉 x입니다. 이들의 차이의 절댓값을 구하면 변화량을 알 수 있습니다.

y도 똑같이 절댓값을 통해 변화량을 구합니다.

만약 이 절댓값이 크다면 깊이가 급격하게 변화한다는 의미입니다.

### $\nabla I$: 원본 이미지의 색상 변화량 (Edge)

```
# [B, 1, H, W-1], [B, 1, H-1, W]
img_dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(1, keepdim=True)
img_dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(1, keepdim=True)
```

이 식에서도 똑같이 절댓값으로 계산을 진행합니다. 

마지막의 ```mean(1, keepdim=True)```를 RGB의 평균치를 구하고 차원을 유지합니다.

최종 결과물은 [B, 1, H, 1], [B, 1, 1, W]이 됩니다.

### $e^{-|\nabla I|}$: 가중치 (Weight)

```
# [B, 1, H, W-1], [B, 1, H-1, W]
weight_x = torch.exp(-img_dx * 10.0)
weight_y = torch.exp(-img_dy * 10.0)
```

img_dx와 img_dy는 위에서 구한 이미지의 색상 변화량입니다.

잠시 torch.exp의 그래프를 구글에서 찾아 0이하의 부분을 보시면 굉장히 천천히 하락하는 그래프임을 확인할 수 있습니다.

보시면 img_dx와 img_dy에 마이너스 부호가 붙어있습니다.

즉, 색상 변화량이 큰놈들은 weight가 낮아지는 것입니다.

이미지에서 색상 변화량이 크다면 외곽선일 가능성이 큽니다.

결국 외곽선의 weight가 낮아지게 됩니다.

### 최종 Loss

```
# [B, 1, H, W-1], [B, 1, H-1, W]
smoothness_x = disp_dx * weight_x
smoothness_y = disp_dy * weight_y

# [1]
return smoothness_x.mean() + smoothness_y.mean()
```

disp_dx와 disp_dy는 깊이 변화량이고 weight_x, weight_y는 원본 이미지에서 변화량이 적으면 1에 가깝고 변화량이 크면 0에 가까워지는 Weight입니다.

이 둘을 곱함으로서 외곽선의 Loss를 0에 가깝게 만드는것입니다.

결국 Loss는 외곽선 부분은 0이 되고 남은 나머지가 되는 것입니다.

## 4. 최종 정리

Edge Aware Smooth Loss는 외곽선은 계산에서 제외하는 Smooth Loss입니다.

이를 통해 컵이나 책상같은 물체가 갖고 있는 외곽선을 뭉개는 문제를 해결할 수 있습니다.

## 세부 튜닝

1. 정규화

```
mean_disp = disp.mean(dim=(1, 2, 3), keepdim=True)
disp = disp / (mean_disp + 1e-7)
```

disp의 평균으로 disp를 나눠주는 것입니다. 

모든 disp가 한쪽으로 쏠리는 현상을 방지할 수 있습니다.

2. 가중치 튜닝

```
weight_x = torch.exp(-img_dx * 10.0)
weight_y = torch.exp(-img_dy * 10.0)
```

여기에 있는 10.0 이건 제가 임의로 넣은 값입니다.

더 큰 값으로 바꾸면 ex)20, 50, 외곽선을 더 강하게 볼것입니다.

3. 특징을 이용한 외곽선 추출

머릿속으로만 생각한것입니다.

빛같은게 비춰버리면 같은 책상이어도 특정 부분은 하얀색이 되어버려 외곽선으로 인식합니다.

만약 특징을 이용한다면 책상은 모두 책상의 특징을 가질것이고 바닥은 바닥의 특징을 가질것입니다.

이를 이용하면 좋지 않은 환경에서도 외곽선을 추출할 수 있을것입니다.