# 03. Back Projection (역투영 / 재투영)

역투영(Back Projection)이란 2차원 좌표 $(u, v)$에 깊이(Depth, $Z$)를 넣어 3차원 좌표로 올리는 연산입니다. 

## 1. 역투영의 2단계 파이프라인

2D 픽셀을 전체 3D 월드 맵에 배치하기 위해서는 카메라 안쪽 세상(내부 파라미터)을 거쳐, 카메라 바깥쪽 세상(외부 파라미터)으로 나아가는 2단계 역산 과정을 거쳐야 합니다.

여기서 2D 픽셀은 2차원 좌표입니다. 이미지 상으로는 좌표보다 픽셀이 더 이해하기 좋아 픽셀로 진행했습니다.

### 단계 1: 2D 픽셀 $\rightarrow$ 3차원 카메라 좌표계 ($2D \rightarrow 3D_c$)
카메라 내부 파라미터 행렬 $K$의 역행렬 $K^{-1}$을 이미지 평면의 동차 좌표 $x = [u, v, 1]^T$에 곱해줍니다. 여기에 딥러닝 모델(Depth Network)이 예측한 실제 깊이 값 $d$를 스칼라로 곱해주면 카메라 기준의 3D 좌표 $X_c$를 얻을 수 있습니다.

```math
\begin{bmatrix}
X_c \\
Y_c \\
Z_c
\end{bmatrix}
=
d \cdot 
\begin{bmatrix}
\frac{1}{f_x} & 0 & -\frac{c_x}{f_x} \\
0 & \frac{1}{f_y} & -\frac{c_y}{f_y} \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}
=
\begin{bmatrix}
\frac{u - c_x}{f_x} \cdot d \\
\frac{v - c_y}{f_y} \cdot d \\
d
\end{bmatrix}
```
2D 픽셀 좌표에서 카메라 중심점$(c_x, c_y)$을 빼서 원점으로 맞춘 뒤, 초점 거리$(f)$비율만큼 나누고 실제 거리 $d$를 곱해 3차원 위치를 찾아내는 과정입니다.

### 단계 2: 카메라 좌표계 $\rightarrow$ 3차원 월드 좌표계 ($3D_c \rightarrow 3D_w$)
카메라 중심 기준의 3D 좌표에서 카메라 포즈 네트워크가 예측한 외부 파라미터 행렬 $(R, t)$을 역산해 월드 좌표계로 올립니다.
```math
X_w = R^T (X_c - t) = R^T X_c - R^T t
```
회전 행렬 $R$은 역행렬과 전치행렬이 같으므로 $R^T$를 곱하면 더 편하게 연산이 가능합니다.

## 2. 통합 역투영 방정식
위는 유클리드 좌표계를 이용한것입니다. 만약 동차 좌표계를 이용한다면 위의 단계 1과 단계 2를 결합하여, 단 한번의 행렬 곱셈으로 2D 픽셀을 3D 월드 좌표로 변환할 수 있습니다.
```math
\begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix}
=
\begin{bmatrix}
R_{11} & R_{21} & R_{31} & -(R^T t)_x \\
R_{12} & R_{22} & R_{32} & -(R^T t)_y \\
R_{13} & R_{23} & R_{33} & -(R^T t)_z \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
\frac{u - c_x}{f_x} \cdot d \\
\frac{v - c_y}{f_y} \cdot d \\
d \\
1
\end{bmatrix}
```

## 3. 최종 정리
* 투영 (Projection): 3D 공간 $(X, Y, Z)$ $\rarr$ 2D 화면 $(u, v)$
* 역투영 (Back Projection): 2D 화면 $(u, v)$ + 깊이 $(d)$ $\rarr$ 3D 공간 $(X, Y, Z)$

위의 공식을 통해 2차원의 픽셀을 3차원으로 올릴 수 있습니다.