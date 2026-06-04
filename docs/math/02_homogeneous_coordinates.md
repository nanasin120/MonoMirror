# Homogeneous Coordinates

동차 좌표계(Homogeneous Coordinates)는 기존의 3차원 점 $(X, Y, Z)$뒤에 하나의 차원을 더하는 것입니다.

## 왜 차원을 하나 더 늘리는가
유클리드 공간에서의 3차원 점 $(X, Y, Z)$로는 카메라 회전($R$)과 이동($t$)을 하나의 선형 연산으로 처리할 수 없습니다.

* **유클리드 변환:**
```math
\begin{bmatrix}
X_c \\
Y_c \\
Z_c
\end{bmatrix}
=
\begin{bmatrix}
R_{11} & R_{12} & R_{13} \\
R_{21} & R_{22} & R_{23} \\
R_{31} & R_{32} & R_{33}
\end{bmatrix}
\begin{bmatrix}
X_w \\
Y_w \\
Z_w
\end{bmatrix}
+
\begin{bmatrix}
t_x \\
t_y \\
t_z
\end{bmatrix}
\implies
X_c = R \cdot X_w + t
```

위의 식에서 $$X_w$$는 유클리드 공간의 3차원 점임을 알립니다.

위의 3차원 점에 만약 차원을 하나 늘려 마지막 성분에 `1`을 추가한 **동차 좌표계**를 도입하면, 더하기 연산이었던 $$+ t$$또한 하나의 선형 연산으로 통합할 수 있습니다.

* **동차 좌표계 변환:**
```math
\begin{bmatrix}
X_c \\
Y_c \\
Z_c \\
1
\end{bmatrix}
=
\begin{bmatrix}
R_{11} & R_{12} & R_{13} & t_x \\
R_{21} & R_{22} & R_{23} & t_y \\
R_{31} & R_{32} & R_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix}
\implies
X_c = T \cdot X_w
```

## 왜 동차 좌표계를 사용하는가

식이 여러개 있다고 생각하면 편합니다. 유클리드 좌표계의 경우에는
```math
X_c = R_3(R_2(R_1X_w+t_1)+t_2)+t_3
```
다음과 같이 괄호가 생겨 식이 지저분해집니다. 하지만 동차 좌표계의 경우에는
```math
X_c = T_3 \cdot T_2 \cdot T_1 \cdot X_w
```
다음과 같이 괄호가 생기지 않고 순서 또한 바꿀 수 있습니다. 이를 이용하면
```math
X_c = T_{total} \cdot X_w
```
모든 $$T$$를 미리 계산하는 최적화를 진행할수 도 있습니다. 

즉 점이 아무리 많아져도 **단 한번의 행렬 곱셈**을 통해 변환을 끝낼 수 있습니다.

## 최종 정리

동차 좌표계를 이용하면 카메라 회전과 이동을 **단 한번의 행렬 곱셈**을 통해 적용시킬 수 있고 이는 전체적인 속도 향상에 큰 영향을 끼칩니다.
