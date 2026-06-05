# 04. Rigid Body Transformation (강체 변환)

강체 변환(Rigid Body Transformation)이란 3차원 공간상에서 위치(이동)과 방향(회전)만 바꾸는 기하학적 변환입니다.

여기서 따로 증명을 하지는 않겠습니다.

## 1. 강체 변환 2가지 구성 요소

강체 변환은 **회전**과 **이동** 두 가지로 이루어집니다.

### 회전

카메라가 원점을 기준으로 얼마나 고개를 돌렸는지(Roll, Pitch, Yaw)를 나타내는 $3 \times 3$ 행렬입니다.
```math
R = \begin{bmatrix} R_{11} & R_{12} & R_{13} \\ R_{21} & R_{22} & R_{23} \\ R_{31} & R_{32} & R_{33} \end{bmatrix}
```

회전 행렬은 물체의 순수한 방향만으로 돌리기 때문에, 수학적으로 직교 행렬이 됩니다. 

직교 행렬은 자신의 전치 행렬(Transpose)이 역행렬이 되는 특성을 갖고 있습니다.
```math
R^T = R^{-1} \implies R^T R = I
```
이는 역행렬 계산에 필요한 복잡한 나눗셈 연산 대신 행과 열을 뒤집는 전치(Transpose)를 통해 역방향 회전을 구할 수 있다는 것을 의미하며 이는 연산 속도 최적화로 이어집니다.

### 이동

카메라나 물체가 $X, Y, Z$축 방향으로 얼마나 미끄러지듯 이동했는지를 나타내는 3차원 벡터입니다.
```math
t = \begin{bmatrix} t_x \\ t_y \\ t_z \end{bmatrix}
```

## 2. 강체 변환 방정식

어떤 3차원 점 $X$를 회전$(R)$시키고 이동$(t)$시켜 새로운 점 $X'$로 만드는 식은 다음과 같습니다.

### 유클리드 공간의 표현
```math
\begin{bmatrix} X' \\ Y' \\ Z' \end{bmatrix}
=
\begin{bmatrix}
R_{11} & R_{12} & R_{13} \\
R_{21} & R_{22} & R_{23} \\
R_{31} & R_{32} & R_{33}
\end{bmatrix}
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
+
\begin{bmatrix} t_x \\ t_y \\ t_z \end{bmatrix}
\implies
X'=R \cdot X + t
```

### 동차 좌표계의 표현
동차 좌표계를 이용하면 회전과 이동을 묶어 $4 \times 4$ 단일 행렬$(T)$로 압축할 수 있습니다.
```math
\begin{bmatrix} X' \\ Y' \\ Z' \\ 1 \end{bmatrix}
=
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
=
\begin{bmatrix}
R_{11} & R_{12} & R_{13} & t_x \\
R_{21} & R_{22} & R_{23} & t_y \\
R_{31} & R_{32} & R_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
\implies X' = T \cdot X
```

## 3. 강체 변환의 역변환 (Inverse Transformation)

A위치에서 B위치로 카메라가 이동했다면, 반대로 B위치의 점들을 A위치의 카메라 시점으로 되돌리는 역변환$(T^{-1})$이 필요합니다.

단순히 $T$의 역행렬을 구하는 것이 아니라, 강체 변환의 성질$(R^{-1}=R^T)$을 이용하면 깔끔하게 연산할 수 있습니다.

$X' = R \cdot X + t$ 식을 원래의 $X$에 대해 정리해 보면
1. 양변에 $t$를 뺍니다: $X' - t = R \cdot X$
2. 양변에 $R^T$ (즉, $R^{-1}$)를 곱합니다: $R^T(X' - t) = X$
3. 전개합니다: $X = R^T X' - R^T t$

이 결과를 동차 좌표계 $4 \times 4$행렬식으로 다시 조립하면 다음과 같은 역변환 공식이 나옵니다.

```math
T^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix} = 
\begin{bmatrix}
R_{11} & R_{21} & R_{31} & -(R_{11}t_x + R_{21}t_y + R_{31}t_z) \\
R_{12} & R_{22} & R_{32} & -(R_{12}t_x + R_{22}t_y + R_{32}t_z) \\
R_{13} & R_{23} & R_{33} & -(R_{13}t_x + R_{23}t_y + R_{33}t_z) \\
0 & 0 & 0 & 1
\end{bmatrix}
```

이를 이용해 동차 좌표계의 식을 전개하면

```math
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
=
\begin{bmatrix}
R^T & -R^T t \\
0 & 1
\end{bmatrix}
\begin{bmatrix} X' \\ Y' \\ Z' \\ 1 \end{bmatrix}
=
\begin{bmatrix}
R_{11} & R_{21} & R_{31} & -(R_{11}t_x + R_{21}t_y + R_{31}t_z) \\
R_{12} & R_{22} & R_{32} & -(R_{12}t_x + R_{22}t_y + R_{32}t_z) \\
R_{13} & R_{23} & R_{33} & -(R_{13}t_x + R_{23}t_y + R_{33}t_z) \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} X' \\ Y' \\ Z' \\ 1 \end{bmatrix}
\implies X = T^{-1} \cdot X'
```

다음과 같이 이전으로 돌아갈 수 있습니다.

## 4. 최종 정리

강체 변환 공식을 이용하면 물체의 회전과 이동을 간단한 연산을 통해 수행할 수 있습니다.