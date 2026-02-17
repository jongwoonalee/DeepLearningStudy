# Convolution Acceleration

Convolution acceleration은 CNN에서 합성곱 연산의 연산량(FLOPs)과 메모리 접근 비용을 줄여 속도를 향상시키는 기법들을 의미했습니다. 기본 2D convolution의 계산 복잡도부터 정리하면 다음과 같았습니다.

## 1. Baseline: Standard Convolution

입력

$$X \in \mathbb{R}^{H \times W \times C_{in}}$$

필터

$$W \in \mathbb{R}^{K \times K \times C_{in} \times C_{out}}$$

출력

$$Y \in \mathbb{R}^{H \times W \times C_{out}}$$

총 연산량:

$$\text{FLOPs} = H \cdot W \cdot C_{out} \cdot (K^2 \cdot C_{in})$$

### 예시

- $H=W=224$
- $K=3$
- $C_{in}=64$
- $C_{out}=128$

$$224^2 \cdot 128 \cdot (3^2 \cdot 64) = 50,176 \cdot 128 \cdot 576 \approx 3.7 \times 10^9 \text{ ops}$$

## 2. 주요 가속 기법

### (1) Depthwise Separable Convolution (MobileNet)

Standard convolution 분해:

1. Depthwise conv: $K \times K \times 1 \times C_{in}$
2. Pointwise conv (1×1): $1 \times 1 \times C_{in} \times C_{out}$

연산량:

$$HW(K^2 C_{in} + C_{in} C_{out})$$

비율 비교:

$$\frac{K^2 C_{in} + C_{in} C_{out}}{K^2 C_{in} C_{out}}$$

$K=3$일 때:

$$\frac{9C_{in} + C_{in}C_{out}}{9C_{in}C_{out}} \approx \frac{1}{9}$$

→ 약 8~9배 연산 감소

### (2) Group Convolution (ResNeXt, AlexNet)

채널을 G개 그룹으로 분할:

$$\text{FLOPs} = \frac{1}{G} \cdot HW K^2 C_{in} C_{out}$$

- $G=1$: standard conv
- $G=C_{in}$: depthwise conv

### (3) 1×1 Convolution (Bottleneck)

차원 축소:

$$C_{in} \rightarrow C_{mid} \rightarrow C_{out}$$

예시 (ResNet bottleneck):

$$1\times1 \ (256 \rightarrow 64)$$ $$3\times3 \ (64 \rightarrow 64)$$ $$1\times1 \ (64 \rightarrow 256)$$

연산량 감소 비율:

$$\frac{64}{256} = 0.25$$

→ 3×3 conv 비용 75% 감소

### (4) FFT-based Convolution

Convolution theorem:

$$x * w = \mathcal{F}^{-1}(\mathcal{F}(x)\cdot\mathcal{F}(w))$$

복잡도:

- Spatial: $O(K^2)$
- FFT: $O(N\log N)$

→ 큰 kernel (≥7×7) 에서만 이득  
→ 실제 CNN에서는 메모리 overhead로 잘 안 씀

### (5) Winograd Convolution

작은 kernel (3×3) 최적화:

$$F(m,r)$$

예:

$$F(2,3) \Rightarrow 2\times2 \text{ output}, 3\times3 \text{ kernel}$$

연산량 감소:

- multiplications 약 2~4배 감소
- cuDNN 내부에서 자동 적용

### (6) Im2col + GEMM

Conv → matrix multiply로 변환

$$\text{Conv} \rightarrow \text{GEMM}$$

|장점|단점|
|---|---|
|GPU 최적화된 BLAS 사용 가능|메모리 사용량 증가|

### (7) Low-rank Factorization

필터 분해:

$$W \approx UV$$

예:

$$K \times K \rightarrow K \times 1 + 1 \times K$$

### (8) Pruning / Sparsity

- weight pruning
- channel pruning
- structured sparsity

실제 연산량:

$$\text{Effective FLOPs} = (1-s) \cdot \text{FLOPs}$$

$s$: sparsity ratio

### (9) Quantization

FP32 → INT8

- 이론적 speedup: 4×
- 실제: 1.5~3×

## 3. 하드웨어 레벨 가속

### GPU

- Tensor Core (FP16, BF16)
- cuDNN autotuning

### NPU / TPU

- systolic array
- data reuse 최적화

## 4. 실전에서 가장 많이 쓰는 조합

|레벨|기법|
|---|---|
|구조|depthwise, group conv|
|블록|bottleneck|
|라이브러리|cuDNN Winograd|
|모델|pruning|
|추론|INT8 quantization|

## 5. 요약

|방법|연산 감소율|
|---|---|
|Depthwise|~8–9×|
|Bottleneck|~4×|
|Group conv (G=4)|4×|
|Winograd|2–4×|
|Quantization|2–4×|

## 핵심 문장

> 가장 효과적인 convolution acceleration은 구조적 설계(depthwise, bottleneck) + 커널 레벨 최적화(Winograd) + 추론 최적화(INT8) 조합이었다.

==========
# Convolution Acceleration – **직관 중심 상세 설명**

아래는 _"왜 빨라지는지"_ 를 **수식 → 직관 → 그림처럼** 이해할 수 있게 정리했습니다.

---

# 1. Standard Convolution이 왜 느린가

## 계산 구조

출력 픽셀 하나 만들 때:

$$K \times K \times C_{in}$$

만큼 곱해야 합니다.

전체 연산:

$$H \cdot W \cdot C_{out} \cdot (K^2 \cdot C_{in})$$

## 직관

- **공간(K×K)**
- **채널(C_in)**
- **출력 채널(C_out)**

👉 3중으로 폭증  
👉 채널 수 늘어나면 **기하급수적으로 느려짐**

---

# 2. Depthwise Separable Convolution

## 구조

### (1) Depthwise

채널별로 **각자 따로 필터 적용**

$$K \times K \times 1$$

### (2) Pointwise (1×1)

채널 섞기

$$1 \times 1 \times C_{in} \times C_{out}$$

---

## 왜 빠른가?

### Standard

$$K^2 \cdot C_{in} \cdot C_{out}$$

### Depthwise

$$K^2 \cdot C_{in}$$

### Pointwise

$$C_{in} \cdot C_{out}$$

총합

$$K^2 C_{in} + C_{in} C_{out}$$

---

## 숫자로 비교

|항목|값|
|---|---|
|$K$|3|
|$C_{in}$|64|
|$C_{out}$|128|

### Standard

$$9 \times 64 \times 128 = 73,728$$

### Depthwise

$$9 \times 64 = 576$$

### Pointwise

$$64 \times 128 = 8192$$

총

$$8768$$

### 비율

$$\frac{8768}{73728} \approx 0.119$$

→ **약 8.4배 빠름**

---

## 직관

> "공간 패턴 찾기"와  
> "채널 섞기"를 **분리**

✔ 계산 분리 → 대폭 감소  
✔ MobileNet 핵심

---

# 3. Group Convolution

## 구조

채널을 **G개 그룹으로 분할**

예:

- 입력 64채널
- G=4  
    → 그룹당 16채널

각 그룹끼리만 convolution

---

## 연산량

$$\frac{1}{G} HW K^2 C_{in} C_{out}$$

---

## 직관

> "전체 채널 다 섞지 말고  
> **부분적으로만 섞자**"

---

## 극단

|G|의미|
|---|---|
|1|standard|
|$C_{in}$|depthwise|

---

# 4. 1×1 Convolution (Bottleneck)

## 목적

**채널 수 줄이기**

---

## 구조 (ResNet)

```
256 → 64 → 64 → 256
 1×1   3×3   1×1
```

---

## 연산 감소

원래

$$3^2 \times 256 \times 256$$

축소 후

$$3^2 \times 64 \times 64$$

비율

$$\frac{64}{256} = 0.25$$

→ **75% 연산 감소**

---

## 직관

> "비싼 3×3 연산 전에  
> 채널 다이어트"

---

# 5. FFT Convolution

## 원리

$$x * w = \mathcal{F}^{-1}(\mathcal{F}(x) \cdot \mathcal{F}(w))$$

- 공간 → 주파수
- 곱셈
- 역변환

---

## 언제 빠른가

|커널|효과|
|---|---|
|3×3|❌|
|7×7|△|
|15×15|✔|

---

## 왜 CNN에서 잘 안씀?

- FFT 변환 비용 큼
- 메모리 많이 먹음
- 작은 커널에선 손해

---

# 6. Winograd Convolution

## 아이디어

> **곱셈 줄이고 덧셈 늘리기**

GPU에서  
→ 곱셈이 비싸서 이득

---

## 예

$$F(2,3)$$

- 3×3 kernel
- 2×2 output

### 효과

- 곱셈: **4→2개**
- 덧셈 증가

---

## 실제

- cuDNN이 자동 적용
- 사용자 직접 구현 X

---

# 7. Im2col + GEMM

## 방법

Conv → Matrix multiply

---

## 장점

- GPU 최적화 BLAS 사용
- 매우 빠름

## 단점

- 메모리 폭증
- 캐시 미스 증가

---

## 직관

> "Conv를  
> **행렬곱 문제로 바꿈**"

---

# 8. Low-rank Factorization

## 필터 분해

$$K \times K \Rightarrow K \times 1 + 1 \times K$$

---

## 예

3×3 → 3×1 → 1×3

---

## 직관

> "2D 패턴을  
> **가로+세로**로 분리"

---

# 9. Pruning

## 방식

- weight 삭제
- channel 삭제

---

## 연산량

$$(1-s) \cdot \text{FLOPs}$$

|sparsity|남은 연산|
|---|---|
|50%|50%|
|80%|20%|

---

## 구조적 pruning 중요

- channel pruning → 실제 속도 개선
- weight pruning → 이론적만

---

# 10. Quantization

## 타입

|형식|bit|
|---|---|
|FP32|32|
|FP16|16|
|INT8|8|

---

## 효과

|타입|속도|
|---|---|
|FP16|2×|
|INT8|2~4×|

---

## 직관

> "숫자 정밀도 줄이면  
> 계산 가벼워짐"

---

# 11. 하드웨어 가속

## GPU

- Tensor Core
- FP16/BF16

## TPU

- systolic array
- 데이터 재사용

---

# 실제 조합 예

**MobileNet**

|단계|기법|
|---|---|
|구조|depthwise|
|채널|bottleneck|
|라이브러리|Winograd|
|추론|INT8|

---

# 핵심 요약

|방법|핵심 아이디어|
|---|---|
|Depthwise|공간/채널 분리|
|Group|부분 연결|
|Bottleneck|채널 축소|
|Winograd|곱셈 줄이기|
|FFT|큰 커널용|
|Im2col|행렬곱 변환|
|Pruning|필요없는 제거|
|Quant|정밀도 감소|

---

# 한 줄 요약

> **"비싼 연산 전에 차원 줄이고, 불필요한 계산을 구조적으로 제거하는 것이 핵심"**

---

## 다음 단계 가능

원하시면:

1. **PyTorch 코드 비교**
2. **실제 속도 측정 실험**
3. **논문별 구조 비교**
4. **MobileNet 내부 완전 해부**

어디까지 볼지 말씀 주세요.