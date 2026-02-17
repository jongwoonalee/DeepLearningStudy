좋아요. Cross Entropy(Cross-Entropy Loss) 식은 **클래스 개수**에 따라 형태가 달라집니다.  
기본 정의부터 단계적으로 정리해드릴게요.

---

## 1. **일반적인 정의 (정보이론 관점)**

두 확률분포 $p$ (정답 분포)와 $q$ (모델의 예측 분포)가 있을 때,  
**Cross Entropy**는 다음과 같이 정의됩니다:

$$  
H(p, q) = -\sum_{i} p_i \log q_i  
$$

- $p_i$: 실제 정답의 확률 (정답 클래스의 one-hot 값)
    
- $q_i$: 모델이 예측한 클래스 $i$의 확률 (softmax 출력 등)
    

---

## 2. **다중분류(Multi-class classification)**

정답이 one-hot (예: $p = [0, 1, 0]$)이라면,  
정답 클래스 하나만 남아서 다음처럼 단순화됩니다:

$$  
L_{CE} = -\log q_{y}  
$$

또는, 배치 전체 평균으로 쓰면:

$$  
L_{CE} = -\frac{1}{N}\sum_{n=1}^{N}\sum_{c=1}^{C} y_{n,c} \log q_{n,c}  
$$

여기서

- $N$: 샘플 수
    
- $C$: 클래스 수
    
- $y_{n,c}$: $n$번째 샘플의 실제 정답 (one-hot)
    
- $q_{n,c}$: $n$번째 샘플의 softmax 확률
    

---

## 3. **이진 분류(Binary classification)**

이건 다중분류의 특수한 경우 ($C=2$).  
이때 식은 다음처럼 단순화됩니다:

$$  
L_{BCE} = -[y\log q + (1-y)\log(1-q)]  
$$

(바로 우리가 흔히 쓰는 **Binary Cross Entropy**)

---

## 4. **다중라벨(Multi-label classification)**

클래스들이 서로 독립이라면, 각 클래스마다 BCE를 계산해서 평균을 냅니다:

$$  
L = -\frac{1}{C}\sum_{c=1}^{C}[y_c\log q_c + (1-y_c)\log(1-q_c)]  
$$

---

## 5. **요약**

|유형|출력|수식|활성함수|
|---|---|---|---|
|Binary|1 sigmoid|$- [y\log q + (1-y)\log(1-q)]$|sigmoid|
|Multi-class|C softmax|$-\sum_{c} y_c \log q_c$|softmax|
|Multi-label|C sigmoid|$-\sum_{c}[y_c\log q_c + (1-y_c)\log(1-q_c)]$|sigmoid|

---

원하신다면,  
$y=[0,0,1]$일 때 softmax 출력 $q=[0.1, 0.2, 0.7]$을 넣어서  
CE 값이 실제로 얼마가 나오는지도 계산 예시로 보여드릴 수 있습니다. 해드릴까요?