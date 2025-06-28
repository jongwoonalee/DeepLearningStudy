# DA-MIL 논문 정리 및 MIL 공부 노트

> 작성일: 2025-06-28  
> 정리자: Jongwoon (Alyssa) Lee

---
# DA-MIL Framework Detailed Breakdown and Clarifications

## 1. Absolute Attention의 한계와 문제점

Absolute attention 방식은 patch들을 독립적으로 처리(i.i.d.)하여 각 patch가 얼마나 중요한지를 attention score로 판단합니다. 이 방식의 문제점은 다음과 같습니다:

- **쉬운 patch만 강조하는 경향**: 예를 들어, 매우 뚜렷한 암세포가 있는 patch는 높은 score를 받기 쉬운 반면, positive지만 모호하거나 주변 조직과 섞여 있는 hard positive는 간과될 가능성이 높습니다.
- **Patch 간 관계 무시**: 어떤 patch 하나만 보면 positive인지 판단하기 어려워도, 주변 context를 함께 보면 명확해지는 경우가 많습니다. 하지만 absolute attention은 이러한 spatial 관계를 반영하지 않습니다.

## 2. Gated Attention이란?

Gated Attention Mechanism(GAM)은 bag-level representation들을 통합할 때 사용되는 attention 방식입니다. 각 bag이 얼마나 중요한지를 평가하기 위해 tanh와 sigmoid의 조합을 사용해 gate를 형성합니다. 이는 복잡한 self-attention을 다시 적용하지 않고도 효과적으로 중요 bag을 선택할 수 있도록 도와줍니다.

## 3. Contrastive Loss에서 왜 zL과 zN 사이 거리를 조정하는가?

DA-MIL은 각 bag에 대해 두 가지 표현을 학습합니다: zL (전체 상태)와 zN (negative instance 기반). 이를 기반으로 contrastive learning을 수행하는 이유는 다음과 같습니다:

- **음성 WSI**: WSI 전체가 negative로만 구성되어 있어 zL ≈ zN 이 되어야 합니다. → 거리를 **최소화**
- **양성 WSI**: positive patch가 존재하므로 zL은 zN과 달라야 합니다. → 거리를 **최대화**하여 모델이 positive 특징을 구별할 수 있도록 유도

## 4. Learnable Token x₀의 위치와 역할

BERT에서 [CLS] token처럼, DA-MIL에서도 learnable한 token x₀를 sequence의 맨 앞에 삽입합니다. 이 토큰은 전체 bag의 정보를 요약하는 역할을 하며, attention 연산 후 이 토큰의 표현이 최종 representation (zL 또는 zN)으로 사용됩니다.

## 5. 이 아이디어는 새로웠는가?

DA-MIL은 기존 MIL 모델의 요소들을 재구성하여 실용적으로 높은 성능을 보이는 구조입니다. 다음은 구성 요소와 기존 연구와의 비교입니다:

- MIL 기본 구조는 CLAM 등에서 사용되었으며, ViT 기반 self-attention도 TransMIL, HIPT 등에서 사용되었습니다.
- 그러나 zL과 zN을 나누어 두 가지 표현을 학습하고, negative-only WSI로부터 순수 negative 표현을 학습하는 구조는 새롭습니다.
- 또한 bag 단위에서 contrastive loss를 적용하고, attention 기반 localization을 IHC와 비교하여 해석하는 방식도 기존 논문들보다 독창적입니다.
