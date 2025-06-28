# DA-MIL 논문 정리 및 MIL 공부 노트

> 작성일: 2025-06-28  
> 정리자: Jongwoon (Alyssa) Lee

---

## 📌 논문 개요

**제목**: Divide-and-Aggregate Multiple Instance Learning (DA-MIL)  
**목표**: H&E stained WSIs를 기반으로 ER/PR hormone receptor status를 MIL 방식으로 분류

---

## 1. 문제 정의 및 기존 방식 한계

### 1.1 MIL의 기본 아이디어
- WSI = 여러 patch로 나눈 "bag"들
- 최소 한 patch라도 positive면 bag 전체를 positive로 간주

### 1.2 Absolute Attention 방식의 한계
- 쉬운 patch만 강조 → hard positive 놓침
- patch 간 상관관계 무시 → false positive or missing region 유발

### 1.3 Self-Attention의 등장
- patch 간 관계를 고려하여 의미 있는 feature 추출 가능
- 그러나 계산량 O(n²) → WSI에서는 비효율적

---

## 2. DA-MIL 주요 구성 요약

| 항목 | 내용 |
|------|------|
| Divide | WSI → Patch → Bag 단위로 나누기 |
| Bag 표현 | zL (label-related), zN (negative-only) |
| Attention 1 | **Self-attention** (bag 내부 patch 간 관계 학습) |
| Attention 2 | **Gated attention** (bag 간 중요도 계산) |
| Contrastive 학습 | zL ↔ zN 사이 거리 조절 (음성: 가까이, 양성: 멀게) |

---

## 3. Gated Attention 정리

**공식**:
```
u_b = tanh(V1 · z_b) ⊙ σ(V2 · z_b)  
a_b = softmax(wᵀ · u_b)
```

- tanh: 내용 정보
- sigmoid: 중요도 게이트 역할
- element-wise 곱으로 중요한 정보만 강조

**장점**:
- bag-level attention이라 계산량 작음
- 중요한 bag만 골라내 WSI-level 표현 구성

---

## 4. Contrastive Loss 설명

| 경우 | Loss 목표 |
|------|-----------|
| 음성 WSI (y=0) | zL ≈ zN (→ 거리 작게) |
| 양성 WSI (y=1) | zL ⊥ zN (→ 거리 멀게) |

공식:
```
L₄(I,y)= 1/2B ∑ [ (1−y)·‖zL−zN‖² + y·max(0, δ−‖zL−zN‖)² ]
```

---

## 5. 통합 Loss 구성

| Loss 이름 | 설명 |
|-----------|------|
| L₁ | WSI-level classification (zW 기준) |
| L₂ | zL을 활용한 bag-level classification |
| L₃ | zN을 활용한 bag-level negative classification (y=0일 때만) |
| L₄ | zL vs zN contrastive loss |

---

## 6. 교수/연구자들의 논문 공부법 정리

| 전략 | 설명 |
|------|------|
| 리서치 다이어리 | 논문마다 핵심 한 줄 요약 + 그림 캡처 |
| 논문 분류 정리 | ‘attention 방식’ 등 카테고리화 |
| 토론 세션 주도 | Lab 세미나에서 논문 발표/비교 |
| 내 연구와 연결 | 논문을 읽기보다 ‘내 문제’로 역추적 |

---

## 7. 내 학습 계획

```yaml
- [x] DA-MIL 논문 완독 및 구조 분석
- [ ] 유사 논문 3편 비교표 작성
- [ ] attention 방식별 정리 슬라이드 제작
- [ ] 논문 기반 코드 복기 실습
- [ ] 내 논문에 응용할 포인트 정리
```

---

## ❤️ 앞으로 이렇게 공부할 것

- "읽고 흘리는" 논문은 없다.
- 논문은 **도구**고, **내 언어로 정리할 때** 진짜가 된다.

---

