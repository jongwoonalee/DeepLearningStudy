---
layout: default
title: DA-MIL Study Notes
---

# 🧠 Deep Learning Study Notes

Welcome to my personal deep learning study repository!  
This space is dedicated to collecting and organizing technical notes from cutting-edge research, especially in **digital pathology**, **Multiple Instance Learning (MIL)**, and **attention-based models**.

---

## 📘 Current Focus: DA-MIL (Divide-and-Aggregate MIL)

### 🔍 Paper Overview

**Title**: Divide-and-Aggregate Multiple Instance Learning for Hormone Receptor Status Classification  
**Context**: Predict Estrogen/Progesterone receptor (ER/PR) status from H&E whole slide images without IHC  
**Core Idea**:  
- Divide gigapixel WSI into small patches  
- Randomly group into bags: \( B_b = \{x^b_1, \dots, x^b_m\} \)  
- Compute dual bag-level representations:
  - \( z_L \): label-related
  - \( z_N \): negative-only  
- Use contrastive learning to differentiate them.

---

## 🧠 Key Concepts

### 🧩 Bag Representation with Self-Attention

Each bag \( B_b \) is processed with a self-attention block. A learnable token \( x_0 \) (like [CLS]) is prepended:

\[
\tilde{z}_L = \sum_{j=0}^m \frac{\exp(q_0^L \cdot k_j^L / \sqrt{d})}{\sum_k \exp(q_0^L \cdot k_k^L / \sqrt{d})} v_j^L
\]

Final bag-level representation:

\[
z_L = \text{MLP}(\text{LN}(\tilde{z}_L)) + \tilde{z}_L
\]

Same for \( z_N \), with separate self-attention parameters.

---

### 🎯 Contrastive Loss Design

Let \( y \in \{0, 1\} \) be the WSI-level label (0: negative, 1: positive).

We apply an adaptive contrastive loss between \( z_L \) and \( z_N \):

\[
\mathcal{L}_4(I, y) = \frac{1}{2B} \sum_b \left[(1 - y) \cdot \|z_L^b - z_N^b\|^2 + y \cdot \max(0, \delta - \|z_L^b - z_N^b\|)^2 \right]
\]

- \( y = 0 \): force \( z_L \approx z_N \)
- \( y = 1 \): force \( z_L \) and \( z_N \) to differ

---

### 📚 Loss Components

| Loss | Description |
|------|-------------|
| \( \mathcal{L}_1 \) | CE loss for WSI-level prediction using \( z_W \) |
| \( \mathcal{L}_2 \) | CE loss on \( z_L \) at bag-level |
| \( \mathcal{L}_3 \) | CE loss on \( z_N \) when \( y = 0 \) |
| \( \mathcal{L}_4 \) | Contrastive loss between \( z_L \) and \( z_N \) |

Total loss:

\[
\mathcal{L}_\text{total} = \mathcal{L}_1 + \mathcal{L}_2 + \mathcal{L}_3 + \mathcal{L}_4
\]

---

## 🧲 Gated Attention Aggregation

To compute WSI-level representation from bag-level features \( z_L^b \):

\[
u_b = \tanh(V_1 z_L^b) \cdot \sigma(V_2 z_L^b)
\quad
a_b = \frac{\exp(w^\top u_b)}{\sum_k \exp(w^\top u_k)}
\quad
z_W = \sum_b a_b z_L^b
\]

This gives a learnable soft selection of important bags.

---

## 🚧 To Do

- [x] DA-MIL 논문 요약 정리
- [ ] PyTorch 코드 복기
- [ ] DA-MIL의 contrastive 구조를 내 연구에 적용
- [ ] 다른 MIL 논문들과 구조 비교 정리

---

## ✨ About Me

👩🏻‍⚕️ MD, PhD / Computational Pathologist  
🎓 PhD in Brain & Cognitive Engineering  
💻 Incoming MSCS @ University of Rochester  
🧠 Interest: AI for Pathology, WSI, MIL, Vision Transformers

GitHub: [@jongwoonalee](https://github.com/jongwoonalee)

