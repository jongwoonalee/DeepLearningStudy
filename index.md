---
layout: default
title: DA-MIL Study Notes
---

<script type="text/javascript"
  async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# 🧠 DA-MIL Study Notes (with LaTeX Support)

This page contains my technical summary of the DA-MIL paper for hormone receptor classification in H&E whole slide images.

---

## 🔍 Contrastive Loss

Let \( y \in \{0, 1\} \) be the label (0 = negative WSI, 1 = positive WSI).

The contrastive loss is defined as:

\[
\mathcal{L}_4(I, y) = \frac{1}{2B} \sum_b \left[ (1 - y) \cdot \|z_L^b - z_N^b\|^2 + y \cdot \max(0, \delta - \|z_L^b - z_N^b\|)^2 \right]
\]

---

## 🧩 Self-Attention-Based Bag Representation

Each bag includes a learnable token \( x_0 \). The self-attention outputs a weighted sum:

\[
\tilde{z}_L = \sum_{j=0}^m \frac{\exp(q_0^L \cdot k_j^L / \sqrt{d})}{\sum_k \exp(q_0^L \cdot k_k^L / \sqrt{d})} v_j^L
\]

Final representation:

\[
z_L = \text{MLP}(\text{LN}(\tilde{z}_L)) + \tilde{z}_L
\]

---

## 🧲 Gated Attention Aggregation

To compute WSI-level embedding:

\[
u_b = \tanh(V_1 z_L^b) \cdot \sigma(V_2 z_L^b)
\quad
a_b = \frac{\exp(w^\top u_b)}{\sum_k \exp(w^\top u_k)}
\quad
z_W = \sum_b a_b z_L^b
\]

---

## ✨ Author

👩🏻‍⚕️ MD/PhD  
🎓 PhD Candidate, Brain & Cognitive Engineering  
💻 MSCS Incoming @ University of Rochester

GitHub: [@jongwoonalee](https://github.com/jongwoonalee)
