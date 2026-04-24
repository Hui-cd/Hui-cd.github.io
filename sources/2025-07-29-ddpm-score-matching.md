---
layout: post
title: "DDPM 与 Score Matching 的统一推导与『为什么要这样做』"
date: 2025-07-29
tags: [Diffusion, DDPM, Score Matching, SDE, EDM, VP, VE]
mathjax: true
---

> 这是一篇把 **DDPM**（离散扩散）与 **Score-based**（连续 SDE / DSM）贯穿起来的文章。目标是：**推导清楚**，并且**讲明白为什么要这样做**。全文仅包含数学与解释，不含程序代码与样例实现。

---

## TL;DR（三句话）

1. **为什么「加噪—去噪」**：直接最大似然需要对归一化常数求解，困难；**Score Matching** 只学习 $\nabla \log p$，绕开归一化；**DDPM** 用前向加噪把数据连到高斯端，使反向去噪可学习，并给出 ELBO（似然下界）。  
2. **为什么用 SDE/Itô**：把 DDPM（离散）与 Score-based（连续）统一到**可分析**的框架；得到 $q(x_t|x_0)$ 的**高斯闭式**与**重参数化**；用 Itô 引理推导**期望/协方差**与 **Fokker–Planck**。  
3. **为什么 VP/VE**：**VP-SDE（DDPM 连续极限）**让均值衰减至 0、末端吻合高斯先验；**VE-SDE（NCSN/EDM 路线）**让方差增大、score 学习直观，配合连续噪声与高阶采样器质量/效率俱佳。

---

## 0. 动机：我们到底在解决什么难点？

- **归一化常数难题**：能量模型/复杂密度的配分函数难算。  
  **Score Matching** 只拟合 $s_\theta(x)\approx \nabla_x\log p(x)$，完全绕开归一化常数。  
- **生成需要“可逆路径”**：仅有 score 还需一条**从高斯回到数据**的动力学路径。  
  **DDPM/Score-based** 给出两步法：  
  ① 前向把数据连续（或离散）推向高斯；② 反向用学到的 score 把噪声“走回”数据。  
- **统一的微分方程视角**：用 **SDE**（及其反向 SDE、概率流 ODE）统一建模，使可分析、可推导、可控制。

---

## 1) 统一设定与两类前向过程（VP / VE）

随机过程 $X_t\in\mathbb{R}^n$，时间 $t\in[0,1]$。前向加噪把 $X_0\sim p_{\text{data}}$ 推向近似高斯的 $X_1$；反向过程从高斯端还原到数据。

**SDE 统一式（$W_t$ 为标准布朗运动）**
$$
dX_t = f(X_t,t)\,dt + G(X_t,t)\,dW_t.
$$

两条经典族：

- **VP-SDE（Variance Preserving，DDPM 连续极限）**
  $$
  dX_t = -\tfrac{1}{2}\beta(t) X_t\,dt + \sqrt{\beta(t)}\,dW_t,
  $$
  $$
  X_t\mid X_0=x_0 \sim \mathcal{N}\!\big(\alpha(t)\,x_0,\ (1-\alpha^2(t))I\big),\qquad
  \alpha(t)=\exp\!\Big(-\tfrac{1}{2}\int_0^t \beta(u)\,du\Big).
  $$

- **VE-SDE（Variance Exploding，NCSN/EDM 路线）**
  $$
  dX_t = \sqrt{\tfrac{d}{dt}\sigma^2(t)}\,dW_t,\qquad
  X_t\mid X_0=x_0 \sim \mathcal{N}\!\big(x_0,\ \sigma^2(t) I\big).
  $$

**重参数化（训练/推导的支点）**  
VP：$x_t=\alpha(t)x_0+\sigma(t)\epsilon,\ \sigma^2(t)=1-\alpha^2(t)$；  
VE：$x_t=x_0+\sigma(t)\epsilon$；$\ \epsilon\sim\mathcal{N}(0,I)$。  
> **为什么重要**：有了可微的高斯闭式，就能稳定、统一地**采样 $x_t$**、构造**损失**、并且在推断时**反解 $x_0$**。

---

## 2) Itô 微积分最小工具箱（以及为什么 $(dW)^2=dt$）

### 2.1 乘法表（保留到 $\mathcal O(dt)$）
$$
dt\cdot dt=0,\qquad dt\cdot dW=0,\qquad dW\cdot dW=dt,\qquad dW_i\cdot dW_j=\delta_{ij}\,dt.
$$
**直觉**：$dW=\mathcal{O}(\sqrt{dt})$，所以 $(dW)^2=\mathcal{O}(dt)$ 不可忽略；而 $dt\cdot dW=\mathcal{O}(dt^{3/2})$、$dt^2=\mathcal{O}(dt^2)$ 可忽略。  
也可由 $dW=\sqrt{dt}\,\varepsilon$（$\varepsilon\sim\mathcal{N}(0,1)$）得 $\mathbb{E}[(dW)^2]=dt$。

### 2.2 多维 Itô 引理（标量函数 $\phi$）
$$
\begin{aligned}
d\phi(X_t,t)
&= \partial_t\phi\,dt + \nabla_x\phi^\top f\,dt \\
&\quad + \frac{1}{2}\,\mathrm{Tr}\!\left[(GG^\top)\,\nabla_x^2\phi\right]\,dt
+ \nabla_x\phi^\top G\,dW_t
\end{aligned}
$$

> **为何关键**：它是“随机版链式法则”，把 SDE 中的噪声如何影响函数量（均值/方差/自由能等）说清楚，支撑后续所有统计与密度演化推导。

---

## 3) 从 SDE 到统计量：期望与协方差的演化（为什么要推这个）

推导这些量的目的：  
**(i)** 理解前向扩散的“信噪比”如何随时间变化（指导调度与加权）；  
**(ii)** 校验模型实现与数值稳定性；  
**(iii)** 在线性情形获得闭式（Kalman–Bucy），便于分析。

### 3.1 期望（均值）
取 $\phi(X_t,t)=X_{t,u}$ 得
$$
\frac{d}{dt}\mathbb{E}[X_{t,u}] = \mathbb{E}[f_u(X_t,t)]
\quad\Longrightarrow\quad
\frac{dm}{dt}=\mathbb{E}[f(X_t,t)],\qquad m(t)=\mathbb{E}[X_t].
$$

### 3.2 二阶矩与协方差
令 $S_t=X_tX_t^\top$，用乘积法则
$$
d(XX^\top)=X\,dX^\top + dX\,X^\top + dX\,dX^\top,
$$
代入 $dX=f\,dt+G\,dW$ 并取期望
$$
\frac{d}{dt}\mathbb{E}[XX^\top]=\mathbb{E}[X f^\top + f X^\top] + \mathbb{E}[GG^\top].
$$
协方差 $P=\mathbb{E}[(X-m)(X-m)^\top]=\mathbb{E}[XX^\top]-mm^\top$：
$$
\frac{dP}{dt}
= \mathbb{E}[(X-m)f^\top]+\mathbb{E}[f(X-m)^\top]+\mathbb{E}[GG^\top].
$$

### 3.3 线性高斯（闭式）
若 $f=A(t)X+b(t)$，$G=Q^{1/2}(t)$：
$$
\frac{dm}{dt}=A m+b,\qquad \frac{dP}{dt}=A P + P A^\top + Q.
$$
> **为什么有用**：它告诉我们在“线性化近似”下噪声如何被系统放大/衰减，有助于选择 $\beta/\sigma$ 调度与损失加权。

---

## 4) 从前向 SDE 到条件核 $q(x_t|x_0)$（以及这一步的意义）

**意义**：  
- 训练要从 $x_0$ 采样 $x_t$（作为输入）；  
- DSM 需要可解的真局部 score $\nabla\log q(x_t|x_0)$；  
- 推断/采样要能从 $x_t$ **反解 $x_0$** 或构造反向动力学。

### 4.1 VP-SDE（DDPM 连续极限）
线性 OU 过程解：
$$
X_t=\alpha(t)X_0+\int_0^t \alpha(t)\alpha(s)^{-1}\sqrt{\beta(s)}\,dW_s
\sim \mathcal{N}\!\big(\alpha(t)X_0,\ (1-\alpha^2(t))I\big).
$$

### 4.2 VE-SDE
$$
X_t=X_0+\int_0^t \sqrt{\tfrac{d}{ds}\sigma^2(s)}\,dW_s
\sim \mathcal{N}\!\big(X_0,\ \sigma^2(t)I\big).
$$

因此得到**重参数化**：
- VP：$x_t=\alpha(t)x_0+\sigma(t)\epsilon$；  
- VE：$x_t=x_0+\sigma(t)\epsilon$；$\ \epsilon\sim\mathcal{N}(0,I)$。

---

## 5) 为什么选 VP / VE（不是别的）？

- **解析监督**：$q(x_t|x_0)$ 高斯 ⇒ 真局部 score $\nabla_{x_t}\log q(x_t|x_0)$ **闭式可写**，DSM 可用。  
- **重参数稳定**：$x_t=\mu(t)x_0+\sigma(t)\epsilon$ ⇒ 训练/推断可微、尺度可控。  
- **先验衔接自然**：VP 让均值衰减到 0（与末端高斯先验一致）；VE 让方差增大（高噪端近似高斯）。  
- **工程成熟**：已有良好的 $\beta/\sigma$ 调度、损失加权（SNR/σ）、高阶采样器（Heun 等）。

---

## 6) 反向动力学：Reverse SDE 与 Probability Flow ODE（为什么能“生成”）

给定前向 $dX_t=f\,dt+g\,dW_t$ 与边缘密度 $p_t$，有

**反向 SDE**
$$
dX_t=\big[f(X_t,t)-g^2(t)\,\nabla_x\log p_t(X_t)\big]\,dt+g(t)\,d\bar{W}_t,\qquad t:1\to 0.
$$

**概率流 ODE（确定性）**
$$
\frac{dX_t}{dt}=f(X_t,t)-\tfrac{1}{2}g^2(t)\,\nabla_x\log p_t(X_t).
$$

> **为什么成立**：这来自 Fokker–Planck 与时间反演理论；**只要我们学到 score**（$\nabla\log p_t$ 的近似），就能用 SDE/ODE 把高斯端“走回”数据端。

---

## 7) 训练目标：DSM 与 DDPM（以及为什么这样做）

### 7.1 Denoising Score Matching（DSM）
高斯加噪核 $q_\sigma(\tilde x|x)=\mathcal{N}(\tilde x;x,\sigma^2 I)$ 的局部 score 为
$$
\nabla_{\tilde x}\log q_\sigma(\tilde x|x)= -\frac{\tilde x-x}{\sigma^2}.
$$
连续噪声 DSM 损失
$$
\mathcal{L}_{\mathrm{DSM}}
=\mathbb{E}_{x,\epsilon,\sigma}\Big[\big\| s_\theta(\tilde x,\sigma)+\tfrac{\epsilon}{\sigma}\big\|^2\Big],\qquad
\tilde x=x+\sigma\epsilon.
$$
> **为什么合理**：在 $x$ 条件期望下，上式是**边缘 score** 的一致监督；因此学到的 $s_\theta$ 可作为反向动力学的“方向场”。

### 7.2 DDPM：从 ELBO 到噪声回归 MSE
离散马尔可夫核：
$$
q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I),\qquad
q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,\ (1-\bar{\alpha}_t)I).
$$
变分下界（ELBO）分解后，最优反向核均值与**预测噪声 $\epsilon_\theta$** 线性相关，得到等价的 MSE：
$$
\mathcal{L}_{\epsilon}=\mathbb{E}_{t,x_0,\epsilon}\big[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\big],\qquad
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon.
$$
统一换算（便于与 DSM 对齐）：
$$
s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sigma(t)}.
$$

> **为什么选噪声回归**：尺度稳定、与 ELBO 一致（能评估似然），且推断时可显式反解 $\hat{x}_0$，便于控制与诊断。

---

## 8) 从 $x_t$ 恢复 $x_0$（反解为何重要）

- **VP**：$x_t=\alpha x_0+\sigma\epsilon \ \Rightarrow\ 
\hat x_0=\dfrac{x_t-\sigma\,\hat\epsilon_\theta(x_t,t)}{\alpha}$.  
- **VE**：$x_t=x_0+\sigma\epsilon \ \Rightarrow\ 
\hat x_0=x_t-\sigma\,\hat\epsilon_\theta(x_t,\sigma)$.  
- **Score 参数化**：$\hat x_0=x_t-\sigma^2(t)\,s_\theta(x_t,t)$（VP 需再除以 $\alpha$）。

> **为什么要会反解**：  
> ① 反向步的均值修正要用；② 监控/理解模型学到的“内容”；③ 与不同参数化（$\epsilon/x_0/v/score$）互相转换。

---

## 9) 设计选择与实践直觉（为什么这些工程细节有效）

- **连续时间/连续噪声（EDM 思路）**：覆盖全 SNR 区间，允许使用二阶/高阶积分器（如 Heun），在更少步数下保持质量。  
- **损失加权（按 SNR 或 $\sigma$）**：不同噪声级的学习难度不同，合适的加权能让网络把容量用在“信息丰富”的区间。  
- **参数化互换**：$\epsilon$-头（DDPM）、$x_0$-头、$v$-头、score-头可线性互换；选择出于**数值稳定**与**任务习惯**。  
- **调度与先验衔接**：VP 末端天然对齐高斯先验；VE 在高噪端近似高斯，配合良好的 $\sigma$ 日程易于稳定采样。


