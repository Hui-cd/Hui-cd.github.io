# GAIT

> Generating Aesthetic Indoor Tours with Deep RL — 用深度强化学习生成室内美学游览路径

## Overview

GAIT 是辉少的研究项目，使用视觉 Actor-Critic 深度强化学习（DrQ-v2 和 CURL）在室内 3D 场景中生成美学相机轨迹。核心挑战在于：如何让智能体学会「美」的概念，并生成平滑、多样、不重复的游览路径。

## Key Facts / Claims

### 框架架构
- **数据循环**：Actor 与环境交互，存储 transition 到 Replay Buffer
- **更新工作器**：从 Replay Buffer 采样，基于 RL 损失更新 Actor 和 Critic
- **共享存储**：同步数据和更新步骤，维持 2:1 的数据-更新比率

### 使用的 RL 算法
1. **DrQ-v2**：基于 Q 学习和 DDPG，critic 估计 $Q(s_t,a_t)$，actor 最大化期望回报
2. **CURL**：加入策略熵项（SAC 风格），增强探索性

### 奖励函数设计（四维）
1. **出界惩罚**：$r_B = -10$，防止相机离开场景
2. **视图美学**：用神经网络美学模型评估单帧质量（240×240 高分辨率渲染）
3. **时间平滑性**：惩罚动作突变，生成平滑轨迹
4. **多样性正则化**：避免不同初始姿态收敛到同一最优姿态
   - 排除姿态：前 4 个回合的最终姿态
   - 距离惩罚：$r_t^D = \min_j(\min(\frac{\|x_t^P - \bar{x}_j^P\|_2}{d_j}, 1))$

### 多 GPU 设置
- Actor 持续与环境交互
- Update Worker 并行更新网络
- 通过共享存储同步，保持 actor 网络最新

## Related
- [[machine-learning-basics]] — RL 的基础数学
- [[diffusion-model]] — 另一大生成模型范式
- [[llm-rl-algorithms]] — DrQ-v2 和 CURL 算法
- [[projects-overview]] — 辉少的项目索引

## Counter-arguments & Data Gaps
- 单帧美学模型无法评估序列整体美感（时间维度）
- 美学评估依赖人群数据，主观性强
- 与基于优化的路径规划方法的对比不足

## Sources
- [GAIT: Generating Aesthetic Indoor Tours](../sources/2023-10-26-GAIT.md) — 2023-10-26
