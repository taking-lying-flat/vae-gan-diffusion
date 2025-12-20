# Generative Adversarial Networks (GAN)

## 1. 论文链接

**Generative Adversarial Nets** (Goodfellow et al., **NIPS 2014**)  

- NeurIPS/NIPS Proceedings 论文主页：  
  https://papers.nips.cc/paper/5423-generative-adversarial-nets

- NeurIPS/NIPS Proceedings 官方 PDF：  
  https://papers.neurips.cc/paper/5423-generative-adversarial-nets.pdf


---

## 2. 总结

GAN 提出了一种通过 **对抗过程 (Adversarial Process)** 来学习生成模型的新框架，其本质是一个 **两人零和博弈 (Zero-Sum Game)**：

- **生成器 (Generator, \(G\))**：输入噪声 \(z\sim p_z(z)\)，输出伪造样本 \(G(z)\)，目标是让判别器将其判为真。
- **判别器 (Discriminator, \(D\))**：输出 \(D(x)\in(0,1)\)，表示样本来自真实分布 \(p_{\text{data}}\) 的概率，目标是区分真样本与假样本。

训练中交替更新 \(D\) 和 \(G\)。在理想纳什均衡处，生成分布满足 \(p_g = p_{\text{data}}\)，此时最优判别器无法区分真假，对任意输入输出 \(D(x)=\frac12\)。

---

## 3. 核心公式与关键结论

### (1) 极小极大博弈目标函数 (Minimax Objective)

GAN 的价值函数定义为：

- **\(D\) 的目标（Max）**：最大化对真实样本的 \(\log D(x)\) 与对生成样本的 \(\log(1-D(G(z)))\)。
- **\(G\) 的目标（Min）**：最小化生成样本被识别为假的概率项 \(\log(1-D(G(z)))\)。

\[
\min_{G}\max_{D}V(D,G)
=
\mathbb{E}_{x\sim p_{\text{data}}(x)}[\log D(x)]
+
\mathbb{E}_{z\sim p_z(z)}[\log(1-D(G(z)))]
\]

---

### (2) 给定生成器 \(G\) 时，判别器 \(D\) 的最优解

固定生成器（即固定 \(p_g\)）时，判别器的最优形式为：

\[
D_G^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}
\]

直观含义：当 \(p_{\text{data}}(x)\) 越大且 \(p_g(x)\) 越小，判别器越倾向于判真；反之判假。

---

### (3) 理论核心：等价于最小化 Jensen–Shannon Divergence

将 \(D_G^*(x)\) 代回价值函数，得到：

\[
V(D_G^*,G) = -\log 4 + 2\cdot \mathrm{JSD}\big(p_{\text{data}} \,\|\, p_g\big)
\]

因此，当判别器达到最优时，优化生成器相当于最小化 \(p_g\) 与 \(p_{\text{data}}\) 的 Jensen–Shannon 散度。最优点为：

\[
p_g = p_{\text{data}}
\]

---

### (4) 生成器的实际训练目标 (Non-saturating Heuristic)

理论 minimax 中，生成器优化：

\[
\min_G\ \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
\]

但在训练早期，判别器往往很强，使得 \(D(G(z))\approx 0\)，从而导致 \(\log(1-D(G(z)))\) 的梯度容易饱和。论文提出常用的替代目标（启发式）：

\[
\max_G\ \mathbb{E}_{z\sim p_z}[\log D(G(z))]
\]

等价写成最小化：

\[
\min_G\ -\mathbb{E}_{z\sim p_z}[\log D(G(z))]
\]

这就是常说的 **non-saturating GAN loss**

---

## 4. 训练算法（实践视角）

- 交替优化（Alternating Optimization）：通常每轮更新 \(D\) 若干步，再更新 \(G\) 一步
- 用 mini-batch 近似期望：上面的两项期望都用小批量采样估计，配合 SGD / Adam 等优化器进行训练
- 目标并非严格凸-凹：实际训练可能出现不稳定、震荡、模式崩塌等现象，后续许多 GAN 变体（WGAN、LSGAN、SN-GAN 等）都是为缓解这些问题提出的
