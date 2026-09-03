# 3. 损失函数设计：从二值一致性到多目标折中

## 3.1 原图案损失

目标区域效率：

\[
\eta_c=
\frac{\sum I_cM_c}
{\sum I_c+\varepsilon}
\]

逐channel峰值归一化只用于形状项：

\[
\widehat I_c=rac{I_c}{\max(I_c)+\varepsilon}
\]

原图案损失：

\[
L_{\text{base}}
=\sum_c
w_c[8(1-\eta_c)]^3
\sum(\widehat I_c-T_c)^2
\]

它保持轮廓形状并鼓励目标区效率，但因为每路各自归一化，无法约束跨通道原始亮度。

## 3.2 二值白线亮度指标

定义：

\[
B_c=
\frac{\operatorname{mean}(I_c[T_c=1])}
{\operatorname{mean}(I_c)+\varepsilon}
\]

当前模型中各通道全平面平均强度相同，所以令 `B_c` 相等等价于令白线平均原始强度相等。使用平均值而非目标区总能量，可以消除不同图案白色像素数的影响。

## 3.3 跨通道一致性损失

\[
\overline B=\frac1K\sum_cB_c
\]

\[
L_{\text{consistent}}
=\frac1K\sum_c
\left(
\frac{B_c-\overline B}
{\overline B+\varepsilon}
\right)^2
\]

其平方根就是亮区CV。高于平均的通道受到压低方向，低于平均的通道受到推高方向。

## 3.4 最差通道保护

\[
r_{\min}=
\frac{\min_cB_c}
{\overline B+\varepsilon}
\]

\[
L_{\text{worst}}=(1-r_{\min})^2
\]

该项重点照顾当前最暗通道。它是相对约束；防止所有通道一起变暗的关键仍是保留 `L_base` 和效率因子。

## 3.5 完整二值损失

\[
L=L_{\text{base}}
+\operatorname{stopgrad}(L_{\text{base}})
\left[
\lambda L_{\text{consistent}}
+\mu L_{\text{worst}}
\right]
\]

`stopgrad(L_base)` 只给两个无量纲损失提供与原损失相近的数值尺度，不复制 `L_base` 的梯度。

当 `λ=μ=0` 时必须严格退回原算法。这是新项目的兼容性测试，不是隐藏降级。

## 3.6 三股梯度的职责

| 损失 | 主要作用 | 单独使用的风险 |
|---|---|---|
| `L_base` | 保持目标形状、提高目标区效率 | 各通道原始亮度可相差很大 |
| `L_consistent` | 缩小亮区平均强度CV | 可能主要压低强通道 |
| `L_worst` | 保护当前最暗通道 | 不规定绝对图案质量 |

三项共同作用后，优化器通常会提高弱通道、适当牺牲过强通道，并保留可辨认图案。

## 3.7 只改损失，为什么输出会变

```text
新的损失
  → 新的梯度
  → Adam找到不同的dx/dy
  → 同一个冻结前向产生不同干涉分布
  → 目标区、背景和杂散之间的能量重新分配
```

禁止在这条链末尾增加逐通道软件缩放。所有改善都必须能从最终 `dx/dy` 通过冻结前向独立回放。

## 3.8 损失设计原则

- 指标必须从未逐channel缩放的原始强度计算；
- 不能只看平均值，必须保留最差通道指标；
- 不能只追求一致性，必须保留图案质量和背景约束；
- 权重变化只改变优化偏好，不增加物理自由度；
- 若软权重无法达到硬容差，可使用增广拉格朗日，但仍不得修改物理前向。
