# 7. 术语与公式速查

## 术语

| 术语 | 含义 |
|---|---|
| channel | 一个固定 `(m,n)` 衍射级次对应的软件输出通道 |
| `dx/dy` | 所有channel共享、允许由优化器更新的两张设计相位图 |
| 物理前向 | 从 `dx/dy` 经级次组合、复指数、FFT到强度的固定映射 |
| 损失函数 | 衡量候选 `dx/dy` 结果好坏的标量目标，不直接改图 |
| 目标区效率 `η` | 全平面能量进入目标掩膜的比例 |
| 白线水平 `B_c` | 白线平均原始强度除以全平面平均强度 |
| 公共灰度响应 | 同一灰度档在不同channel中对应同一原始强度 |
| 档内均匀性 | 同一channel、同一灰度档像素的离散程度 |
| 杂散 | 落在非目标或允许区域之外的实际能量 |
| 统一曝光 | 所有channel共用同一显示尺度；只用于查看，不改变原始数组 |
| 逐channel归一化 | 每路除以自己的峰值；只可看形状，不可验收绝对亮度 |

## 固定前向

\[
\Phi_c=m_cdx+n_cdy
\]

\[
U_c=e^{i\Phi_c}
\]

\[
I_c=|fftshift(FFT2(U_c))|^2
\]

\[
\sum I_c=P^2
\quad\text{（当前单位模、默认未归一化FFT）}
\]

## 二值亮度一致性

\[
B_c=mean(I_c[T_c=1])/mean(I_c)
\]

\[
L_{consistent}=mean[((B_c-\overline B)/(\overline B+\varepsilon))^2]
\]

\[
L_{worst}=[1-min(B_c)/(\overline B+\varepsilon)]^2
\]

## 四灰度响应

\[
B_{c,l}=mean(I_c[T_c=l])/mean(I_c)
\]

\[
L_{level}=mean_{c,l}(B_{c,l}-Q_l)^2
\]

\[
B_{c,l+1}-B_{c,l}\ge\Delta_l
\]

\[
L_{uniform}=\sum_{c,l}Var(I_c[T_c=l])
\]

## 不可变规则

```text
只允许：目标、损失、权重、优化器、日志、指标
禁止：修改前向公式、逐channel增益、FFT后缩放、独立曝光验收
```
