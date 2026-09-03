# 4. 四灰度跨通道统一方案

## 4.1 目标定义

设目标包含4个灰度档：

\[
l\in\{0,1,2,3\}
\]

希望所有channel的三个非零线条档位相对各自背景共享一条公共响应；黑背景本身不要求跨channel一致：

\[
Q_1<Q_2<Q_3
\]

对任意channel `c` 和非零档 `l∈{1,2,3}`：

\[
\mu_{c,l}-\mu_{c,0}\approx Q_l
\]

## 4.2 输入要求

- 输入不再限制为 `{0,1}`，而是明确的4档标签；
- 每个channel都应包含足够数量的4档像素；
- 第一轮实验使用相同校准块，而不是复杂自然图；
- 必须记录每个 `(channel, level)` 的有效像素数；
- 缺失某一档的channel不能参与该档一致性验收。

## 4.3 统一标尺

禁止每channel各自除以最大值。定义：

\[
B_{c,l}=
\frac{\mu_{c,l}-\mu_{c,0}}
{\operatorname{mean}(I_c)+\varepsilon}
\]

响应矩阵为：

\[
R=[B_{c,l}]\in\mathbb R^{K\times4}
\]

理想状态是三个非零档对应的矩阵列各自接近公共常数。背景档单独统计，但不要求不同channel取相同均值。

## 4.4 公共灰度损失

若四档由用户固定：

\[
L_{\text{level}}
=\frac1{4K}\sum_{c,l}(B_{c,l}-Q_l)^2
\]

跨通道一致性：

\[
\overline B_l=\frac1K\sum_cB_{c,l}
\]

\[
L_{\text{cross}}
=\frac1{4K}\sum_{c,l}
\left(
\frac{B_{c,l}-\overline B_l}
{\overline B_l+\varepsilon}
\right)^2
\]

## 4.5 防止四档塌缩

必须要求相邻档有最小间隔：

\[
B_{c,l+1}-B_{c,l}\ge\Delta_l
\]

软约束形式：

\[
L_{\text{gap}}
=\sum_{c,l}
\operatorname{ReLU}
\left[
\Delta_l-(B_{c,l+1}-B_{c,l})
\right]^2
\]

否则优化器可能让四档都变成同一亮度，从而获得虚假的跨通道一致性。

## 4.6 档内均匀性

平均值正确不代表每个像素正确。加入：

\[
L_{\text{uniform}}
=\sum_{c,l}
\operatorname{Var}
\{I_c(x,y)\mid T_c(x,y)=l\}
\]

它抑制同一灰度档内部的强散斑和局部过亮/过暗。

## 4.7 黑背景、杂散和最差组合

- 黑色背景：不约束跨channel均值，首轮也不优化背景均匀性；
- 杂散比例：作为观察指标，只有确实淹没线条时才考虑加入弱约束；
- 最差组合：统计所有 `(c,l)` 中最大的公共曲线误差；
- 单调性：每个channel的三档线条必须满足 `B_c,1<B_c,2<B_c,3`，并保持相对背景可辨认。

## 4.8 推荐完整损失

\[
L=
L_{\text{image}}
+\lambda_1L_{\text{level}}
+\lambda_2L_{\text{cross}}
+\lambda_3L_{\text{gap}}
+\lambda_4L_{\text{uniform}}
+\lambda_5L_{\text{worst}}
+\lambda_6L_{\text{black}}
+\lambda_7L_{\text{leak}}
\]

所有损失都只能读取冻结前向产生的原始 `I_c`；不得为满足灰度曲线而修改前向或添加逐通道增益。

## 4.9 公共曲线的两种选择

### 固定曲线

用户直接给出 `Q_0…Q_3`。优点是目标和验收明确，适合第一轮可行性验证。

### 可学习公共曲线

允许优化器在统一上下限内学习4档，但必须锚定黑色上限、白色下限和最小间隔。否则会出现整体缩放或四档塌缩。

首版推荐固定曲线。

## 4.10 能量预算

若第 `l` 档在channel `c` 中有 `N_{c,l}` 个像素：

\[
E_{c,\text{image}}
\approx\sum_lN_{c,l}Q_l
\]

不同channel的灰度直方图不同，所需图像能量不同；剩余固定总能量必须落在允许的背景或泄能区域。若没有能量出口，公共灰度曲线和低黑底可能同时不可行。

这仍然只能通过损失引导能量分配，不能为解决冲突而改动固定前向。
