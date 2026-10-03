# 物理模型与论文对齐

## 证据状态

参考论文：Z. Zhang et al., *Breaking the Diffraction-Encoding Limit for High-Capacity Meta-Holography via Multiorder Decoupling*, ACS Nano 20 (2026) 18823–18830, DOI `10.1021/acsnano.6c04285`。

已核对论文库正文与 [ACS 官方补充材料](https://acs.figshare.com/articles/journal_contribution/Breaking_the_Diffraction-Encoding_Limit_for_High-Capacity_Meta-Holography_via_Multiorder_Decoupling/32774869) 的 S5–S6。S5 式 (S9) 定义 `SNR(dB)=20log10(μ_signal/σ_noise)`，15 dB 是平均 SNR 的可接受阈值；S6 报告平均 SNR 18.77 dB。本文用目标非黑像素和黑像素分别定义两个区域，补充材料未明确逐像素掩膜，因此不能直接对标论文图表。补充材料中的精确级次表、Table S3、效率与串扰数字尚未逐项核对。

以下区分 [Paper] 原文内容、[Inference] 模型推导与 [Question] 未验证问题。

## 固定软件前向

当前比较 configs/circle-ocean-3x3-p800-500-strict9.json 与 configs/circle-ocean-4x4-p1050-500-fullcircle.json。这两组阵列为 500×500；λ=532 nm、φ=45°、NA=1。3×3 为 P=800 nm、θ=70.1276°、m,n=-2…0；4×4 为 P=1050 nm、θ=20.9938°、m,n=-2…1。这些来自用户截图，不作为论文样品参数复现。

[Paper] 已核实原论文 Methods 使用 Adam、25000 次迭代、lr=0.001。PP、目标图、当前共同尺度损失与同灰度一致项属于本项目设置。

[Inference] 共同入射居中条件为 sinθ=√2·|mean(m)|·λ/P，得到 70.127606° 与 20.993831°，与截图保留四位小数一致。3×3 中心 9/9 能传播，名义单元为 1 个完整、8 个部分可见；4×4 中心 12/16 能传播，4 个完整、12 个部分可见。两组实际非黑目标均完整入圆，4×4 保留四个部分可见的角单元。

[Inference] 空气侧 N×N 阵列单个 FFT 频率格的方向余弦间距是 λ/(P·N)。N=500 时，3×3 为 0.00133，4×4 为 0.00101333。4×4 的角域采样更细，也增加通道数、计算成本与共享相位的目标约束；仅靠中心传播数或网格数不能判定重建画质。完整方格范围与实际目标支撑的角域必须区分。

`coding/optimization/model.py` 使用：

```text
Phi_c = m_c * dx + n_c * dy + PP
U_c   = exp(i * Phi_c)
I_c   = |fftshift(fft2(U_c))|^2
```

本项目采用每个像素可独立设定的软件相位 `dx/dy/PP`，PP 以 `[-π,π)` 表示并按 `2π` 周期回绕，边界不是优化硬限制。在 `(0,0)` 级次，`Phi_00=PP`；正负级次的相位和为 `2PP`，不再强制共轭。每个像素仍只有三个共享自由度，多个级次保留仿射相位关系。PP 扩展是本项目的模型，不是下述论文的原文机制。

这与正文第 3 页 Eq. (1) 的位移相位关系 `q_mn=2πmΔx/Px+2πnΔy/Py` 及 Figure 3e 的 `phi_mn=m phi_x+n phi_y` 对应。由此只能得到 `Δx=mod(dx/(2π)*Px,Px)`、y 同理；实际结构可实现性还需单元响应、边界和制造模型。

[Paper] 纯位移模型中，正负级次近场互为复共轭，对应远场强度的镜像关系。

[Inference] 对当前逐像素 PP 扩展，`U[-m,-n]=exp(2i*PP)*conj(U[m,n])`。空间变化的因子不再能从 FFT 中提出，因而不强制远场镜像；若 PP 是常量，镜像关系仍存在。两个相反级次可赋予不同的目标相位，例如 `(1,0)` 和 `(-1,0)` 的任意相位 A、B 可用 `dx=(A-B)/2`、`PP=(A+B)/2` 实现。

[Inference] 这不提供逐级次独立自由度：`Phi[m,n]+Phi[-m,-n]=2Phi[0,0]`，以及 `Phi[1,0]+Phi[0,1]=Phi[1,1]+Phi[0,0]` 仍成立。因此可选候选包括零级和正负级次，但编码质量和可用角域还受共享相位关系、共同入射传播锥、NA 与指定级次图覆盖目标限制。

## 当前验证边界

[Inference] 空气侧级次中心由 u=sinθ cosφ+mλ/Px、v=sinθ sinφ+nλ/Py 决定。局部 FFT 频率网格围绕这个中心展开；几何预检分别核对级次中心、方格与实际非黑目标支撑。中心在传播圆外不代表整个方格不可见。

[Inference] P800 的严格九宫格覆盖约 99.9700% 视场圆面积，未选方向不约束。P1050 的 16 级次配置覆盖完整圆。两类配置分别保留，不自动增减级次。

当前数值检查使用独立 NumPy 前向与 Parseval。SNR 采用各选中级次完整 FFT 平面的原始强度，前景 T>0、背景 T=0；展示使用共同曝光。目标区能量占比仅表示软件结果的能量分布。

[Question] 逐像素 PP 的器件实现、单元响应、偏振、联合分光、全波与实验均未验证。PP 不能直接称为 PB 旋转相位。几何通过与软件画质不能作为器件效率或论文复现的证明。
