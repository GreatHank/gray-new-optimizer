# 衍射级次物理可行性验证

`coding/python/validate_diffraction_orders.py` 用二维光栅方程检查任意级次集合能否在指定输出介质中传播、是否落入收集数值孔径，并报告零级、正负共轭和整数谐波关系。级次整数坐标不必以 `(0,0)` 为中心；斜入射产生的横向波矢可以把一个偏置的连续级次区域整体平移进固定的空气锥。

## 两种不能混用的物理模式

### 固定入射的大视场模式

`--mode fixed-incidence` 表示所有级次共享同一束入射光。空气锥固定在实验室坐标系中，改变入射角实际改变的是入射横向波矢，因此整组衍射级次点阵相对空气锥平移。各个可传播级次位于不同出射方向，共同拼接或扩展角域；它们不会同时位于视场中心。

论文的23通道样例正是这种情况：补充材料Table S1和Figure S5给出的级次主要位于 `m=-3,-2,-1` 的单侧区域，并补充 `(0,1)`、`(0,2)`，而不是围绕零级对称排列。Table S3的出射角与 `lambda=488 nm`、`P_x=P_y=2000 nm`、约 `46°` 的+x方向自由空间斜入射相符。斜入射使这些负m级次落入空气锥的可用区域。需要注意，Table S3的表体还额外列出 `(0,-1)`、`(0,-2)`，合计25项，与“23 orders”标题存在内部不一致；代码的 `paper-23` 以明确展示23幅重建图的Figure S5为准。

### 逐级次居中模式

`--mode center-each-order` 对每个 `(m,n)` 分别反算一种入射条件，使该级次垂直出射：

```text
n_in sin(theta_i) cos(phi_i) = -m lambda / Lambda_x
n_in sin(theta_i) sin(phi_i) = -n lambda / Lambda_y
```

只有所需横向动量不超过 `n_in` 时，自由空间入射才可实现。输出表中的每一行代表一次独立照明，不能解释成一次照明同时让全部级次居中。若自由空间动量不足，可进一步研究波导或倏逝波入射，但当前脚本尚未把它们等效成自由空间角度。

李仲阳团队的大视场工作不是以“级次中心点到原点的最远距离乘二”定义最终视场。补充材料Figure S9说明，大视场图案由相邻衍射级次承载的多个全息区域级联形成；实际视场取决于被利用的全息角谱区域，而不只是离散级次中心。因此当前脚本输出的是级次中心传播可行性，尚不能单独复现论文的137°或168°完整图像视场。

补充材料S3还给出更大的近180°方案：在x/y位移产生的迂回相位之外，再增加纳米结构旋转角产生的PB相位。PB相位对LCP/RCP分别取相反符号，可解除纯迂回相位下正负共轭级次的约束，并使 `(0,0)` 也可由PB相位承载图案。`--conjugate-encoding detour-only` 对应当前只有 `dx/dy` 的代码；`detour-pb` 只用于表示增加PB自由度后的运动学筛查，现有优化器尚未实现该旋转角变量。

## 模型

设真空波长为 `lambda`，二维衍射周期为 `Lambda_x/Lambda_y`，入射极角 `theta` 从表面法线量起，方位角为 `phi`。输出方向余弦为：

```text
u_x = (n_in sin(theta) cos(phi) + m lambda / Lambda_x) / n_out
u_y = (n_in sin(theta) sin(phi) + n lambda / Lambda_y) / n_out
```

当 `u_x^2 + u_y^2 <= 1` 时，该级次可在输出介质传播；同时满足 `n_out sqrt(u_x^2+u_y^2) <= NA` 时，才会被给定数值孔径收集。

## 使用

按仓库当前名义参数验证推荐连续7×7级次：

```powershell
python coding/python/validate_diffraction_orders.py `
  --wavelengths-nm 480 `
  --period-x-nm 850 `
  --period-y-nm 850 `
  --incident-theta-deg 0 `
  --incident-azimuth-deg 0 `
  --n-in 1 `
  --n-out 1 `
  --na 1 `
  --grid-size 7 `
  --order-m-start -9 `
  --order-n-start 3 `
  --target-mat output/circular_7x7_target_49/circular_7x7_target.mat `
  --output-dir output/diffraction_order_validation_480nm_850nm
```

多波长和逐波长入射角也可同时输入。例如三个波长应分别给三个极角和三个方位角；只给一个角度时，该角度会用于全部波长。

默认会依据目标MAT中的 `grid_positions` 忽略纯黑图块。若要验证完整级次网格（包括当前没有分配图案的角级次），加入 `--all-grid-positions`。

### 7×7 大圆形目标的推荐参数

圆形目标四个角块为纯黑，因此实际使用45个通道。选择连续单侧级次
`m=-7…-1、n=-3…3`，并令入射横向方向余弦满足
`sin(theta_i)=4 lambda/P`，可把该偏置级次块的中心 `(m,n)=(-4,0)`
平移到输出空气锥中心。该选择不含零级，也不同时包含任一对正负共轭级次。

在 `lambda=488 nm`、空气中传播和 `NA=1` 下，已验证三组参数：

| 周期 P | 入射极角 | 有效通道 | 最差空气锥余量 | 最外有效级次极角 | 结论 |
|---:|---:|---:|---:|---:|---|
| 2000 nm | 77.4219° | 45/45 | 0.1202 | 61.61° | 角域最大，但近掠入射风险高 |
| 2200 nm | 62.5325° | 45/45 | 0.2002 | 53.11° | 推荐折中 |
| 2400 nm | 54.4229° | 45/45 | 0.2669 | 47.15° | 传播最稳，但角域较小 |

推荐命令：

```powershell
python coding/python/validate_diffraction_orders.py `
  --mode fixed-incidence `
  --wavelengths-nm 488 `
  --period-x-nm 2200 `
  --period-y-nm 2200 `
  --incident-theta-deg 62.532515 `
  --incident-azimuth-deg 0 `
  --n-in 1 `
  --n-out 1 `
  --na 1 `
  --grid-size 7 `
  --order-m-start -7 `
  --order-n-start -3 `
  --target-mat output/circular_7x7_target_49/circular_7x7_target.mat `
  --output-dir output/circular_order_validation_m-7_-1_n-3_3_P2200
```

这只证明45个级次中心在运动学上可传播和收集。最终圆形图案的连续覆盖范围还取决于每个级次承载的局部全息角谱宽度；器件效率与串扰仍需联合优化及电磁仿真验证。

上述2200 nm方案只是在允许改变有效衍射周期时的理论折中。用户现已明确禁止改变物理结构，因此它不再作为当前器件的实施参数。

### 固定850 nm结构下的圆形裁切

若现有 `850 nm` 确为真实衍射周期，且波长固定为 `480 nm`，则级次中心间距为 `lambda/P≈0.5647`。连续7×7中心阵列的总跨度约3.388，大于自由空间方向余弦圆的直径2；入射角只能整体平移点阵，不能缩小间距，因此45个有效中心不可能全部进入空气锥。

对7×7方阵做亚级次平移搜索后，半格偏置可使最多12个级次中心落入单位圆。进一步把每个级次对应的局部角谱暂按一个 `lambda/P` 宽的方格估计时，有16个方格与单位圆相交；这些边界方格的可见部分可以被圆形孔径裁切，从而覆盖到接近±90°的几何边界。其余完全在圆外的级次不能形成传播远场。当前验证器只验证级次中心，尚未把局部角谱方格、圆内像素比例和跨级次拼接纳入正式输出。

重新核对论文补充材料 Figure S2 和 Figure S9 后，应把“方格与圆相交”从近似展示升级为下一版验证器的核心模型。Figure S9明确采用9个相邻级次区域拼接时钟，并以红色圆形区域定义实际图案视场；Figure S2的近180°示例同样采用3×3方格，但额外使用PB相位独立编码共轭级次和零级。因此，圆形裁切思路本身正确，但纯 `dx/dy` 结构只能验证传播区域，不能自动获得论文近180°方案的全部独立编码自由度。详细阅读记录见 [李仲阳团队衍射级次解耦与大视场论文阅读记录](li-zhongyang-large-fov-reading-notes.md)。

复现论文补充材料中的23级次和斜入射布局：

```powershell
python coding/python/validate_diffraction_orders.py `
  --mode fixed-incidence `
  --order-preset paper-23 `
  --wavelengths-nm 488 `
  --period-x-nm 2000 `
  --period-y-nm 2000 `
  --incident-theta-deg 46 `
  --incident-azimuth-deg 0 `
  --n-in 1 `
  --n-out 1 `
  --na 1 `
  --output-dir output/diffraction_validation_paper23_488nm_2000nm_46deg
```

逐级次反算中心入射角：

```powershell
python coding/python/validate_diffraction_orders.py `
  --mode center-each-order `
  --wavelengths-nm 480 `
  --period-x-nm 2400 `
  --period-y-nm 2400 `
  --grid-size 7 `
  --order-m-start -3 `
  --order-n-start -3 `
  --target-mat output/circular_7x7_target_49/circular_7x7_target.mat `
  --output-dir output/diffraction_order_center_each_480nm_2400nm
```

## 输出

- `diffraction_order_map.png`：空气锥、NA边界和所选级次位置。
- `selected_order_metrics.csv`：每个波长和级次的方向余弦、出射角、传播状态与边界余量。
- `independent_candidate_orders.csv`：传播并满足当前共轭编码假设的候选级次。纯迂回相位时每对共轭只保留一个；选择 `detour-pb` 时允许同时保留共轭级次。
- `conservative_primitive_orders.csv`：在上一组基础上继续排除同方向整数谐波的保守候选级次。
- `center_incidence_requirements.csv`：仅在逐级次居中模式生成，记录每个级次所需的独立入射角及自由空间可行性。
- `continuous_block_ranking.csv`：搜索范围内连续级次方阵的物理可行性排名。
- `validation_summary.json`：本次假设、参数和汇总结论。

## 边界

程序假设输入的 `period-x/y-nm` 是决定倒空间级次间距的真实衍射周期。仓库中的 `850 nm` 也可能只是纳米单元间距；在该含义确认之前，使用850 nm得到的结果只能作为条件性诊断。

## 无零级完整6×6的最新诊断

用户要求`(0,0)`绝对排除，并要求36个有效图像通道构成完整连续6×6。仿照论文23-channel的单侧布局，取`m=-6…-1、n=-3…2`，该集合没有零级，也没有正负共轭对。

- `lambda=480 nm、P=850 nm`：遍历全部自由空间允许的共同入射方向，最多只有10/36个级次中心进入空气锥。根因是六个连续中心的单轴跨度`5lambda/P≈2.824`已经超过空气锥直径2。
- `lambda=480 nm、P=2000 nm`理论对照：级次间距为0.24，使用`theta_i≈58.05°、phi_i≈8.13°`可使36/36中心传播，最小空气锥余量约0.1515。
- 论文23-channel：`lambda=488 nm、P=2000 nm、theta_i≈46°`时23/23传播；集合为三个负m列、七个n行共21项，加`(0,1)`与`(0,2)`，零级和共轭对均未入选。

因此，在850 nm确为真实衍射周期且结构不得改变的前提下，“无零级、连续、完整有效6×6”三项不能同时满足。2000 nm结果仅用于定位物理瓶颈，不是实施建议。

角度约定：`theta_i`是相对表面法线的极角，`phi_i`是入射横向波矢从`+x`朝`+y`的方位角。对`P=2000 nm`理论6×6，入射方向余弦为`(0.84,0.12)`，故`theta_i≈58.05°、phi_i≈8.13°`。完整6×6内接圆角谱的标称全视场为`2asin(3lambda/P)≈92.11°`；若引用方阵对角最远中心之间的夹角，则约为116.10°，必须注明这是方形包络而非圆图视场。

纯迂回相位模型中的 `(0,0)` 不含 `m*dx+n*dy` 相位，不能承载当前模型定义的独立目标。纯迂回相位下，若同时选择 `(m,n)` 与 `(-m,-n)`，两者复场严格共轭；论文S3通过增加PB相位自由度解决这两个限制。整数倍同方向级次会共享基础相位组合，但论文的联合优化确实使用了这类级次，因此脚本只把它标为相关性风险，不能据此判定“不可解耦”。

传播条件和上述关系筛查都不能代替解耦优化、效率仿真或实验。要预测每个级次的实际功率、串扰、偏振响应和材料损耗，还需要联合优化结果以及RCWA/FDTD，并补充材料折射率、层厚、纳米结构几何、偏振和边界条件。

## 资料依据

- Z. Zhang et al., *Breaking the Diffraction-Encoding Limit for High-Capacity Meta-Holography via Multiorder Decoupling*, ACS Nano 20, 18823–18830 (2026), DOI: https://doi.org/10.1021/acsnano.6c04285 。
- 论文补充材料，DOI: https://doi.org/10.1021/acsnano.6c04285.s001 。Table S1、Figure S5和Table S3给出23级次、2000 nm周期及出射角；S3给出迂回相位与PB相位混合编码共轭级次的近180°方案；S5给出约51通道、15 dB SNR阈值和14.2°最小级次间隔的容量估计。
- *衍射级次解耦方法、多维加密超构光学存储的设计、显示及应用方法*, CN120010114B：https://patents.google.com/patent/CN120010114B/zh 。用于核对改变入射条件使目标级次垂直出射的光栅方程和实验例子。
