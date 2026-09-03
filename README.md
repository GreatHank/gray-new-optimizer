# gray_new

基于固定物理前向模型的多通道衍射灰度优化研究项目，用共享的二维相位梯度生成最多36个不同衍射级次的目标图案。

## 背景与目标

项目面向几何相位超表面/远场全息设计。当前物理配置使用第一象限 `m,n∈{1,…,6}` 的36个级次，验证共享相位下的多通道成像、灰度分级和跨通道亮度一致性，并为后续相位导出、版图生成和 FDTD 验证提供数据。

适用场景包括算法实验、不同损失权重对比、结果可视化，以及由相位矩阵生成 CIF 掩模。

## 当前能力

- 从 MAT 文件读取1–36个正方形通道、标签为 `0/1` 或 `0、1/3、2/3、1` 的目标 `bw_all`。
- 优化共享相位变量 `dx/dy`，按第一象限6×6级次矩阵重建最多36通道远场强度。
- 可选跨通道亮区一致性损失与最差通道惩罚。
- 支持固定灰度比例、跨通道灰度一致性、线内均匀性、背景空间均匀性和全背景亮度带损失。
- 支持不区分方向的背景局部亮度集中损失，可用不同邻域尺度抑制背景亮斑成块。
- 默认使用总能量归一化的灰度分布损失，使稀疏水果目标按 `1:2:3` 分配衍射能量；旧的总 MSE 和前景/背景分区 MSE 仍可显式选择用于对照。
- 支持逐目标像素最低响应损失，并以灰度线自身均值归一化线内方差，用于减少缺线和线内斑驳。
- 输出优化结果、逐通道指标和目标/结果对比图。
- 每次优化自动输出结构完整、灰度分级、同灰度跨通道一致性和背景均匀性四类评价，并生成逐通道与汇总 CSV。
- 提供权重计算、已有 NPZ 结果可视化及 MATLAB CIF 掩模生成辅助脚本。
- 提供单一全局光强倍率测试，验证所有channel同步增亮时绝对强度线性增加，而CV和相对灰度比例保持不变。

## 技术栈与环境

- Python 3
- NumPy、SciPy、PyTorch、Matplotlib
- MATLAB（仅 `mask_generate.m`）
- CUDA 可选；主脚本也支持 CPU

本项目目前没有锁定依赖版本或自动安装脚本。已验证环境为 Python 3.13.1、NumPy 2.2.1、SciPy 1.15.1、PyTorch 2.11.0、Matplotlib 3.10.0。

## 目录结构

```text
gray_new/
├─ README.md
├─ doc/                     # 需求、算法说明和概念迁移文档
│  ├─ project-architecture.md
│  └─ concept_migration_pack/
├─ coding/
│  ├─ python/               # 优化、目标生成、权重计算和结果可视化脚本
│  └─ matlab/               # 相位到 CIF 掩模转换脚本
├─ tests/                   # 固定前向和校准目标回归测试
├─ teach/                   # CV优先损失方案的简洁教学课程与速查资料
├─ input/                   # 原始输入数据
│  ├─ grayscale_image.mat
│  ├─ thin_letter_reference.png
│  └─ geometric_network_reference.png
└─ output/                  # 仅存放可删除、可重新生成的运行结果
```

`input/` 与 `output/` 严格分离：用户提供的原图或 MAT 数据只放入 `input/`；目标 MAT、训练结果、指标和预览图只放入 `output/`。2026-08-31 已按用户要求清空旧 `output/`，因此下文早期实验路径仅作为历史指标记录，文件本身已不存在；当前磁盘只保留本轮几何线稿实验产物。

## 运行方式

在项目根目录运行主优化流程：

```powershell
python coding/python/order_decoupling_grayscale.py `
  --mat-file input/grayscale_image.mat `
  --output-dir output/run_001 `
  --device auto
```

快速验证：

```powershell
python -m compileall -q coding/python
python coding/python/order_decoupling_grayscale.py --mat-file input/grayscale_image.mat --output-dir output/smoke_test --epochs 1 --device cpu
```

固定物理前向回归验证：

```powershell
python tests/test_physical_forward.py
```

该测试使用独立的 NumPy 参考实现，对固定随机 `dx/dy` 的36个通道逐值比对，并检查 Parseval 能量关系、第一象限级次范围和共享相位配置。

生成三线四标签校准目标：

```powershell
python coding/python/create_three_line_target.py --output-dir output/three_line_calibration
```

工具生成 `36 × 500 × 500` 的 `bw_all` MAT 文件、统一曝光预览图和逐通道像素计数 CSV。三条水平线长度 200、宽度 5、中心行间隔 60，从上到下为 `1/3、2/3、1`。

从 `input/grayscale_image.mat` 的二值水果线稿生成填充三灰度目标：

```powershell
python coding/python/create_three_level_fruit_target.py `
  --output-dir output/fruit_three_level_filled
```

脚本先逐通道填充封闭轮廓，再按距离边界的远近生成外层 `1/3`、中层 `2/3`、核心 `1`，输出 MAT、预览图和像素统计。

### 6×6 满幅细线字母目标

用户提供的字母细线图已保存为 `input/thin_letter_reference.png`。运行：

```powershell
python coding/python/create_thin_letter_target.py
```

脚本先按有效内容范围切成6×6共36块，并将全部图块按行优先顺序映射到 `(1,1)…(6,6)`。每个图块直接重采样铺满500×500 channel，禁止居中粘贴或追加黑色保护区；黑色只表示原图中主体线条之间的真实背景。接近背景的浅装饰纹会被过滤，保留线条再量化为 `1/3、2/3、1` 三档。36通道目标位于 `output/thin_letter_6x6_target_36`。

最新几何线稿使用空间连续分档，避免细线的三个灰度只落在线宽抗锯齿边缘而视觉上接近全白：

```powershell
python coding/python/create_thin_letter_target.py `
  --input-file input/geometric_network_reference.png `
  --output-dir output/geometric_network_target_36 `
  --background-delta 15 `
  --min-line-intensity 30 `
  --level-assignment spatial
```

`spatial` 会把每个满幅图块中的有效线条按空间位置连续划分为深灰、中灰、亮灰三组；`source` 则保留按原图像素亮度量化的对照方式。两种方式都不会增加通道边缘保护区，也不会对优化结果做逐通道增益。

本轮约 1000 次验证命令为：

```powershell
python coding/python/order_decoupling_grayscale.py `
  --mat-file output/thin_letter_6x6_target/thin_letter_6x6_target.mat `
  --output-dir output/thin_letter_main_strokes_1000 `
  --epochs 1000 `
  --lr 1e-3 `
  --device auto
```

该验证仅使用主图案损失，没有逐 channel 亮度缩放，也没有提前加入 CV、背景或线内均匀辅助项。

阶段 D 短轮实验可从已有结果 warm-start，并固定阶段 C 的公共尺度 `Q`：

```powershell
python coding/python/order_decoupling_grayscale.py `
  --mat-file output/three_line_calibration/three_line_target.mat `
  --output-dir output/stage_d_candidate_007 `
  --epochs 1500 `
  --lr 5e-4 `
  --device auto `
  --initial-results output/baseline_four_level_lines/optimized_results.npz `
  --reference-q 29.23249 `
  --level-weight 0.2 `
  --cross-level-weight 0.1 `
  --gap-weight 0.02 `
  --line-uniformity-weight 0.02 `
  --visibility-weight 0.05 `
  --worst-level-weight 0.1 `
  --background-uniformity-weight 0.2 `
  --background-row-uniformity-weight 3.0
```

主流程会额外输出 `grayscale_level_metrics.csv`，包含三档有效响应、比例、单调性、背景方差和背景逐行方差。稀疏三灰度目标默认使用 `--image-loss-mode energy`，直接匹配目标与重建的总能量分布，不再使用容易受单个亮斑支配的最大值归一化。用 `--channel-count N` 可以只优化前 N 个通道，用于检查共享相位变量的多通道可行性。

每次优化结束还会生成 `evaluation_channel_metrics.csv` 和 `evaluation_summary.csv`，并在终端打印四项评价。指标定义和读法见 [优化结果评价指标](doc/evaluation-metrics.md)。

CV优先方案的简洁课程见 [CV优先灰度损失课程](teach/lessons/0001-cv-priority-loss.html)，公式速查见 [损失速查页](teach/reference/loss-map.html)。

水果目标从随机相位起跑时，先保持灰度、线内均匀性和背景辅助损失为默认的 `0`。旧的 `balanced` 加多项辅助损失组合会在早期压过图案主损失，使三档比例数值看似正确但整体亮度塌缩。基础图案收敛后如需改善跨通道亮度或背景，再从已有结果热启动并逐项增加约束。

背景亮度带实验不针对横向纹理，而是对所有背景像素统一计算 `I / 全平面均值` 的越界惩罚。只限制亮斑时使用 `--background-band-lower 0 --background-band-upper 1.0`；同时限制过亮和过暗区域时，例如使用 `--background-band-lower 0.6 --background-band-upper 1.0`。两种方案的权重由 `--background-band-weight` 控制。

背景局部集中实验使用 `--background-cluster-weight`、`--background-cluster-kernel` 和 `--background-cluster-upper`。它先计算每个背景像素邻域内的背景平均亮度，再惩罚超过局部上限的邻域；9×9 主要观察小亮斑/小亮块，21×21 主要观察较宽的成块和扩散纹。

本轮已完成两组短实验：`output/stage_d_background_band_upper`（上限方案）和 `output/stage_d_background_band_two_sided`（上下限方案）。两组均保持 23/23 通道三档响应单调，但统一曝光图中仍能看到线条附近的残余扩散纹，因此暂不视为通过背景验收。

随后完成两组局部集中短实验：`output/stage_d_background_cluster_9x9` 和 `output/stage_d_background_cluster_21x21`。9×9 结果的背景方差约为 `0.5214`，21×21 约为 `0.5280`；两者仍保留三档结构，但尚未消除可见残余扩散纹。

在此基础上又完成 9×9 加强实验 `output/stage_d_background_cluster_9x9_stronger`：局部集中权重为 `2.0`、局部上限为 `0.9`，背景方差降至约 `0.5064`，但统一曝光图仍残留扩散纹，尚不能作为最终方案。

输出目录必须是尚不存在的新目录。完整优化默认 30000 轮，运行时间和显存/内存占用明显高于快速验证。

辅助脚本：

```powershell
python coding/python/calculate_weight.py
python coding/python/show_save_img.py
```

全局入射光强测试：

```powershell
python coding/python/simulate_global_illumination.py `
  --results-file output/geometric_network_36_brightness_q2_strong_stage2_500/optimized_results.npz `
  --output-dir output/global_illumination_test_1x_2x_4x `
  --factors 1 2 4
```

该测试只对全部36个通道共同乘以一个光强倍率，不重新优化相位，也不允许逐channel增益。固定曝光图用于观察整体变亮；每种倍率分别归一化的图会完全相同，不能用于判断绝对亮度。

`show_save_img.py` 和 `coding/matlab/mask_generate.m` 保留了原有的本机绝对路径/固定目录配置，使用前需要在脚本内按实际结果位置修改。当前仍未改写这些路径。

## 核心架构与流程

主流程是：输入目标 → 初始化共享 `dx/dy` → 固定物理前向重建各通道 → 计算损失并用 Adam 优化 → 重建与计算指标 → 写入 NPZ、CSV 和 PNG。

不可变物理映射为 `Φc = mc·dx + nc·dy`、`Uc = exp(iΦc)`、`Ic = |fftshift(FFT2(Uc))|²`。详细模块职责、数据流和输出格式见 [项目架构说明](doc/project-architecture.md)，当前目标见 [四级灰度需求](doc/grayscale-requirements.md)，执行步骤见 [四级灰度实施计划](doc/grayscale-implementation-plan.md)，理论与迁移约束见 [概念迁移包](doc/concept_migration_pack/README.md)。

## 配置与外部依赖

主脚本通过命令行参数配置 MAT 文件、轮数、学习率、设备、输出目录、日志间隔、灰度一致性损失、背景均匀性损失和背景亮度带上下限。没有环境变量、网络服务、数据库或认证依赖。

MAT输入必须包含 `bw_all`，允许1–36个正方形通道，标签只能是 `0/1` 或 `0、1/3、2/3、1`。MATLAB脚本另依赖优化流程导出的 `phdx.csv` 和 `phdy.csv`；当前Python主脚本只在NPZ中保存这两个数组，并不导出对应CSV。

## 已知限制与风险

- 当前默认级次为第一象限 `(1,1)…(6,6)` 共36项，基础损失权重统一为1；旧23通道MAT仍可显式使用 `--channel-count 23`。
- 6×6正级次没有正负共轭或坐标轴级次，但含有 `(1,1)/(2,2)…` 等整数倍谐波关系，因此36项不是36个完全独立方向。
- 主脚本以逐通道循环执行 2D FFT，完整训练计算量较大。
- 辅助可视化和 MATLAB 脚本含本机路径，尚未参数化。
- Python 输出与 MATLAB 输入格式目前没有直接衔接：MATLAB 需要相位 CSV，主流程只生成 NPZ。
- `mask_generate.m` 原文件存在中文注释编码显示异常；为保持源码不变，本次未转换编码。
- 阶段 C 已完成一次 30000 轮固定比例基线；阶段 D 已完成多组短轮探索，并新增全背景亮度带与局部集中方案。背景残余扩散纹仍未解决；阶段 E 的 9×9 候选 5000 轮正式短跑（seed42）已完成，尚未进入第二随机种子复核。这些结果不等同于 FDTD/器件验证。
- 阶段 G 已从 23 张二值水果线稿重新生成填充三灰度目标，填充后前景占比约 `1.98%`。已修复二值输入误计算空灰度档导致的 NaN，并加入 `--channel-count` 通道缩减验证。进一步确认最大值归一化 MSE 与多项线条辅助损失会导致亮度塌缩，现已改用总能量归一化的灰度分布损失。

## 当前状态与下一步

项目已完成目录归类、架构梳理和主流程最小运行验证。当前处于 `experiment/four-level-lines` 分支的阶段 G。按用户指定的水果灰度轮廓线目标 `output/fruit_three_level/fruit_three_level_target.mat` 完成了 23 通道、1000 轮验证，结果位于 `output/verify_fruit_three_level_lines_ch23_1000`：23/23 通道三档均单调，平均比例约为 `0.351:0.683:1`，所有水果轮廓线在统一曝光图中可辨认。但三个灰度档的跨通道 CV 均约为 `0.53`，最高档响应范围为 `2.52–14.68`，尚未达到通道亮度一致。

下一步采用两阶段热启动：从上述结果继续优化，固定公共最高档尺度 `Q=5.834266`（本轮最高档响应均值），只启用低权重的 `level` 与 `cross-level` 约束，把每个通道三档响应拉向共同的 `Q/3、2Q/3、Q`。暂不同时加入线内均匀性和背景约束，避免再次压过图案主损失；候选结果必须同时检查跨通道 CV、最弱通道、前景效率和统一曝光轮廓。

首个一致性热启动候选 `output/verify_channel_consistency_1000` 已完成1000轮。三个灰度档的跨通道 CV 从约 `0.53` 降至 `0.31–0.33`；最高档最小值从 `2.52` 提升到 `4.63`、最大值从 `14.68` 降至 `13.27`，最大/最小比从约 `5.82` 降至 `2.87`。23/23 通道保持单调，平均前景效率由 `4.46%` 提升到 `5.87%`。方向有效，但当前低权重尚未达到最终通道亮度一致。

按新增四项评价复算该候选：结构余弦相似度均值/最差为 `0.661/0.505`，前景覆盖率均值/最差为 `88.3%/71.1%`；23/23 通道灰度单调，比例 RMSE 均值/最差为 `0.022/0.033`；三档跨通道 CV 为 `0.322/0.327/0.310`；背景像素 CV 约 `1.00`、P95/均值约 `3.00`、逐行 CV 约 `0.048`。因此当前结论是轮廓大体形成、通道内灰度分级良好、跨通道一致性仍不足、背景存在明显细粒度散斑但无强烈行方向起伏。

已进入四目标联合损失阶段。改动严格限定在损失函数：新增逐目标像素最低响应项 `structure-completeness`，并将线内均匀性改为按各灰度线自身均值归一化；固定物理前向、目标、通道共享关系和显示方式均未改变。

四目标联合优化完成三个递进候选，当前综合最优为 `output/joint_loss_candidate_3_1000`。最终结构相似度均值/最差为 `0.761/0.740`，前景覆盖率均值/最差为 `98.2%/94.2%`；23/23 通道灰度单调，比例 RMSE 均值/最差为 `0.0092/0.0133`，三档均值比例约为 `0.347:0.675:1`；三档跨通道 CV 为 `0.138/0.139/0.135`；背景像素 CV 为 `0.945`、P95/均值为 `2.905`、逐行 CV 为 `0.052`。与联合优化前相比，结构、灰度比例、同档一致性和像素级背景均有改善，逐行背景CV略有增加但仍处于较低水平。

用户随后将同灰度跨通道CV明确设为最高优先级。CV优先热启动结果位于 `output/cv_priority_candidate_1000`：三档跨通道CV进一步降到 `0.026/0.043/0.050`；结构相似度均值/最差为 `0.771/0.725`，前景覆盖率为 `98.7%/95.8%`；23/23通道单调，比例RMSE均值为 `0.0077`；背景像素CV为 `0.949`，未出现明显恶化。该结果取代第三联合候选，成为当前推荐结果。

新增满幅细线字母验证 `output/thin_letter_main_strokes_1000`。目标来自 6×6 网格中的 23 个高对比主体线条块，未添加保护区。1000 轮纯主图案损失后，结构相似度均值/最差为 `0.530/0.392`，前景覆盖率均值/最差为 `34.1%/15.3%`；23/23 通道三档单调，比例 RMSE 均值/最差为 `0.015/0.024`；三档跨通道 CV 为 `0.277/0.271/0.276`；背景像素 CV 为 `1.000`、P95/均值为 `2.995`、逐行 CV 为 `0.046`。统一曝光下各通道主体轮廓可辨，但细线偏暗且部分低灰度笔画不完整，因此这是可行性结果而非最终高质量候选；后续若继续，应从该相位热启动并在结构不回退的前提下再加入跨通道一致性损失。

针对细线灰度不明显与同档不一致，已验证“单边抬亮 + 一致性收尾”的损失顺序。对称 `level` 损失会同时压低过亮通道，不适合作为可见性主项；最终方案使用 `reference-q=4`，以 `worst-level` 作为只惩罚低于 `Q×[1/3,2/3,1]` 的亮度下限，再用较强 `cross-level` 收紧同档跨通道差异。最终候选位于 `output/thin_letter_gray_final_candidate_300`：结构相似度均值/最差 `0.547/0.468`，覆盖率均值/最差 `37.1%/24.8%`，23/23 通道单调，比例 RMSE 均值 `0.0218`；三档 CV 降至 `0.035/0.079/0.102`，最高档均值/最小值从原始 `2.21/1.25` 提高到 `2.45/1.68`，背景指标基本不变。优化前后图表由 `coding/python/plot_grayscale_comparison.py` 生成，结果为 `output/thin_letter_gray_final_candidate_300/grayscale_loss_comparison.png`。

按用户确认新增36级次方案：`PAIR_MAT={(m,n)|m,n∈{1,…,6}}`，不含零轴和负级次，36项基础权重统一为1。目标位于 `output/thin_letter_6x6_target_36`，1000轮随机相位结构基线位于 `output/thin_letter_36channels_main_1000`。结果结构相似度均值/最差为 `0.491/0.155`，覆盖率均值/最差为 `34.0%/6.3%`；36/36通道三档单调，比例RMSE均值为 `0.0189`；三档CV为 `0.567/0.570/0.572`。最差结果集中在 `(1,1)、(2,1)、(1,2)、(2,2)`及同方向谐波附近，证明仅排除正负共轭不足以保证级次独立；该配置已实现且可运行，但当前36通道一致性明显弱于原23通道候选。

当前不是 Keil 工程，不涉及标准空工程选择。

### 2026-08-31 当前结果：36通道几何线稿

输入为 `input/geometric_network_reference.png`，目标为 `output/geometric_network_target_36/thin_letter_6x6_target.mat`。1000轮结构基线的结构相似度均值为 `0.492`、覆盖率均值为 `0.352`，但三档跨通道CV约为 `0.726/0.732/0.735`。随后只通过损失权重和热启动进行4段各500轮的CV优先收尾，当前候选为 `output/geometric_network_36_cv_stage4_500`：

- 结构相似度均值/最差：`0.358/0.232`；前景覆盖率均值/最差：`0.162/0.089`。
- 36/36通道灰度单调；比例RMSE均值/最差：`0.038/0.199`。
- 深灰/中灰/亮灰跨通道CV：`0.054/0.133/0.180`。
- 背景像素CV均值 `1.001`，P95/均值 `2.996`，逐行CV `0.051`。

深灰档已接近用户期望的 `0.05`，但中灰和亮灰未达到；继续强压CV会进一步降低结构和可见亮度，因此没有把该结果描述为“三档均达到0.05”。限制主要来自36个第一象限级次中的整数倍谐波耦合，而不是显示缩放。固定物理前向、共享 `dx/dy` 和统一基础权重均未改变。

在此候选上先完成500轮温和公共亮度校准，再完成两段各500轮的强化增亮，最终结果位于 `output/geometric_network_36_brightness_q2_strong_stage2_500`。只把所有通道共用的最高档目标从 `Q=1` 提高到 `Q=2`，并提高已有档位下限、结构完整和前景能量损失权重；没有逐通道增益。相对增亮前，三档有效亮度由 `0.399/0.823/1.262` 提高到 `0.479/1.013/1.580`，增幅约为 `20.2%/23.0%/25.2%`；结构相似度由 `0.358` 提高到 `0.403`，覆盖率由 `0.162` 提高到 `0.223`，三档CV为 `0.052/0.116/0.150`。这证明真实增亮可行，但继续逼近 `Q=2` 仍会受到36级次耦合和固定总能量约束。

随后用该结果验证统一提高入射光强。公共倍率为 `1×/2×/4×` 时，平面均值及三个灰度档的原始有效亮度严格按倍率增加；三档CV始终为 `0.052/0.116/0.150`，相对灰度比例始终为 `0.311:0.646:1`。因此在当前线性模型中，提高照明强度可以让整幅图同步变亮，但不会改善或破坏相对灰度关系、通道一致性和结构。

用户随后取消优化器的绝对增亮目标，最终只保留灰度比例/间隔、跨通道CV、结构完整和前景覆盖约束，并在输出端统一使用物理入射光强 `2×`。两段各500轮CV收尾后的推荐结果为 `output/geometric_network_36_cv_gap_stage2_500`：三档CV为 `0.009/0.049/0.078`，比例RMSE为 `0.0181`，平均相邻差值为 `0.460/0.485`，36/36通道单调；结构相似度均值/最差为 `0.383/0.263`，覆盖率均值/最差为 `0.198/0.147`。相对上一候选，结构均值下降约5%、覆盖率均值下降约11%，但最差覆盖率提高约20%，背景指标基本不变。最终 `2×` 固定曝光验证位于 `output/geometric_network_36_cv_gap_stage2_global_2x`。
