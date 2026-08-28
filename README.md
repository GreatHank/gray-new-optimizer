# gray_new

基于固定物理前向模型的多通道衍射灰度优化研究项目，用共享的二维相位梯度生成 23 个不同衍射级次的目标图案。

## 背景与目标

项目面向几何相位超表面/远场全息设计。当前阶段使用 23 通道二值目标验证“固定前向、只调整损失与优化策略”能否改善成像质量和跨通道亮度一致性，并为后续相位导出、版图生成和 FDTD 验证提供数据。

适用场景包括算法实验、不同损失权重对比、结果可视化，以及由相位矩阵生成 CIF 掩模。

## 当前能力

- 从 MAT 文件读取形状为 `23 × 500 × 500`、标签为 `0/1` 或 `0、1/3、2/3、1` 的目标 `bw_all`。
- 优化共享相位变量 `dx/dy`，按固定级次组合重建 23 通道远场强度。
- 可选跨通道亮区一致性损失与最差通道惩罚。
- 支持固定灰度比例、跨通道灰度一致性、线内均匀性和背景逐行均匀性损失。
- 输出优化结果、逐通道指标和目标/结果对比图。
- 提供权重计算、已有 NPZ 结果可视化及 MATLAB CIF 掩模生成辅助脚本。

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
├─ input/                   # 原始输入数据
│  └─ grayscale_image.mat
└─ output/                  # 运行结果与验证产物
```

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

该测试使用独立的 NumPy 参考实现，对固定随机 `dx/dy` 的 23 个通道逐值比对，并检查 Parseval 能量关系和共享通道级次配置。

生成三线四标签校准目标：

```powershell
python coding/python/create_three_line_target.py --output-dir output/three_line_calibration
```

工具生成 `23 × 500 × 500` 的 `bw_all` MAT 文件、统一曝光预览图和逐通道像素计数 CSV。三条水平线长度 200、宽度 5、中心行间隔 60，从上到下为 `1/3、2/3、1`。

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

主流程会额外输出 `grayscale_level_metrics.csv`，包含三档有效响应、比例、单调性、背景方差和背景逐行方差。

输出目录必须是尚不存在的新目录。完整优化默认 30000 轮，运行时间和显存/内存占用明显高于快速验证。

辅助脚本：

```powershell
python coding/python/calculate_weight.py
python coding/python/show_save_img.py
```

`show_save_img.py` 和 `coding/matlab/mask_generate.m` 保留了原有的本机绝对路径/固定目录配置，使用前需要在脚本内按实际结果位置修改。当前仍未改写这些路径。

## 核心架构与流程

主流程是：输入目标 → 初始化共享 `dx/dy` → 固定物理前向重建各通道 → 计算损失并用 Adam 优化 → 重建与计算指标 → 写入 NPZ、CSV 和 PNG。

不可变物理映射为 `Φc = mc·dx + nc·dy`、`Uc = exp(iΦc)`、`Ic = |fftshift(FFT2(Uc))|²`。详细模块职责、数据流和输出格式见 [项目架构说明](doc/project-architecture.md)，当前目标见 [四级灰度需求](doc/grayscale-requirements.md)，执行步骤见 [四级灰度实施计划](doc/grayscale-implementation-plan.md)，理论与迁移约束见 [概念迁移包](doc/concept_migration_pack/README.md)。

## 配置与外部依赖

主脚本通过命令行参数配置 MAT 文件、轮数、学习率、设备、输出目录、日志间隔和两项一致性损失权重。没有环境变量、网络服务、数据库或认证依赖。

MAT 输入必须包含 `bw_all`，且为 23 个正方形通道，标签只能是 `0/1` 或 `0、1/3、2/3、1`。MATLAB 脚本另依赖优化流程导出的 `phdx.csv` 和 `phdy.csv`；当前 Python 主脚本只在 NPZ 中保存这两个数组，并不导出对应 CSV。

## 已知限制与风险

- 当前数据与通道数固定为 23，权重和衍射级次矩阵写在主脚本内。
- 主脚本以逐通道循环执行 2D FFT，完整训练计算量较大。
- 辅助可视化和 MATLAB 脚本含本机路径，尚未参数化。
- Python 输出与 MATLAB 输入格式目前没有直接衔接：MATLAB 需要相位 CSV，主流程只生成 NPZ。
- `mask_generate.m` 原文件存在中文注释编码显示异常；为保持源码不变，本次未转换编码。
- 阶段 C 已完成一次 30000 轮固定比例基线；阶段 D 已完成多组短轮探索，候选 007 当前背景逐行方差最低且保留三档结构，但仍需正式长跑和第二随机种子复核。这些结果不等同于 FDTD/器件验证。

## 当前状态与下一步

项目已完成目录归类、架构梳理和主流程最小运行验证。当前处于 `experiment/four-level-lines` 分支的阶段 D：已在固定物理前向下加入灰度比例、跨通道一致性、档位间隔、线内均匀性、可见性、最差档和背景均匀性损失，并完成候选 001–007 短轮探索。候选 007 当前最接近要求：23 个通道三档仍单调，背景逐行方差约为 `0.0243`；但统一曝光图仍有弱横线，尚未进入正式长跑或水果轮廓实验。四级灰度的核心需求和实施计划已按用户反馈更新。

当前不是 Keil 工程，不涉及标准空工程选择。
