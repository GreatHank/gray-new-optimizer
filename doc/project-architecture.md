# 项目架构说明

## 1. 项目定位

该项目是脚本式科研计算流程，不是 Web 服务或安装包。核心任务是在不可修改的衍射物理前向下，优化两个共享二维相位变量 `dx`、`dy`，使 23 组衍射级次分别形成对应目标图案，并评估各通道的效率、对比度和亮度一致性。

## 2. 模块划分

| 分类 | 文件 | 职责 |
|---|---|---|
| 核心优化 | `coding/python/order_decoupling_grayscale.py` | 读取目标、执行固定前向、计算损失、Adam 优化、重建并保存结果 |
| 权重工具 | `coding/python/calculate_weight.py` | 根据给定通道强度计算补偿后的自定义权重 |
| 结果工具 | `coding/python/show_save_img.py` | 读取既有 `optimized_results.npz`，输出目标图、重建图、曲线和逐通道 CSV |
| 版图工具 | `coding/matlab/mask_generate.m` | 将 `phdx/phdy` 相位 CSV 映射为单元位移并生成 CIF 掩模 |
| 输入 | `input/grayscale_image.mat` | 变量 `bw_all`，形状 `23 × 500 × 500`，类型 `uint8`，值为 0/1 |
| 文档 | `doc/` | 原始任务说明、算法说明、固定前向与损失设计资料 |
| 输出 | `output/` | 每次实验的 NPZ、CSV、PNG 等运行产物 |

## 3. 主流程

1. `load_targets` 从 MAT 文件读取并校验 `bw_all`。
2. `select_device` 选择 CPU 或 CUDA。
3. 随机初始化两个 `500 × 500` 的共享相位变量 `phdx/phdy`。
4. `total_cost` 对 23 个 `(m,n)` 级次逐一执行固定物理前向：
   - `Φc = mc·dx + nc·dy`
   - `Uc = exp(iΦc)`
   - `Ic = |fftshift(FFT2(Uc))|²`
5. 基础损失综合目标图 MSE、目标区域能量效率和通道权重；可选项约束目标亮区的跨通道变异系数及最差通道。
6. Adam 更新共享的 `phdx/phdy`。
7. `reconstruct` 生成原始强度、逐通道归一化强度和全局统一尺度强度。
8. `channel_metrics` 统计能量效率、前景/背景均值和对比度。
9. 保存数值结果与对比图。

## 4. 数据与输出

主脚本的 `optimized_results.npz` 包含：

- 优化变量：`phdx`、`phdy`
- 训练记录：`costs`、`eta_history`、亮区一致性相关历史
- 固定配置：`pairMat`、`weights`、损失权重
- 输入与重建：`targets`、`optimized_raw`、`optimized_01`、`optimized_global_01`

同一实验目录还会产生：

- `channel_brightness_metrics.csv`
- `target_vs_optimized_0_1.png`
- `target_vs_optimized_global_scale.png`

## 5. 设计约束

- 物理前向不能改变；改进范围限于目标、损失、损失权重、优化策略和验收指标。
- `dx/dy` 是 23 通道共享设计变量，不能通过逐通道后处理缩放冒充器件自身的亮度一致性。
- 跨通道亮度验收应使用原始强度或统一尺度结果，逐通道归一化结果只适合观察图案形状。
- 当前程序结构是单脚本主流程，规模较小，暂不需要引入包化、框架化或额外抽象。

## 6. 当前断点

Python 主流程输出 `phdx/phdy` 到 NPZ，而 MATLAB 版图脚本读取 CSV，两者之间缺少正式导出步骤。辅助脚本还含固定的本机路径。由于本次要求不修改任何源代码，以上问题仅记录，未增加兼容层或隐式兜底。
