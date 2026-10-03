# 多级次共享相位灰度优化

当前输入为老虎、圆形星空、乌龟、山涧、牡丹、盘龙和凤凰。全部级次共享 dx、dy、PP 三张相位图：调整相位 → FFT → 比较固定目标 → Adam 更新。前向 Φ=m dx+n dy+PP，强度为 |fftshift(FFT2(exp(iΦ)))|²；不逐通道调整增益或曝光。

## 查看结果

`input/` 和 `output/` 仅保存在本地，不纳入 Git 版本管理。克隆仓库后需自行准备配置对应的输入图片；已有结果索引所引用的数据也需另外提供。

打开 output/runs，按原图分类。老虎、圆形星空、乌龟和山涧各保留严格 3×3 与 4×4 的全平面、圆内约束结果；牡丹、盘龙和凤凰各有一张严格 3×3 圆内约束结果。每次运行只放一张左 target、右最终效果的 PNG。图上标平均 SNR，统一曝光 1×。全平面与圆内 SNR 的统计区域不同，不能直接比较。

相位、CSV、完整报告及跨实验比较在 output/data，派生目标在 output/targets。configs/selected-results.json 对应效果图与原始数据；续跑使用 data 中的相位文件。

## 级次与配置

默认入口为 configs/circle-ocean-3x3-p800-500-strict9.json，即乌龟、500×500 阵列、严格九级次、Adam 25000 步。各图的 `*-observable.json` 配置使用对应网格原目标与优化参数，仅将图像损失限制在固定传播/NA 可观测掩膜内。

| 方案 | 周期与入射极角 | 级次 | 配置 |
|---|---|---|---|
| 严格 3×3 | P800 nm、θ70.1276° | m,n=-2…0，共9个 | circle-{cosmic,ocean,forest}-3x3-p800-500-strict9.json |
| 4×4 | P1050 nm、θ20.9938° | m,n=-2…1，共16个 | circle-{cosmic,ocean,forest}-4x4-p1050-500-fullcircle.json |
| 新图严格 3×3 圆内约束 | P800 nm、θ70.1276° | m,n=-2…0，共9个 | circle-{peony,dragon,phoenix}-3x3-p800-500-strict9-observable.json |

当前阵列为500×500，共750000个共享相位参数。共同 λ532 nm、φ45°、NA1。圆形模式把源圆映射到视场圆，保留圆内明暗；P800 严格九级次覆盖约99.9700%圆面积，4×4可覆盖完整视场圆。老虎使用 tiger-3x3-p800-500.json 或 tiger-4x4-p1050-500.json，整图缩放、放入黑画布，不裁内容。

精修仅使用严格3×3。圆形星空续跑配置为 configs/circle-cosmic-3x3-p800-500-strict9-refine.json，使用相同视场圆目标、从25000步相位续跑10000步。strict9-consistency2.json配置将同灰度亮度一致权重设为2，用于比较SNR和亮度均衡的折中；4×4不再续跑精修。


## 运行

依赖 Python、NumPy、SciPy、Pillow、Matplotlib、PyTorch；测试需 pytest，CUDA 可选。从项目根目录执行：

```powershell
python -m coding.targets.prepare
python -m coding.optimization.run
python -m pytest
```

目标已存在时跳过 prepare。优化结束自动发布对比图；数据目录按配置、阵列、优化器、步数和北京时间命名，禁止覆盖。其他图案用 --config 指定配置，完整报告与比较命令见 [运行说明](doc/usage.md)。

代码按 coding/targets、optimization、geometry、results、tests 分工。前向始终保留完整 FFT 和原始尺度；旧结果使用全平面指标，新 `observable` 实验只在固定掩膜内评价图像，旧新分数须在同一掩膜内重评后比较。几何通过和软件画质不等于器件实现、联合分光或论文复现。

[实验结果](doc/experiments.md) · [有效约束](doc/requirements.md) · [物理依据](doc/physics.md)。visualization_tools 包含独立的[级次预览](visualization_tools/order_visualizer/index.html)与[球面图像观测台](visualization_tools/spherical_viewer/index.html)：后者可直接双击打开，支持全景角度贴图与衍射方向余弦圆图；默认从底面圆心以常规透视环顾建筑，也支持 180° 整顶仰视、外部观察和大屏观赏。内置武大黑天全景示例。main 为长期主线，参数实验使用配置与独立 run id。
