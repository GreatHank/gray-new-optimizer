# 运行入口

当前输入为老虎、圆形星空、乌龟、山涧、牡丹、盘龙、凤凰。阵列为500×500，默认乌龟严格九级次。正常优化结束自动生成 output/runs/<图片名称>/<实验名称>.png，左target、右最终效果；相位和指标放 output/data，派生目标在 output/targets。

## 默认运行

```powershell
python -m coding.targets.prepare
python -m coding.geometry.preflight --config configs/circle-ocean-3x3-p800-500-strict9.json
python -m coding.optimization.run
```

目标已存在时跳过prepare。5000步短程试验可添加 --epochs 5000；新数据目录与效果图禁止覆盖。

## 配置选择

- 圆形星空、乌龟与山涧严格九级次：configs/circle-{cosmic,ocean,forest}-3x3-p800-500-strict9.json。
- 三张圆形图的4×4全圆映射：configs/circle-{cosmic,ocean,forest}-4x4-p1050-500-fullcircle.json。
- 圆形星空保留完整原图：configs/circle-cosmic-{3x3-p800,4x4-p1050}-500-original.json。
- 老虎整图：configs/tiger-{3x3-p800,4x4-p1050}-500.json。
- 四张图的圆内约束九级次：configs/tiger-3x3-p800-500-observable.json，以及 configs/circle-{cosmic,ocean,forest}-3x3-p800-500-strict9-observable.json。
- 四张图的圆内约束 4×4：configs/tiger-4x4-p1050-500-observable.json，以及 configs/circle-{cosmic,ocean,forest}-4x4-p1050-500-fullcircle-observable.json。
- 牡丹、盘龙、凤凰的圆内约束九级次：configs/circle-{peony,dragon,phoenix}-3x3-p800-500-strict9-observable.json。分别使用对应附件原图及独立源圆半径，不裁切原始输入。
- 圆形星空3×3视场圆精修：configs/circle-cosmic-3x3-p800-500-strict9-refine.json；从同目标25000步结果续跑10000步、lr0.0003、亮度一致权重1。权重2对照使用configs/circle-cosmic-3x3-p800-500-strict9-consistency2.json，与权重1从同一基线分别续跑。精修仅做严格九级次，4×4不再续跑。

prepare、preflight和run均可用 --config 显式指定配置。source_circle 的center_px、radius_px、outside_background_max_gray分别定义每张源图的圆心、半径及可清除的圆外背景上限，不能混用不同图片的参数。

P800严格九级次覆盖约99.9700%视场圆面积，require_full_field_coverage=false；未选方向不约束。P1050的4×4可覆盖完整视场圆，require_full_field_coverage=true。两者圆内明暗均不配平，原始input不变。默认 `full_plane` 沿用完整 FFT 平面损失与指标；`observable` 配置固定逐像素传播/NA 掩膜，只约束和评价圆内像素。圆内原图黑背景仍计入，圆外不参与图像损失。新旧分数须使用同一掩膜重评。

## 设置与可选报告

默认Adam25000步、lr0.001、seed43、gray_consistency_weight=1。可覆盖 --epochs、--lr、--optimizer-name adam/lbfgs；L-BFGS线搜索可能增加求值次数。--initial-results只恢复三张相位并重建优化器。

平时看图无需生成详细报告。需独立NumPy/Parseval核查或逐级次指标时运行：

```powershell
python -m coding.results.report --result-dirs output/data/<run-id> --output-dir output/data/<report-id>
```

报告默认目标是乌龟严格九级次；其他实验必须加 --target-dir，并指向配置中mat_file的父目录。跨网格比较使用 coding.results.compare_grids，显式传两组运行、对应报告和新的output/data目录；比较不改变原始数据或统计窗口。

四张图的圆内约束实验使用上述各自 `observable` 配置。完成后可用 `python -m coding.results.compare_observable --baseline output/data/<旧运行> --candidate output/data/<新运行> --output-dir output/data/<新比较目录>` 重评两组完整 raw；命令检查相同目标、优化参数、几何掩膜、独立 NumPy 前向与完整平面 Parseval，并保存统一 1× 曝光对比图、逐级次 CSV 和 JSON 报告。`observable` 运行的 `optimized_results.npz` 保存完整 raw、相位和固定掩膜；`gray_levels.csv` 是掩膜内逐级次原始灰度响应。

重新发布已有结果可用 python -m coding.results.publish --result-dirs output/data/<run-id>；同名图片会报错。SNR=20log10(前景均值/背景标准差)，前景T>0、背景T=0，汇总逐级次dB的平均和最低值。统一曝光只改展示，目标区能量占比不等于器件效率。

## 验证

```powershell
python -m pytest
```

没有自动制造导出、中途checkpoint或提前停止。物理支撑通过不等于图像、器件或论文复现。
