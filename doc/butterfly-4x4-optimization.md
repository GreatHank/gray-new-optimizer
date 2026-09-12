# 蝴蝶图连续4×4级次优化记录

## 固定条件

- 输入：`input/butterfly.png`，2048×2048，严格只有`0/85/170/255`
- 波长：480 nm
- 周期：`P_x=P_y=2000 nm`
- `n_in=n_out=1`，`NA=1`
- 连续级次：`m,n=1…4`，按m优先排列，共16项
- 映射：整幅源图统一旋转180°后切4×4；最终场景重排后再整体旋转180°恢复方向
- 灰度：最近邻缩放到500×500，固定映射为`0/1/3/2/3/1`

没有加入`(0,0)`，没有单独交换图块，没有PB相位、逐channel增益或FFT后
亮度修正。优化仍使用共享相位`Φc=mc·dx+nc·dy`。

## 物理级次验证

使用共同入射`theta_i=58.05194°、phi_i=-135°`。该角度把4×4级次方阵
中心平移到空气锥内：16/16级次中心传播并被NA=1收集，最小空气锥余量为
`0.490883`，最小同波长级次间距为`0.24`。

名义圆形FoV为：

```text
2*asin(4*480/(2*2000)) = 57.37080°
```

集合不含共轭对，但含8对共线关系、5个非本原谐波级次和20项加法关系。
因此运动学传播通过不等同于16幅图完全独立。

验证命令：

```powershell
python coding/python/validate_diffraction_orders.py `
  --wavelengths-nm 480 --mode fixed-incidence `
  --order-preset continuous --conjugate-encoding detour-only `
  --period-x-nm 2000 --period-y-nm 2000 `
  --incident-theta-deg 58.05194 --incident-azimuth-deg -135 `
  --n-in 1 --n-out 1 --na 1 `
  --grid-size 4 --order-m-start 1 --order-n-start 1 `
  --target-mat output/butterfly_4x4_rot180_target_16/butterfly_4x4_target.mat `
  --all-grid-positions `
  --output-dir output/butterfly_4x4_m1_4_n1_4_P2000_validation
```

## 目标与优化命令

```powershell
python coding/python/create_butterfly_4x4_target.py `
  --input-file input/butterfly.png `
  --output-dir output/butterfly_4x4_rot180_target_16

python coding/python/order_decoupling_grayscale.py `
  --mat-file output/butterfly_4x4_rot180_target_16/butterfly_4x4_target.mat `
  --output-dir output/butterfly_4x4_m1_4_n1_4_main_100 `
  --epochs 100 --lr 5e-4 --seed 42 --channel-count 16 `
  --order-grid-size 4 --order-m-start 1 --order-n-start 1 `
  --image-loss-mode energy --device auto --log-interval 20

python coding/python/order_decoupling_grayscale.py `
  --mat-file output/butterfly_4x4_rot180_target_16/butterfly_4x4_target.mat `
  --output-dir output/butterfly_4x4_m1_4_n1_4_main_total_2500 `
  --epochs 2400 --lr 5e-4 --seed 42 --channel-count 16 `
  --order-grid-size 4 --order-m-start 1 --order-n-start 1 `
  --image-loss-mode energy --device auto --log-interval 200 `
  --initial-results output/butterfly_4x4_m1_4_n1_4_main_100/optimized_results.npz
```

后者从100轮结果续跑2400轮，因此累计为2500轮。两段均为纯主图案损失，
所有CV、亮度、背景和结构辅助权重为0。

## 结果

| 指标 | 100轮 | 累计2500轮 |
|---|---:|---:|
| 结构相似度 mean/min | 0.2941/0.1649 | 0.6628/0.3558 |
| 前景覆盖率 mean/min | 0.0661/0.0505 | 0.4242/0.0844 |
| 三档比例 | 0.339:0.696:1 | 0.348:0.675:1 |
| 三档跨通道CV | 1.424/1.371/1.321 | 1.170/1.179/1.183 |
| 灰度单调通道 | 15/16 | 16/16 |
| 比例RMSE mean/max | 0.0777/0.3128 | 0.0131/0.0298 |

续跑用时579.4秒，总损失从续跑起点约`40.9843`降至`23.1739`。累计2500
轮统一2×曝光下，完整蝴蝶、四翼轮廓、翅脉、斑点、躯干和触角均可辨。
当前主要限制是图块亮度差异，而不是结构未形成。

### 2500轮前景/背景SNR复算

在不改变相位、不继续训练的条件下，使用原始强度补算前景/背景分离指标：

```powershell
python coding/python/evaluate_saved_results.py `
  --results-file output/butterfly_4x4_m1_4_n1_4_main_total_2500/optimized_results.npz `
  --output-dir output/butterfly_4x4_m1_4_n1_4_main_total_2500_snr_evaluation
```

16通道CNR mean/min为`3.331/0.404`，按2026-09-06前旧口径计算的SNR-like mean/min/max为
`5.23/-7.87/22.03 dB`，中位数为`2.88 dB`，其中10/16通道高于`0 dB`。
最弱通道是`(2,2)`，CNR为`0.404`、SNR为`-7.87 dB`；其次为
`(2,3)`、`(2,1)`、`(3,2)`和`(3,3)`，均低于`-2 dB`。SNR与结构相似度、
覆盖率的样本相关系数分别为`0.938`和`0.986`，说明它确实捕捉到了当前结果中
“目标前景是否压过背景散斑”的主要差异。

对角倍数组的SNR并非统一失败：`(1,1)/(2,2)/(3,3)/(4,4)`分别为
`-0.28/-7.87/-2.31/7.89 dB`。因此倍频耦合会限制独立分配，但本次低SNR还与
具体图块复杂度和亮度分配共同相关，不能仅凭倍数组标签解释。

本轮只把SNR加入评价，没有把它作为训练损失。原因是它与现有结构/覆盖率高度相关，
直接强压可能再次以牺牲结构或灰度比例换取单一数值改善。下一阶段若继续，应先从2500
轮相位热启动，用低权重、带结构回退门槛的前景—背景分离损失做短轮对照。

四个重点复杂源图块在2500轮时：

| 源图块→级次 | 结构 | 覆盖率 |
|---|---:|---:|
| (4,4)→(1,1) | 0.3558 | 0.1619 |
| (2,3)→(3,2) | 0.5246 | 0.1235 |
| (3,2)→(2,3) | 0.5287 | 0.1212 |
| (2,2)→(3,3) | 0.5398 | 0.1350 |

对角倍数组中的复杂图块仍形成了结构，但中心块比部分外侧图块暗，说明倍频
关系没有导致必然失败，却会限制独立亮度和质量分配。

## 输出与结论

- 目标：`output/butterfly_4x4_rot180_target_16`
- 物理验证：`output/butterfly_4x4_m1_4_n1_4_P2000_validation`
- 100轮结果：`output/butterfly_4x4_m1_4_n1_4_main_100`
- 累计2500轮结果：`output/butterfly_4x4_m1_4_n1_4_main_total_2500`
- 共用尺度2×对比：`output/butterfly_4x4_main_100_vs_total_2500_full_scene_2x`
- SNR复算：`output/butterfly_4x4_m1_4_n1_4_main_total_2500_snr_evaluation`

结论：该连续4×4级次组合值得保留。它满足16/16中心传播并能在纯主损失下
形成完整蝴蝶；下一步若继续，应从2500轮结果热启动，温和加入跨通道亮度一致
性和最弱通道约束，而不是重新改变级次或图块映射。
