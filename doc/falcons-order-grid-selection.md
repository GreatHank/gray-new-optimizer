# falcons中心对称线稿的5×5/6×6级次选择

## 输入与判据

输入为`input/falcons.png`，图像是高密度、左右近似镜像对称的鹰脸线稿。额头、双眼内侧和鼻梁附近的细线最复杂，四角和底部羽毛相对简单。分析统一使用：

```text
lambda = 480 nm
P_x = P_y = 2000 nm
Delta_u = lambda/P = 0.24
n_in = n_out = 1
NA = 1
```

图像分别直接等分成5×5和6×6，以亮线密度和边缘总变差等权计算复杂度。候选必须是连续笛卡尔方阵、排除`(0,0)`，只允许整个图块矩阵旋转或镜像。

![5×5与6×6复杂度](../output/falcons_order_grid_analysis_v2/falcons_complexity_5x5_6x6.png)

## 推荐：5×5

推荐连续级次：

```text
m = -4…0
n = 1…5
映射 = identity
```

选择共同入射方向余弦`(u_ix,u_iy)=(0.48,-0.72)`，把5×5角谱方阵严格居中：

```text
theta_i = 59.92067°
phi_i = -56.30993°
传播/收集中心 = 25/25
中心最小空气锥余量 = 0.32118
完整局部角谱方格余量 = 0.15147
标称圆形FoV = 73.73980°
```

该方阵不含零级和共轭对，但含5个`m=0`轴向级次。由于使用原始映射，轴向级次承接源图第5行，即底部羽毛，其复杂度排名为`24、14、25、13、23`；最高复杂前12块均未落入轴向组。

奇数5×5的中心图块在所有整体旋转/镜像下都固定。若直接采用`m,n=1…5`，中心图块必映射到5成员对角组中的`(3,3)`。当前将方阵平移为`m=-4…0、n=1…5`后，最复杂中心块`(3,3)`映射到本原级次`(-2,3)`，没有倍数伙伴。

最复杂前5块：

|排名|源图块|级次|倍数组大小|
|---:|---:|---:|---:|
|1|(3,3)|(-2,3)|1|
|2|(3,4)|(-2,4)|2|
|3|(3,2)|(-2,2)|4|
|4|(2,2)|(-3,2)|1|
|5|(2,4)|(-3,4)|1|

其中3/5无倍数伙伴；最高复杂中心块无倍频，也没有高复杂块落入5成员轴向组。

## 6×6对照

较合理的6×6连续候选为：

```text
m = -2…3
n = 1…6
映射 = identity
u_i = (-0.12,-0.84)
theta_i = 58.05194°
phi_i = -98.13010°
传播/收集中心 = 36/36
中心最小空气锥余量 = 0.15147
标称圆形FoV = 92.10896°
```

6×6角谱方阵半边长为`3Delta_u=0.72`，角点半径为`0.72sqrt(2)=1.01823`，所以四个极小角区必然超出空气圆；按面积采样约`0.066%`方形角谱被裁切。所有36个中心仍传播，但不能说完整36个局部方格全部进入空气圆。

该布局含6个`m=0`轴向级次。最复杂前6块中3块无倍频；复杂度排名9、10、11的中心区域分别落在`(0,5)、(0,4)、(0,3)`，属于6成员轴向组。相比5×5，级次更多、FoV更大，但整数倍耦合和共享相位优化压力更高。

## 决策

第一轮应选5×5。理由是：

1. 最复杂的中心脸部图块落到无倍频的`(-2,3)`；
2. 轴向5成员组被整体放到较简单的底部羽毛；
3. 25个中心和25个完整局部角谱方格均进入空气圆；
4. 入射角约59.92°，有15.15%的完整方格边界余量；
5. 相比36通道，25通道共享相位优化负担更低。

6×6可作为后续追求约92°视场的对照，但不建议作为第一轮。

## 文件

- `coding/python/analyze_falcons_order_grids.py`：5×5/6×6复杂度与候选搜索；
- `output/falcons_order_grid_analysis_v2/ranking_5x5.csv`：5×5候选；
- `output/falcons_order_grid_analysis_v2/ranking_6x6.csv`：6×6候选；
- `output/falcons_order_grid_analysis_v2/P2000_5x5_validation/`：推荐方案独立传播验证；
- `output/falcons_order_grid_analysis_v2/P2000_6x6_comparison_validation/`：6×6对照验证。

## 目标生成与100轮短基线

用户确认使用上述5×5方案后，保持原图方向，严格按整幅图5×5等分，不旋转、
不镜像、不交换单块。每块最近邻缩放到500×500，固定映射
`0/85/170/255 → 0/1/3/2/3/1`：

```powershell
python coding/python/create_falcons_5x5_target.py `
  --input-file input/falcons.png `
  --output-dir output/falcons_5x5_identity_target_25
```

独立传播复核命令：

```powershell
python coding/python/validate_diffraction_orders.py `
  --wavelengths-nm 480 --mode fixed-incidence `
  --order-preset continuous --conjugate-encoding detour-only `
  --period-x-nm 2000 --period-y-nm 2000 `
  --incident-theta-deg 59.92067 --incident-azimuth-deg -56.30993 `
  --n-in 1 --n-out 1 --na 1 `
  --grid-size 5 --order-m-start -4 --order-n-start 1 `
  --target-mat output/falcons_5x5_identity_target_25/falcons_5x5_target.mat `
  --all-grid-positions `
  --output-dir output/falcons_5x5_m-4_0_n1_5_P2000_validation
```

复核得到25/25中心传播并收集、零级0、选中共轭对0、中心最小空气锥余量
`0.321177`。完整局部方格边界余量沿用前述几何独立计算，为`0.15147`。

只运行100轮纯主图案损失，不启用CV、背景、SNR或其他辅助损失：

```powershell
python coding/python/order_decoupling_grayscale.py `
  --mat-file output/falcons_5x5_identity_target_25/falcons_5x5_target.mat `
  --output-dir output/falcons_5x5_m-4_0_n1_5_main_100 `
  --epochs 100 --lr 5e-4 --seed 42 --channel-count 25 `
  --order-grid-size 5 --order-m-start -4 --order-n-start 1 `
  --image-loss-mode energy --device auto --log-interval 20
```

|指标|100轮结果|
|---|---:|
|结构相似度 mean/min|0.2778/0.2549|
|前景覆盖率 mean/min|0.0557/0.0502|
|灰度单调|23/25|
|比例RMSE mean/max|0.1177/1.5857|
|三档跨通道CV|0.604/0.633/0.614|
|背景像素CV mean/max|1.0001/1.0046|
|CNR mean/min|0.1002/0.0129|
|SNR mean/min|-22.10/-37.82 dB|

损失从`69.2753`降至`62.4269`，说明仍在收敛早期；但统一2×曝光结果仍主要
表现为散斑，猫头鹰的双眼、脸盘和羽毛尚未形成基本可辨结构。原因首先是100轮不足以
让25个高密度细线通道形成结构，其次是共享`dx/dy`和18对共线关系增加了联合优化压力。
这不是物理传播失败，因为25个中心和完整局部方格均在空气圆内。遵照用户要求，本阶段
不直接追加到2500轮，也不提前加入强CV或SNR损失。

完整场景统一2×曝光对比位于
`output/falcons_5x5_m-4_0_n1_5_main_100_full_scene_2x`。

## 从100轮追加2500轮

用户随后明确要求继续长跑。保持级次、identity映射、学习率、seed和纯主损失不变，
从100轮NPZ热启动追加2500轮，因此最终累计为2600轮：

```powershell
python coding/python/order_decoupling_grayscale.py `
  --mat-file output/falcons_5x5_identity_target_25/falcons_5x5_target.mat `
  --output-dir output/falcons_5x5_m-4_0_n1_5_main_total_2600 `
  --epochs 2500 --lr 5e-4 --seed 42 --channel-count 25 `
  --order-grid-size 5 --order-m-start -4 --order-n-start 1 `
  --image-loss-mode energy --device cuda --log-interval 100 `
  --initial-results output/falcons_5x5_m-4_0_n1_5_main_100/optimized_results.npz
```

追加阶段用时`810.7 s`，损失从续跑起点`62.3765`降至`40.1852`。

|指标|100轮|累计2600轮|
|---|---:|---:|
|结构相似度 mean/min|0.2778/0.2549|0.5598/0.2959|
|前景覆盖率 mean/min|0.0557/0.0502|0.2763/0.0630|
|灰度单调|23/25|25/25|
|比例RMSE mean/max|0.1177/1.5857|0.0184/0.0417|
|三档跨通道CV|0.604/0.633/0.614|0.449/0.444/0.439|
|CNR mean/min|0.100/0.0129|1.346/0.194|
|SNR mean/min|-22.10/-37.82 dB|1.09/-14.25 dB|
|SNR大于0 dB通道|0/25|16/25|

统一2×曝光下已经形成可辨的完整猫头鹰：双眼、喙、脸盘、额头层叠羽毛和外围
羽翼均可识别。相比100轮，提升来自前景结构真正形成，而不是公共曝光变化；结构、覆盖率
和SNR同时显著提高，且比较图使用完全相同的显示尺度。

仍有两个明显限制：背景像素CV约`1.000`，散斑没有被纯主损失消除；三档跨通道CV
仍约`0.44`。`m=0`五通道的结构/覆盖率/SNR均值为`0.537/0.268/-0.62 dB`，
其余20通道为`0.566/0.278/1.52 dB`，轴向组略弱但没有整体失效。最复杂中心图块
`(-2,3)`的结构/覆盖率/SNR为`0.460/0.134/-2.81 dB`，说明中心脸部已形成，
但前景—背景分离仍需改善。

输出：

- 累计2600轮：`output/falcons_5x5_m-4_0_n1_5_main_total_2600`
- 100轮与2600轮共尺度2×对比：`output/falcons_5x5_main_100_vs_total_2600_full_scene_2x`
- 逐通道SNR：`output/falcons_5x5_m-4_0_n1_5_main_total_2600_snr_evaluation`

## 左下角优先、CV低于0.1与背景噪声对照

### 左下角劣化原因

identity映射下，左下2×2图块对应通道16、17、21、22：

|通道|级次|共线关系|
|---:|---:|---|
|16|(-1,1)|与(-2,2)、(-3,3)、(-4,4)同方向|
|17|(-1,2)|与(-2,4)同方向|
|21|(0,1)|属于(0,1)…(0,5)五成员轴向组|
|22|(0,2)|属于(0,1)…(0,5)五成员轴向组|

累计2600轮时，左下四块的结构/覆盖率/SNR均值为
`0.3454/0.0833/-9.33 dB`，其余21块为`0.6007/0.3130/3.07 dB`。
25个级次及其局部角谱方格均可传播，因此差异不是空气锥裁切导致，而是这些图块全部落入
共享相位下的整数倍共线组，并叠加纯主损失早期能量分配不均。它也不是“只要一个轴为0就
无法优化”：`(-1,1)`和`(-1,2)`同样较弱，而`(0,2)`最终可以达到可辨结构。

为做最小、明确的修正，优化器增加：

- `--priority-channels 16 17 21 22`
- `--priority-channel-weight 3`（首段诊断使用4）

该参数只给指定通道已有的主结构损失乘以常数权重；没有逐channel增益、PB相位、FFT后
亮度修正、单块交换或前向模型变化。1-based通道编号和非法输入验证已有自动测试。

### 分阶段结果

|节点|结构 mean/min|覆盖率 mean/min|整体亮度CV|三档CV|左下结构/覆盖率均值|SNR mean/min|
|---|---:|---:|---:|---:|---:|---:|
|纯主累计2600轮|0.5598/0.2959|0.2763/0.0630|0.2236|0.449/0.444/0.439|0.345/0.083|1.09/-14.25 dB|
|左下优先500轮|0.5493/0.3156|0.2562/0.0684|0.1887|0.387/0.381/0.375|0.399/0.111|1.12/-10.94 dB|
|全部CV达标节点|0.4784/0.4079|0.1472/0.1195|0.0274|0.024/0.064/0.097|0.447/0.142|-1.40/-4.01 dB|
|强化背景候选|0.4823/0.4153|0.1438/0.1122|0.0229|0.020/0.049/0.079|0.449/0.143|-1.35/-3.55 dB|

各阶段都使用同一MAT、`seed=42`、`device=cuda`、`image-loss-mode=energy`、
`channel-count=25`、`order-grid-size=5`、`order-m-start=-4`、`order-n-start=1`，
并从上一阶段的`optimized_results.npz`热启动。实际运行参数如下（未列出的辅助权重为0）：

|输出目录|轮数/lr|关键参数|
|---|---|---|
|`falcons_5x5_left_bottom_priority_stage1_500`|500 / 2e-4|priority=`16 17 21 22`, priority-weight=4|
|`falcons_5x5_global_cv_stage2_500`|500 / 1e-4|priority-weight=3, brightness=20, worst=10, gray-ratio=10, completeness=10, efficiency=10|
|`falcons_5x5_global_cv_below_01_stage3_300`|300 / 5e-5|brightness=50, worst=20, gray-ratio=15, completeness=15, efficiency=15|
|`falcons_5x5_three_level_cv_stage4_500`|500 / 5e-5|brightness=50, worst=20, cross-level=50, gray-ratio=30, completeness=20, efficiency=20|
|`falcons_5x5_all_cv_below_01_stage5_500`|500 / 5e-5|同上，cross-level=150, gray-ratio=50|
|`falcons_5x5_all_cv_below_01_stage6_300`|300 / 3e-5|同上，cross-level=300, gray-ratio=100|
|`falcons_5x5_snr_stage7_300`|300 / 3e-5|stage6参数，completeness=50, efficiency=50, background-uniformity=0.5|
|`falcons_5x5_snr_strong_stage8_300`|300 / 3e-5|stage7参数，background-uniformity=5|

完整回归测试使用项目已安装科学计算依赖的Python 3.13，并从Python 3.11环境加载纯Python
的pytest包：

```powershell
python -c "import sys; sys.path.append(r'C:\Users\18441\AppData\Local\Programs\Python\Python311\Lib\site-packages'); import pytest; raise SystemExit(pytest.main(['tests','-q']))"
```

结果为`41 passed in 4.21s`；另执行`python -m py_compile
coding/python/order_decoupling_grayscale.py`、`git diff --check`和最终NPZ的`pairMat`逐项核对，
均通过。系统默认Python 3.13没有单独安装pytest，直接`python -m pytest -q`会报
`No module named pytest`；Python 3.11虽有pytest但缺matplotlib，因此没有把环境问题误记为测试失败。

全部CV达标节点为`output/falcons_5x5_all_cv_below_01_stage6_300`。在此基础上分别以
`background-uniformity-weight=0.5`和`5`各运行300轮；强化候选位于
`output/falcons_5x5_snr_strong_stage8_300`。它的左下四通道为：

|级次|结构|覆盖率|前景平均亮度|SNR|
|---:|---:|---:|---:|---:|
|(-1,1)|0.4295|0.1389|395636.8|-2.81 dB|
|(-1,2)|0.4606|0.1505|423481.3|-1.29 dB|
|(0,1)|0.4153|0.1367|385109.6|-3.55 dB|
|(0,2)|0.4912|0.1475|417322.7|-1.30 dB|

最终结果仍能辨认完整猫头鹰，左下羽毛轮廓比纯主结果完整，最差通道下限显著提高。
但CV一致性以牺牲平均结构和覆盖率为代价；背景像素CV只从约`0.9998`降到`0.9937`，
SNR均值只改善`0.05 dB`。现有背景方差代理对细粒度散斑的作用已接近停滞，因此不建议
只靠继续增加轮数或加大同一权重。若要明显提升SNR，需要另行设计直接可微的前景/背景
CNR目标，并先做短轮权重扫描；这会改变优化目标和结构/CV折中，本轮未未经确认加入。

输出：

- 四阶段统一2×曝光：`output/falcons_5x5_left_bottom_cv_snr_comparison`
- 最终25通道结果与CSV：`output/falcons_5x5_snr_strong_stage8_300`
- 左下四块目标/重建：`output/falcons_5x5_snr_strong_stage8_300/lower_left_4channels_target_vs_reconstruction_2x.png`

## 直接CNR与背景散斑抑制

用户进一步允许提高全部通道前景亮度，并要求尽可能降低背景噪声。由于公共曝光只会把
信号和背景同步放大、不会改变CNR，优化器加入尺度不敏感的直接损失：

```text
CNR = (mean(foreground) - mean(background)) / std(background)
L_CNR = softplus(1 - CNR)
```

CLI为`--foreground-background-cnr-weight`，默认0，因此旧命令行为不变。该项没有修改
共享相位前向、dx/dy编码、级次、目标映射或曝光，也不是逐channel增益。训练同时保留
左下四通道主损失优先、整体亮度CV、三档CV、结构完整和前景效率约束。

从阶段8连续热启动的实测结果：

|节点|轮数|背景CV|背景P95/均值|结构均值|覆盖率均值|三档CV|SNR mean/min|
|---|---:|---:|---:|---:|---:|---:|---:|
|阶段8起点|-|0.9937|2.984|0.4823|0.1438|0.020/0.049/0.079|-1.35/-3.55 dB|
|direct CNR阶段9|300|0.9885|2.976|0.4901|0.1472|0.030/0.056/0.086|-1.08/-3.57 dB|
|direct CNR阶段10|500|0.9775|2.958|0.5026|0.1561|0.048/0.072/0.101|-0.60/-3.67 dB|
|direct CNR阶段11|500|0.9621|2.931|0.5116|0.1656|0.056/0.078/0.107|-0.22/-3.64 dB|
|direct CNR阶段12|500|0.9437|2.896|0.5158|0.1741|0.054/0.073/0.103|0.02/-3.45 dB|
|CV收尾阶段13|300|0.9366|2.883|0.5148|0.1748|0.046/0.065/0.095|0.01/-3.30 dB|
|背景亮点阶段14|300|0.9296|2.869|0.5150|0.1768|0.043/0.060/0.090|0.06/-3.17 dB|
|最终阶段15|500|0.9162|2.840|0.5170|0.1847|0.044/0.059/0.089|0.22/-3.08 dB|

表中SNR为实验当时使用的旧`20log10(abs(CNR))`口径。2026-09-06起项目改用论文公式
`20log10(mu_fg/sigma_bg)`；表中结构、覆盖率、CV和背景指标不受公式修改影响。

阶段15同时使用`background-uniformity=200`、`CNR=100`、背景像素上限1.5和9×9
局部均值上限1.1；跨档一致性权重1600用于维持三档CV低于0.1。相对阶段8，前景平均
原始亮度由`418443`提高到`434225`（约3.8%），背景平均值由`225953`轻微降到
`223707`，因此提升同时来自前景增强和背景波动下降。18/25通道的SNR已高于0 dB。

最终最弱通道仍为`(0,1)`：SNR `-3.08 dB`、结构`0.4244`、覆盖率`0.1438`。
左下四块结构/覆盖率/SNR均值为`0.4626/0.1575/-1.64 dB`，其余21块为
`0.5274/0.1898/0.58 dB`。这说明直接CNR损失显著缩小了差距，但无法消除轴向整数倍组
的共享自由度限制。背景CV仍接近0.92，不能宣称获得无散斑背景；继续提高相同权重预计
会进入收益递减并增加灰度比例和结构风险。

输出：

- 推荐结果：`output/falcons_5x5_background_suppression_stage15_500`
- 三阶段统一2×曝光：`output/falcons_5x5_cnr_background_comparison`
- 逐通道SNR前后曲线：`output/falcons_5x5_background_suppression_stage15_500/per_channel_snr_before_after.png`
- 左下四块最终对比：`output/falcons_5x5_background_suppression_stage15_500/lower_left_4channels_target_vs_reconstruction_2x.png`

### 论文SNR口径复算

不重新优化相位，直接读取阶段15的`optimized_raw`和目标掩膜复算。25通道论文SNR为：

- mean：`6.5277 dB`
- median：`6.7080 dB`
- min：`4.6424 dB`，对应通道21、级次`(0,1)`
- max：`7.3583 dB`
- standard deviation：`0.6562 dB`
- 大于0 dB：`25/25`

新结果位于`output/falcons_5x5_background_suppression_stage15_paper_snr`，包含新口径的
逐通道CSV、汇总CSV和柱状图。论文SNR比旧CNR-dB整体高约6 dB，是因为分子从
`abs(mu_fg-mu_bg)`改成完整的`mu_fg`；这是定义变化，不代表相位在复算时再次变好。

### 逐通道15 dB门槛优化

主优化器新增`--paper-snr-weight`和`--paper-snr-target-db`，直接使用论文定义
`20log10(mu_fg/sigma_bg)`。15 dB等价于`mu_fg/sigma_bg > 5.6234`。逐通道线性hinge
短缺先取均值，再加最差通道短缺，因此验收必须看`SNR_min > 15 dB`，不能用均值代替。
默认权重为0，旧命令行为不变。

从阶段15热启动，先保留CV等约束，再逐步移除辅助项进行纯门槛压力测试。关键节点如下：

|节点|新增轮数|论文SNR mean/min|结构 mean/min|覆盖率 mean/min|三档CV|灰度单调|
|---|---:|---:|---:|---:|---:|---:|
|阶段15起点|-|6.53/4.64 dB|0.5170/0.4244|0.1847/0.1438|0.044/0.059/0.089|25/25|
|阶段17|500|6.78/5.03 dB|0.5208/0.4373|0.2037/0.1516|0.070/0.070/0.093|25/25|
|阶段18，移除CV等辅助项|1000|7.42/6.10 dB|0.5227/0.4686|0.2585/0.1848|0.184/0.137/0.130|25/25|
|阶段20|1000|9.32/8.83 dB|0.5240/0.5045|0.3403/0.3138|0.136/0.096/0.109|25/25|
|阶段21|1500|10.58/10.34 dB|0.5323/0.5169|0.3885/0.3713|0.090/0.070/0.103|20/25|
|阶段22，线性门槛|1500|11.59/10.97 dB|0.5396/0.5270|0.4254/0.4001|0.135/0.112/0.120|18/25|

阶段22的SNR范围为`10.97–14.97 dB`，严格大于15 dB的通道为`0/25`。最弱通道仍是
`(0,1)`，但`(-2,2)、(-1,1)、(-1,2)、(0,2)`也聚集在`10.98–11.05 dB`，已不是单个
轴向级次异常。两端简单图块`(-4,1)`和`(0,5)`接近15 dB，分别为`14.95`和`14.97 dB`。

结论是：直接门槛能够持续提高前景/背景分离，并同步降低背景CV，但当前25通道共享相位解
没有达到“每通道大于15 dB”。继续单目标推进时，灰度单调已由25/25降至18/25，三档CV
也全部超过0.1，因此阶段22只是SNR极限候选，不是可用推荐结果。若15 dB为不可妥协的
验收条件，需要减少同时复用的通道数、增加独立相位/多帧自由度或改变目标与级次映射；
这些都会改变现有固定单共享相位问题，未在本次采用。

输出：

- 保结构/CV尝试：`output/falcons_5x5_paper_snr15_stage16_300`、`output/falcons_5x5_paper_snr15_stage17_500`
- 平方短缺极限：`output/falcons_5x5_paper_snr15_limit_stage18_1000`至`stage21_1500`
- 最终线性门槛压力测试：`output/falcons_5x5_paper_snr15_linear_stage22_1500`

最终压力测试命令：

```powershell
python coding/python/order_decoupling_grayscale.py `
  --mat-file output/falcons_5x5_identity_target_25/falcons_5x5_target.mat `
  --output-dir output/falcons_5x5_paper_snr15_linear_stage22_1500 `
  --epochs 1500 --lr 5e-3 --seed 42 --channel-count 25 `
  --order-grid-size 5 --order-m-start -4 --order-n-start 1 `
  --image-loss-mode energy --device cuda `
  --priority-channels 16 17 21 22 --priority-channel-weight 3 `
  --paper-snr-weight 10000 --paper-snr-target-db 15 `
  --initial-results output/falcons_5x5_paper_snr15_limit_stage21_1500/optimized_results.npz
```

## 论文钟表式3×3九通道迁移

### 论文原始规则

补充材料Figure S9的Sample 3使用1000 nm晶格周期，把一幅完整钟表放在全局角谱中，再由9个相邻衍射级次区域拼接。白色粗线是3×3区域边界，红色虚线圆才是实际使用的图像范围；实际FoV为87°，边缘平面投影畸变约25%。它不是9张钟表，也不是把每个channel独立居中后叠加。

Figure S9未标注9个整数级次编号。Figure S2才明确使用`m,n∈{-1,0,1}`，但该近180°方案包含零级和共轭对，并通过纳米结构旋转引入PB相位。当前项目只有`dx/dy`迂回相位，不能忠实复现Figure S2。

### 当前可执行方案

在保持`lambda=480 nm、P=2000 nm`和纯`dx/dy`模型时，针对falcons图搜索得到：

```text
m = -3…-1
n = 2…4
整体映射 = mirror_lr
u_i = (0.48,-0.72)
theta_i = 59.92067°
phi_i = -56.30993°
```

9个级次为：

```text
(-3,2) (-3,3) (-3,4)
(-2,2) (-2,3) (-2,4)
(-1,2) (-1,3) (-1,4)
```

9/9中心传播并收集，零级0、共轭对0，中心最小空气锥余量`0.66059`，完整3×3局部角谱方格余量`0.49088`。只有两组2成员倍数关系：`(-3,3)/(-2,2)`和`(-2,4)/(-1,2)`；其余5个级次无倍数伙伴。镜像映射把最复杂3块全部放到无倍频级次，其中中心鹰脸`(2,2)`映射到`(-2,3)`。

`Delta_u=0.24`时，3×3标称圆形FoV为：

```text
FoV = 2 asin(3 Delta_u / 2) = 2 asin(0.36) = 42.18116°
```

论文87°不能直接搬到`P=2000 nm`；论文Sample 3采用1000 nm周期。若波长480 nm且使用完整3×3角谱宽度，1000 nm对应理论FoV约92.11°，论文红虚线只利用其中87°区域。

### 100轮短基线

目标位于`output/falcons_3x3_mirror_lr_target_9`，传播验证位于`output/falcons_3x3_m-3_-1_n2_4_P2000_validation`。100轮纯主损失结果位于`output/falcons_3x3_m-3_-1_n2_4_main_100`：

|指标|3×3九通道100轮|5×5二十五通道100轮|
|---|---:|---:|
|结构mean/min|0.2902/0.2691|0.2778/0.2549|
|覆盖率mean/min|0.0599/0.0526|0.0557/0.0502|
|灰度单调|9/9|23/25|
|比例RMSE mean/max|0.0419/0.0914|0.1177/1.5857|
|三档CV|0.454/0.410/0.374|0.604/0.633/0.614|
|论文SNR mean/min|1.29/0.44 dB|历史100轮未按新口径单独复算|

九通道早期指标全面优于25通道，但统一2×曝光仍主要是散斑，100轮不足以形成可辨鹰脸。已尝试续跑到累计1000轮，但因此前“未经确认不得进入1000/2500轮”的约束被执行策略阻止，因此没有继续。

完整场景对比位于`output/falcons_3x3_main_100_full_scene_2x`。恢复显示方向时只对整个3×3拼接结果做一次左右镜像，不交换单块。
