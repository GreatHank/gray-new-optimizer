# 李仲阳团队大视场论文实例参数复算

## 核对范围

本文核对 Zhe Zhang 等发表于 ACS Nano 的 *Breaking the Diffraction-Encoding Limit for High-Capacity Meta-Holography via Multiorder Decoupling*（DOI `10.1021/acsnano.6c04285`）及官方补充材料（DOI `10.1021/acsnano.6c04285.s001`）。所有数值分为“论文明确公开”“由公开参数复算”和“参数不足，不能严格复跑”三类。

## 结果总览

| 实例 | 论文明确公开的参数 | 论文FoV | 本项目复算 | 判断 |
|---|---|---:|---|---|
| 23通道级次解耦 | `lambda=488 nm`、`P_x=P_y=2000 nm`、`theta_i=46°`、`phi_i=0°`、空气；21个 `m=-3,-2,-1, n=-3…3` 加 `(0,2)、(0,1)` | Table S1约`137°` | 23/23中心传播；最小空气余量`0.127205`；中心射线最大夹角`102.085°`；把每级次扩展为完整局部方形角谱后，边界射线最大夹角约`143.071°` | 光栅方程严格复现；`137°`属于连续角谱覆盖口径，不是中心点夹角 |
| Sample 3时钟 | `P=1000 nm`、3×3共9个相邻级次、只使用红虚线内图案 | `87°` | 若按`488 nm`，3×3完整圆形标称FoV为`94.109°`；若按当前网站`480 nm`，为`92.109°` | 论文明确说明实际图案没有占满可编码区域，因此87°低于理论值 |
| 正文最大视场实例 | 正文摘要只明确给出最终结果 | `168°` | 未复跑 | 公开SI没有同时给齐波长、绝对级次集合和入射角，不能补造参数 |
| Figure S2近半球数值例 | `m,n∈{-1,0,1}`，混合迂回相位与PB相位 | 接近`180°` | 未严格复跑 | SI没有给齐`lambda/P`和入射角；当前仅`dx/dy`的结构也缺少PB旋转自由度 |

## 23通道严格复算

采用的23个级次为：

```text
m = -3, -2, -1；n = -3, -2, -1, 0, 1, 2, 3，共21项
再加 (0,2)、(0,1)
```

公开参数：

```text
lambda = 488 nm
P_x = P_y = 2000 nm
theta_i = 46 deg
phi_i = 0 deg
n_in = n_out = 1
NA = 1
```

复算结果：

- 级次间距 `Delta_u=0.244`。
- 23/23个中心均传播并被NA收集。
- 最小空气余量为`0.1272045`。
- `(-3,3)`出射极角`47.0635°`，论文Table S3为`47.06°`。
- `(-2,0)`出射极角`13.3760°`，论文为`13.38°`。
- `(-1,0)`出射极角`28.3815°`，论文为`28.38°`。
- `(0,2)`出射极角`60.3715°`，论文为`60.37°`。

这些逐级次角度一致，说明当前项目的二维光栅方程、入射角定义和符号约定正确。

论文Table S1报告该工作的FoV约为`137°`。只连接23个中心得到的最大射线夹角为`102.085°`，明显偏小；将每个级次恢复为宽度`Delta_u`的局部方形角谱后，完整方格边界给出的几何上限约为`143.071°`。论文值位于两者之间，符合“实际全息角谱没有填满全部方格”的情况。这也证明FoV不能只从级次中心估计。

补充材料内部有一处需要谨慎：Figure S5展示23个优化图案，当前复算按其级次集合；Table S3的排版却能读出25个级次，其中额外出现`(0,-1)`和`(0,-2)`，但标题仍写23。当前项目不把这两个表内额外项并入23通道实例。

## Sample 3时钟图

Figure S9明确说明：

- 周期为`1000 nm`；
- 完整钟表由9个相邻衍射级次级联；
- 红色虚线区域才是实际使用的全息图案；
- 实际FoV为`87°`，边缘平面投影畸变约`25%`。

该节没有写出工作波长、9个级次的绝对整数编号和共同入射角。因此只能复算相对3×3几何：

```text
lambda=488 nm时：Delta_u=0.488，完整3×3圆形标称FoV=94.10865°
lambda=480 nm时：Delta_u=0.480，完整3×3圆形标称FoV=92.10896°
```

`87°`对应的方向余弦直径为`1.376709`。它约占488 nm假设下完整3×3宽度的`94.04%`，或占480 nm假设下的`95.60%`。因此网站显示约92°或94°而论文写87°并不矛盾：网站报告完整方格可容纳的标称圆形FoV，论文报告实际使用的红虚线图案范围。

## 168°与近180°实例的边界

论文摘要报告最高`168°`，Figure S2报告混合PB相位后的接近`180°`数值重建。公开SI没有给齐这两个实例的完整`lambda、P、theta_i、phi_i、绝对级次/有效角谱范围`组合。S12虽提到`3000 nm`周期和`1000 nm`结构尺寸，但没有在该节明确把它们逐项绑定到168°实例，因此当前不把它们拼成所谓“严格复现参数”。

Figure S2还包含零级和四对正负共轭级次，依赖纳米结构旋转产生PB相位。当前方形纳米砖与`dx/dy`位移前向只能复算几何传播，不能复现该实例的独立图像编码。

## 复现命令与输出

汇总计算：

```powershell
python coding/python/reproduce_li_zhongyang_fov_examples.py `
  --output-dir output/li_zhongyang_paper_instances
```

23通道逐级次验证：

```powershell
python coding/python/validate_diffraction_orders.py `
  --order-preset paper-23 `
  --wavelengths-nm 488 `
  --period-x-nm 2000 --period-y-nm 2000 `
  --incident-theta-deg 46 --incident-azimuth-deg 0 `
  --n-in 1 --n-out 1 --na 1 `
  --output-dir output/li_zhongyang_paper_instances/paper23
```

主要结果：

- `output/li_zhongyang_paper_instances/fov_reproduction_summary.json`
- `output/li_zhongyang_paper_instances/paper23/diffraction_order_map.png`
- `output/li_zhongyang_paper_instances/paper23/selected_order_metrics.csv`
- `output/li_zhongyang_paper_instances/sample3_relative_3x3_488nm/diffraction_order_map.png`
