# 公开 benchmark 微调与同标准评测

这组实验从 Walrus 基础权重出发，在认可度高、协议成熟、有多个模型公开结果的 CFD
benchmark 上各微调一次，并严格按原 benchmark 的划分、分辨率、输入步数和指标评测，
用来判断 BladeNet 上的微调结论是否可信。所有数据都不在 Walrus 的 19 个预训练数据集中
（`pretrained/walrus/extended_config.yaml`，论文 Table 6）。

| 名称 | 来源 | 类型 | 数据（只读） | 评测器 |
| --- | --- | --- | --- | --- |
| `airfoil` | Geo-FNO / Transolver | 稳态 | `infraExp/airfoil/naca` | `StaticBenchmarkTrainer` |
| `pipe` | Geo-FNO / Transolver | 稳态 | `infraExp/pipe` | `StaticBenchmarkTrainer` |
| `darcy` | FNO / Transolver | 稳态 | `infraExp/Darcy_421` | `StaticBenchmarkTrainer` |
| `ns2d` | FNO / Transolver | 时序 | `infraExp/NavierStokes_V1e-5_N1200_T20` | `RolloutBenchmarkTrainer` |
| `pdebench_cns_M{0.1,1.0}_eta{0.01,0.1}` | PDEBench | 时序 | `infraExp/pdebench/2D_CFD` | `RolloutBenchmarkTrainer` |

```bash
python scripts/prepare_benchmark.py --benchmark <名称>     # 缓存、统计与 runs/<名称>_setup/finetune.yaml
cd /WORK/PUBLIC/xuqy_work/run
BENCHMARK=<名称> sbatch -J Walrus<名称> runWalrusBenchmark.sh   # 输出 runs/<名称>_<jobid>/
```

`runWalrusAirfoil.sh`、`runWalrusPipe.sh` 是较早的专用脚本，与通用脚本做同样的事。

## 共同的微调配方

`scripts/prepare_benchmark.py` 以 `prepare_bladenet.build_config` 为基础：全模型微调、
AdamW 学习率 1e-5、weight decay 1e-4、batch 1、MAE 损失、BF16 训练、FP32 评测（关闭 TF32）、
梯度检查点，只保存 best 与 last。没有针对任何 benchmark 调参。

每个 2D 轴的编码器总步幅从 {2, 4} 中选使 token 数最接近 32 的一个（官方模型对 2D 网格的
目标）：Airfoil 221×51 → 56×13，Pipe 129² → 33²，Darcy 85² → 22²，64² 网格（NS、CNS）用
`[[2,1],[2,1]]` → 32²。网格不能整除时开启 `pad_to_patch_multiple`，在高端按边缘复制填充，
解码后裁回原网格，损失和指标只覆盖原始网格点。

## 稳态 benchmark（Airfoil、Pipe、Darcy）

与 BladeNet 相同的静态单步设置：输入是目标的训练集均值（归一化后为零）和常量场，
目标按训练集均值/标准差归一化，评测在物理单位下进行。默认 200 轮，预热 5 轮，cooldown 10 轮。

| 名称 | 划分（与原脚本一致） | 网格 | 输入 | 目标（字段名） |
| --- | --- | --- | --- | --- |
| airfoil | train `[0,1000)`，test `[1000,1200)` | C 网格 221×51 | 网格坐标 | Q 第 4 通道 Mach（`mach`，新字段） |
| pipe | train `[0,1000)`，test `[1000,1200)` | 结构网格 129×129 | 网格坐标 | Q 第 0 通道水平速度（`velocity_x`） |
| darcy | train smooth1 `[0,1000)`，test smooth2 `[0,200)`，每 5 点取 1 | 85×85 | 系数场与网格坐标 | 解，即压力（`pressure`） |

目标与预训练物理量一致时沿用预训练字段名，与 BladeNet 复用 `pressure`、`density` 相同。
原协议都没有验证集：Airfoil、Pipe 取协议未使用的样本 `[1200,1400)`，Darcy 取 smooth2 的
`[200,400)`，只用于监控和选最优检查点，不参与训练。

指标与 Transolver `TestLoss.rel` 相同：每个测试样本在全部网格点上计算
`||pred - target||_2 / ||target||_2`，再对样本取平均。同时给出逐样本相对 L1 平均和全局
ratio-of-sums 相对 L1。

Transolver 论文 Table 2 的该指标：

| 名称 | Transolver | GNOT | LSM | ONO | Geo-FNO / FNO |
| --- | ---: | ---: | ---: | ---: | ---: |
| airfoil | 0.0053 | 0.0076 | 0.0059 | 0.0061 | 0.0138 |
| pipe | 0.0033 | 0.0047 | 0.0050 | 0.0052 | 0.0067 |
| darcy | 0.0057 | 0.0105 | 0.0065 | 0.0076 | 0.0108 |

训练结束时测试集评测两次，结果在 `<run>/viz/<名称>/`：`test_epoch<N>_*` 为最后一轮权重
（主结果，与已发表基线口径一致），`test_best_*` 为验证集最优检查点。

### Airfoil、Pipe 的坐标输入

Airfoil、Pipe 只输入网格坐标，共四个常量通道（字段名以基准名为前缀）：全局归一化的 x、y
坐标，以及 `(坐标 - 该节点训练集均值) / 该节点训练集标准差` 的网格偏移。逐节点统计存于
`runs/<名称>_setup/mesh_statistics.npz`，哈希记录在 `normalization.json`。两个数据集所有
样本的 x 坐标都相同，形状只体现在 y 上。

偏移通道的原因：Airfoil 网格覆盖约 ±40 倍弦长，同一节点在不同样本间只移动约 0.02。
只用全局归一化坐标时，形状差异约为通道尺度的 0.3%（样本间逐节点标准差 0.0031），
与编码器在 BF16 autocast 下的量化误差（约 0.0005）同一量级。作业 565858 只用这两个坐标
通道，6 轮后验证 rel L2 停在 0.104，与"逐节点预测训练集平均 Mach 场"的平凡基线
（val 0.1002、test 0.1030）相当，因此提前终止。偏移通道的样本间差异约为 1。

## 时序 benchmark（NS、PDEBench CNS）

训练沿用官方模型自己的时间推进设置（从 `extended_config.yaml` 读取）：因果时间注意力、
预测增量（delta）、逐样本 RMS 归一化（`SamplewiseRevNormalization`）、周期边界下的 patch
jittering、6 帧上下文。训练样本是训练轨迹中"连续 6 帧 → 下一帧"的所有窗口。

评测严格复现原 benchmark 的自回归：给定前 10 帧（模型使用其中最后 6 帧），之后每一帧都由
模型自己的上一步输出递推得到，直到轨迹结束；指标在物理单位下对全部预测帧计算。patch
jittering 在推理时也会随机平移，评测固定随机种子（`evaluation_seed`，默认 0）。

| 名称 | 划分 | 网格 | 字段 | 给定帧 → 预测帧 | 指标 |
| --- | --- | --- | --- | --- | --- |
| ns2d | train `[0,1000)`，test 最后 200 条 `[1000,1200)` | 64² | 涡量（`vorticity`，新字段） | 0–9 → 10–19 | Transolver `test_l2_full`：逐样本对全部预测帧的 rel L2，取平均 |
| pdebench_cns_* | test 为文件前 10% `[0,1000)`，train `[1000,10000)` | 128² 每 2 点取 1 → 64² | `density`、`pressure`、`velocity_x`、`velocity_y` | 0–9 → 10–20 | PDEBench `metric_func` |

PDEBench 的设置来自官方 `config_2DCFD.yaml`（`reduced_resolution: 2`、`initial_step: 10`、
`t_train: 21`、`batch_size: 20`）与 `FNODatasetSingle`（`test_ratio=0.1`，前 10% 为测试）。
`metric_func` 从 PDEBench 移植（`walrus/trainer/rollout_benchmark_trainer.py`），测试中与原代码
逐项核对：RMSE、nRMSE、cRMSE、max error、bRMSE、fRMSE（低/中/高频）。

PDEBench 的 `metrics()` 按评测 batch（20 个样本，源文件顺序）累加 batch 均值后除以最后一个
batch 的下标（即 batch 数减一），所以官方报告值比 batch 平均偏大 50/49。结果同时保存：

- `pdebench_reported`：与官方代码一致，用于和论文表格对比；
- `batch_mean`：batch 平均（等于逐样本平均），训练中的监控也用它。

PDEBench 论文附录表 11、12 的 nRMSE（仅 U-Net、FNO；CNS 未报告 PINN）：

| 配置 | U-Net | FNO |
| --- | ---: | ---: |
| M=0.1，η=ζ=0.01 | 0.71 | 0.17 |
| M=0.1，η=ζ=0.1 | 5.1 | 0.36 |
| M=1.0，η=ζ=0.01 | 0.36 | 0.096 |
| M=1.0，η=ζ=0.1 | 0.92 | 0.098 |

NS 的 Transolver 论文 Table 2：Transolver 0.0900，ONO 0.1195，GNOT 0.1380，LSM 0.1535，FNO 0.1556。

两个 benchmark 都没有可用的验证集：训练轨迹以外的样本全部是测试集。验证 loader 对固定的
一部分训练轨迹做完整 rollout，只用于监控训练，不用于选择检查点；主结果为最后一轮权重的
测试结果（`test_epoch<N>_*`）。

训练预算按一步窗口计：A800 上每个 6 帧窗口约 1.3 秒（开启梯度检查点时；时序任务关闭
梯度检查点，显存约 23 GB 起），每个任务限制在 1–1.5 天：NS 每轮 14000 个窗口（全部训练
窗口）× 6 轮，CNS 每轮随机 9000 个窗口 × 10 轮，约 8–9 万个窗口，少于 Walrus 论文微调的
50 万。预热 1 轮，cooldown 2 轮。`scripts/prepare_benchmark.py --epochs/--samples-per-epoch`
可调整。FP32 评测时一批 20 条轨迹的一次前向约 21 秒，所以每轮只 rollout 20 条训练轨迹做
监控；最终测试覆盖完整测试集（NS 200 条约 35 分钟，CNS 1000 条约 3 小时）。

PDEBench 的 512² 文件没有公开的官方训练配置，论文也未说明其降采样，暂不纳入。

## 结果（2026-09-30）

全部为完整测试集、FP32 评测（关闭 TF32）。静态任务取独立验证集上的最优检查点（Airfoil、
Pipe 为第 200 轮，Darcy 为第 199 轮）；NS、CNS 没有验证集，取最后一轮。官方协议报告最后
一轮，两者只在 Darcy 上不同。下表对比数字取自 Transolver 论文 Table 2 与 PDEBench 论文
附录表 11、12；与 2023–2026 年方法的比较见“与近期方法的比较”。还没有同架构从零训练的对照。

| benchmark | 作业 | Walrus | 已发表最好 | 其他已发表 |
| --- | --- | ---: | ---: | --- |
| airfoil（rel L2） | 566129 | **0.00433** | Transolver 0.0053 | LSM 0.0059，ONO 0.0061，GNOT 0.0076，Geo-FNO 0.0138 |
| pipe（rel L2） | 567328、588603、588604 | 0.00342 ± 0.00006（3 次） | **Transolver 0.0033** | GNOT 0.0047，LSM 0.0050，ONO 0.0052，Geo-FNO 0.0067 |
| darcy（rel L2） | 569268 | **0.00465**（最后一轮 0.00476） | Transolver 0.0057 | LSM 0.0065，ONO 0.0076，GNOT 0.0105，FNO 0.0108 |
| ns2d（rel L2） | 569269 | **0.0777** | Transolver 0.0900 | ONO 0.1195，GNOT 0.1380，LSM 0.1535，FNO 0.1556 |
| CNS M=0.1 η=0.01（nRMSE） | 569270 | **0.0325** | FNO 0.17 | U-Net 0.71 |
| CNS M=0.1 η=0.1（nRMSE） | 569271 | **0.0212** | FNO 0.36 | U-Net 5.1 |
| CNS M=1.0 η=0.01（nRMSE） | 569272 | **0.0373** | FNO 0.096 | U-Net 0.36 |
| CNS M=1.0 η=0.1（nRMSE） | 569273 | **0.0282** | FNO 0.098 | U-Net 0.92 |

CNS 为与官方代码一致的 `pdebench_reported`；逐样本平均（`batch_mean`）依次为 0.0319、
0.0208、0.0366、0.0277。Airfoil、Pipe 的最优检查点就是第 200 轮，验证误差到最后仍在下降。

### Pipe 与 Transolver 的差距

三次独立微调（567328 种子未记录，588603、588604 为种子 1、2，配置只差种子）的测试误差为
0.003482、0.003354、0.003416，相对标准差 1.9%。以下分析基于 567328。
其平均 rel L2（0.00348）略高于 Transolver（0.0033），但逐样本中位数只有 0.00083：
平均值由 9 个样本决定（超过 0.02），它们 98%–100% 的误差平方都在出口段（最后 16 行，
`i >= 112`），前 112 行几乎没有误差。分两类：

- A 类（1072、1162、1152、1033、1036，误差 0.044–0.061）：管道在出口处有急转的 S 形弯，
  真实流速峰值在最后十几行内从一侧甩到另一侧，模型预测的峰值偏向另一侧。
- B 类（1065、1193、1123、1188）：出口靠 `j=128` 一侧有强倒流（u 低至 -0.86），最后一两行剧烈变化。

同样曲率的弯出现在管道中段时误差中位数只有 0.0007，出现在出口段时为 0.0080、最大 0.17；
入口端没有这种现象。出口段弯曲最急的 60 个训练样本，出口段误差中位数 0.0028、最大 0.09，
说明是欠拟合而非仅泛化问题。上下壁方向的填充不是原因（流量偏向未填充的 `j=0` 一侧时误差更大）。

诊断实验（作业 586204、586223，只作诊断，不替换正式结果）：从第 200 轮权重出发，以新的
AdamW、学习率 1e-5 分别用 MAE 和物理单位 rel L2（`PhysicalRelativeL2`）续训 20 轮。A 类误差在
三组中几乎不变，与损失函数和训练长度无关；B 类在 rel L2 下改善（0.016–0.023），在 MAE 下变差
（0.045–0.069）。两组的中位数误差都翻倍（约 0.0019）：重新启动优化器并把学习率从 1e-7 升回
1e-5 会破坏已收敛的模型，续训应保留优化器状态并用小学习率。

用官方 `Transolver_Pipe.sh` 配置在本地复现 Transolver（作业 591511，500 轮，同一测试集和
指标）：平均 0.004329，比论文值高 31%。A 类中 1162、1152、1033、1036 两个模型误差几乎相同
（0.04–0.06），说明是样本本身难，而不是 Walrus 的填充或表示问题；只有 1072 是 Walrus 独有
（三次平均 0.058，Transolver 0.0036）。去掉这 5 个样本后，Walrus 平均 0.0021，Transolver
0.0033；200 个样本中 Walrus 有 150 个误差更低。

### 与近期方法的比较

加入 2023–2026 年正式发表的方法（各自论文的主表数值）后，各列最好的是 LaMO（NS 0.0460、
Darcy 0.0039、Airfoil 0.0041）和 LRSA（Pipe 0.0023）。Walrus 在已发表方法中的名次：Airfoil
4/15、Darcy 6/15、Pipe 10/13、NS 8/15（NS 只输入 6 帧，参考方法 10 帧，仅作参考）；与各列
最好相差 +5.6%（Airfoil）到 +48.6%（Pipe）。结论：同协议下 Walrus 达到 Transolver 水平或
更好，但不是最优，计算代价高数倍（Pipe 训练约 30 小时，Transolver 约 5 小时）。

### 评测口径核对（2026-09-29）

逐项对照官方源码（thuml/Transolver 的 `exp_airfoil.py`、`exp_pipe.py`、`exp_darcy.py`、
`exp_ns.py`；PDEBench 的 `metrics.py`、`fno/utils.py`、`config_2DCFD.yaml`）：测试集、分辨率、
目标字段、指标公式与汇总方式均一致，指标都在物理量上计算。测试目标与按官方切片取出的
数组逐元素相同（Airfoil、Pipe、Darcy、NS 全部测试样本；CNS 每个工况抽查 5 条完整轨迹）。
与官方的差别只有：NS、CNS 只输入最后 6 帧（官方 10 帧，对 Walrus 更不利）；静态任务按独立
验证集选检查点（官方取最后一轮）。

## 兼容名称

作业 566129（Airfoil）与 567328（Pipe）启动时的配置引用了 `walrus.data.airfoil`、
`walrus.data.geofno.GeoFNONormalization`、`walrus.trainer.airfoil_trainer`、
`walrus.trainer.geofno_trainer`，这些名称保留为指向通用实现的别名。
