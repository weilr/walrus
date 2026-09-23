# BladeNet 单帧稳态场微调

这条训练路径把官方 Walrus base 模型迁移到“叶片几何与已知工况 → 稳态流场”。
时间长度固定为 1，不把空间切片当作时间，也不使用真实目标场作为输入。
预处理和检查命令不会启动训练或更新模型参数。

## 数据约定

默认数据目录：
`/WORK/PUBLIC/xuqy_work/bladenet/bladenet_grid_webdataset_256_32_32_metrics`。
直接读取现有 `train.tar`、`val.tar`、`test.tar` 中的 NPZ，不改写源数据，
也不需要安装 TurbineBladeNet 的 PyVista/WebDataset 依赖。

实际空间尺寸为 `256×64×32`：两个 `256×32×32` block 沿第二个计算轴拼接。
训练、验证、测试分别有 1279、274、275 个样本。沿用这些划分用于与原工程比较；
同一叶型的不同工况可以跨 split，所以这不是严格的未见叶型泛化测试。

输出顺序固定为 `pressure, temperature, mach, density`。输入包括：

- 坐标中心化后的三个分量。
- `sdf`、`wall_mask`、三个法向，以及九个 `metrics_*`：14 个逐点特征。
- 六个已知工况：`inlet_total_pressure_p01`、`inlet_static_pressure_p1`、
  `inlet_temperature_t1`、`velocity_y`、`velocity_z`、`outlet_static_pressure_p2`。

六个工况从 `coefficients.csv` 按 `design_id` 读取原值，再按训练集统计归一化，
不使用 NPZ 中已经按全表归一化的 `features` 对象，也不输入求解后的出口统计量。
额外常量字段有独立的 `bladenet_*` 名称，避免与预训练速度等动态字段混淆。
`wall_mask` 只表示壁面，不触发 Walrus 对通用 `mask` 字段的清零行为。

数据适配后的 shape：

```text
input_fields     [B, 1, 256, 64, 32, 4]   固定训练均值场，归一化后为零
constant_fields  [B,    256, 64, 32, 23]  归一化的几何与工况
output_fields    [B, 1, 256, 64, 32, 4]   原始物理量目标
```

目标场采用训练集的固定逐字段归一化；训练损失作用于归一化目标，验证输出会
还原为物理单位，再按下文的 TurbineBladeNet 全场评测协议计算指标。
最佳检查点按四个变量各自的全局相对 L1 再取算术平均选择。
这个平均分是本适配器明确定义的选优分数，不等同于 TurbineBladeNet 的复合 val_loss。
首版使用等权归一化 MAE，没有启用 TurbineBladeNet 的壁面加权或额外物理约束损失。
使用当前同一数据划分对比时，应同时说明损失和归一化方案的差别。

## 准备

在项目根目录、已激活的 `walrus` 环境中执行：

```bash
python scripts/prepare_bladenet.py \
  --data-path /WORK/PUBLIC/xuqy_work/bladenet/bladenet_grid_webdataset_256_32_32_metrics
```

默认生成 `runs/bladenet_setup/` 下的 tar 索引、`normalization.json`、
`manifest.json` 和 `finetune.yaml`。默认统计采用固定种子选择的 64 个训练样本，
空间统计步幅为 `4 2 2`；这是训练集抽样统计，未使用验证或测试数据。
若要覆盖全部训练样本及全部网格点，可执行：

```bash
python scripts/prepare_bladenet.py \
  --data-path /WORK/PUBLIC/xuqy_work/bladenet/bladenet_grid_webdataset_256_32_32_metrics \
  --stats-samples 0 --stats-stride 1 1 1 --force
```

变更数据位置、统计选项后应重新准备缓存；生成的配置包含这台机器上的绝对路径。
`--pretrained-config` 与 `--checkpoint` 可指定官方 base 配置和权重，默认读取
`pretrained/walrus/extended_config.yaml` 与 `pretrained/walrus/walrus.pt`。

## 训练前检查

```bash
python scripts/check_bladenet_setup.py \
  --config runs/bladenet_setup/finetune.yaml \
  --check-checkpoint \
  --report runs/bladenet_setup/preflight.json
```

检查包括真实数据接口、小模型的单帧前向/反向、物理单位输出，以及官方大模型
权重严格加载。小模型使用降采样网格；这不等同于官方大模型在完整网格上的 GPU
训练验证。检查不调用 `optimizer.step()`，不启动 epoch，不生成训练检查点。

## 启动训练

在有可用 CUDA 的计算节点上执行：

```bash
bash walrus/run_scripts/finetune_bladenet.sh
```

默认是单 GPU、batch size 1、全模型微调、学习率 `1e-5`、50 epoch、BF16 AMP、
梯度检查点，以及完整原生网格。评测默认关闭 AMP、使用严格 FP32，并在评测范围内
关闭 TF32；结束后恢复调用方的精度设置，与训练精度分开。
可通过 Hydra 覆盖，例如：

```bash
bash walrus/run_scripts/finetune_bladenet.sh \
  runs/bladenet_setup/finetune.yaml \
  trainer.max_epoch=30 optimizer.lr=5e-6
```

默认输出目录为 `runs/bladenet_finetune/`。如果进行不同实验，在准备阶段指定新的
`--run-dir`，避免与原有检查点混用。启动脚本在该目录已有检查点时退出，不会隐式
恢复旧实验。直接调用 `walrus.train` 仍沿用上游检查点恢复规则。
启动脚本还会在 CUDA 不可用或设备不支持配置中的 BF16 时退出。
正式训练显存需求需在目标 GPU 上测量；CPU 检查不代表完整模型一定能放入该显卡。

### Slurm：a01 分区

提交入口放在 `/WORK/PUBLIC/xuqy_work/run/`，与其他项目的提交脚本放在一起：

```bash
cd /WORK/PUBLIC/xuqy_work/run
sbatch runWalrusBladeNet.sh
```

资源为 a01 分区、1 个节点、1 个任务、1 张 GPU、8 个 CPU 核、96 GB 主机内存，
时间上限 48 小时。`a01` 是分区名；具体节点由调度器分配。脚本激活共享 Miniconda
中的 `walrus` 环境，保留 Slurm 分配的 `CUDA_VISIBLE_DEVICES`，使用单进程训练入口。
`run/runWalrusBladeNet.sh` 集中展示 SBATCH、环境变量、Conda 激活、工作目录及 GPU
启动信息，然后直接调用项目内的 `walrus/run_scripts/finetune_bladenet.sh`。
仓库中的 `walrus/run_scripts/runWalrusBladeNet_a01.sh` 保留同内容模板。

脚本按 run 目录的结构编写：SBATCH → 必要环境变量 → Conda 激活 → 项目路径 →
节点/GPU 信息 → 调用训练入口 → 输出退出码。只设置本项目的 Python 路径、实时日志、
Hydra 错误信息、CUDA 分配器以及 OMP/MKL/OpenBLAS 的 `1/1/1` 线程限制。
没有复制其他项目的 W&B 账户、项目名或分布式通信配置；当前配置保持 `logger.wandb=false`。
此前单轮测试使用 OMP/MKL 为 8，耗时不能直接当作当前单线程配置的实测值。

默认读取 `runs/bladenet_setup/finetune.yaml`（当前为 50 epoch），每个作业输出到
`runs/bladenet_a01_<jobid>/`，使用已有的 TurbineBladeNet 全场评测协议。日志写到
`/WORK/PUBLIC/xuqy_work/run/log/WalrusBladeNet/WalrusBladeNet-<jobid>.out` 与 `.err`；
日志父目录必须在提交前存在，安装提交脚本时已创建。

例如提交一轮测试，或调整学习率：

```bash
sbatch /WORK/PUBLIC/xuqy_work/run/runWalrusBladeNet.sh trainer.max_epoch=1
sbatch /WORK/PUBLIC/xuqy_work/run/runWalrusBladeNet.sh optimizer.lr=5e-6
```

仅检查语法和 Slurm 资源请求、不提交或启动训练：

```bash
bash -n /WORK/PUBLIC/xuqy_work/run/runWalrusBladeNet.sh
sbatch --test-only /WORK/PUBLIC/xuqy_work/run/runWalrusBladeNet.sh
```

需要使用另一份已准备配置时，修改提交脚本中的 `CONFIG_PATH`；修改 `RUN_DIR` 可指定
新的输出目录。当前脚本用于从官方 base 开始新实验，不用于隐式续训已有检查点。

模型使用显式两阶段步幅 `[[4,4],[2,2],[2,2]]`，总下采样倍数为 `(16,4,4)`，
绕过原自动策略不支持最后一轴长度 32 的问题。时间因果预测、时间 rollout、
随机 patch 位移和 Well 的旋转增强在此任务中关闭。边界编码只用于模型的数值
padding，不能视为已经实现 CFD 壁面、入口、出口条件的硬约束。

### 长训练与续训：runWalrusBladeNetLong.sh

`/WORK/PUBLIC/xuqy_work/run/runWalrusBladeNetLong.sh` 在同一份准备配置上通过 Hydra
覆盖启动长训练：默认 200 epoch、学习率仍为 1e-5、cooldown 放大到 10 epoch、
`checkpoint_frequency=0`（每轮只维护 `best` 与 `last`，训练结束时上游额外保存
`step_<max_epoch>`）、`prioritize_resume=true`、`data_workers=4`。输出目录为
`runs/bladenet_long_<jobid>/`，日志在 `run/log/WalrusBladeNetLong/`，Slurm 时限 5 天。
训练正常结束后脚本自动调用 `scripts/evaluate_bladenet.py` 对 `best` 检查点评测
val 与 test，结果写到 `runs/bladenet_long_<jobid>_eval_best/`；上游 Trainer 自带的
最终 test 用的是最后一轮权重，两者只有在最后一轮恰为最优时才一致。

```bash
cd /WORK/PUBLIC/xuqy_work/run
sbatch runWalrusBladeNetLong.sh                       # 新开一次
MAX_EPOCH=300 sbatch runWalrusBladeNetLong.sh         # 改轮数
sbatch runWalrusBladeNetLong.sh optimizer.lr=2e-5     # 追加 Hydra 覆盖
RESUME_DIR=/WORK/PUBLIC/xuqy_work/walrus/runs/bladenet_long_<jobid> \
  sbatch runWalrusBladeNetLong.sh                     # 续训被中断的目录
```

续训由 `finetune_bladenet.sh` 的 `BLADENET_RESUME=1` 开关放行：要求
`checkpoints/last/full_checkpoint.pt` 存在且 `checkpoint.prioritize_resume=true`，
`walrus.train` 会恢复模型、优化器、已完成轮数和历史最优分数。续训前必须确认原作业
已停止。已知的小误差：续训后学习率比不间断时多衰减一步，在衰减阶段约低 1%–3%。

已完成的 50 epoch 运行 `runs/bladenet_a01_549685/`（作业 549685，学习率 1e-5，
每 5 epoch 存一份含优化器状态的检查点，共 139 GB）测试集全局相对 L1 为
pressure 1.2856%、temperature 0.6403%、mach 1.7027%、density 1.2567%，验证指标
在最后几轮仍在下降，因此改用上述长训练配置继续。

## 与 TurbineBladeNet 对齐的全场评测

参考工程为 `/WORK/PUBLIC/xuqy_work/TurbineBladeNet`。原工程有两种不同聚合方式：

- 最终报告：`tests/agg_ratio_of_sums.py`，逐物理场把全部测试样本的误差分子、
  真值分母分别累加，最后相除。它是当前正式对比的主指标。
- 常规日志：`cfd/figconvnet/utils/eval_funcs.py`、`AverageMeterDict`，先算一批的比值，
  再对批等权平均。当前导出的诊断指标明确使用 batch=1 的逐样本均值；如果旧日志
  使用 batch>1，两者不能直接视为同一聚合口径。

每个字段的主指标为：

```text
global_rel_l1 = sum_over_samples_and_points(abs(pred - target))
                / sum_over_samples_and_points(abs(target))
```

四个字段分别计算，不能把压力、温度、Mach、密度拼起来用同一个物理量分母。
CSV/JSON 中的 `rel_*` 保存原始比值，例如 `0.02` 表示 `2%`；比值允许大于 1，
不会被裁剪到 `[0,1]`。
测试集使用全部 275 个样本、每个样本原生 `256×64×32` 网格的全部点；不排除壁面，
不加体积权重或壁面权重，不裁剪预测。原函数没有 epsilon；零分母指标会明确标为
未定义并记录数量，不能通过添加 epsilon 或丢弃异常样本悄悄改变协议。

输出还包括逐样本 `rel_l1`、`rel_l2` 的均值，以及明确标为附加指标的全局相对 L2。
`norm_*` 辅助指标使用参考工程 `GridBladeNetConst` 的固定 min-max 范围，而不是
本模型训练时的 mean/std：

| 字段 | min | max |
| --- | ---: | ---: |
| pressure | 14069.078886979270 | 258291.277824503806 |
| temperature | 207.152981336621 | 496.408293993872 |
| mach | 0 | 1.749396919163 |
| density | 0.131447802851 | 2.592536052343 |

辅助指标为 R²、MSE、MAE、最大绝对误差、RMSE 和相对 L1/L2；保留原函数中的
`rrmse` 名字，但它实际计算的是普通 RMSE。为匹配参考包装器，真值经过同一套
float32 min-max encode/decode；模型的物理量预测本身不改变。

逐样本导出同时记录 `design_id`、NPZ 成员名和 tar 原始顺序编号。原工程按 tar
顺序读取，而当前适配器按 NPZ 编号读取；比较单个样本必须按 ID 匹配，不能仅靠
两份 CSV 的行号。整体全场比值不受样本遍历顺序影响。

本节对齐的是全场指标。声速带、物理梯度分区和前后缘等区域分析是另一层协议，
不能把全场结果标成区域结果。训练损失仍为当前标准化 MAE，并未改成原工程的
壁面加权或 EOS 复合损失。

训练中的验证和测试由 `BladeNetTrainer` 自动导出可追溯的 JSON/CSV。已有模型
可以只重跑评测，无需重新训练：

```bash
python scripts/evaluate_bladenet.py \
  --run-dir runs/bladenet_a800_epoch1_20260919_150640 \
  --out runs/bladenet_turbine_eval_epoch1 \
  --split both
```

评测脚本严格加载该实验的字段映射和已训练检查点，不再次按官方 base 的旧字段表
重排权重。新结果写入独立目录，保留旧版 NRMSE 报告；来源配置、检查点和参考指标
实现的哈希随结果保存。完整比较时不要使用诊断用的样本数量限制。

已用一轮 A800 检查点按此协议完成全部 274 个验证样本和 275 个测试样本的重评。
结果目录为 `runs/bladenet_turbine_eval_epoch1_20260919_160329/results/`；
主报告位于 `viz/turbinebladenet/test_epoch1_summary.csv`。测试集全局相对 L1 为
pressure **8.7265%**、temperature **5.2823%**、mach **16.4202%**、density **8.6943%**。
本次没有重新训练或更新权重；这些 L1 指标不能与旧版 13.93% 平均 NRMSE 当成同一
指标比较。区域分析未包含在这次全场重评中。

## 本次准备记录（2026-09-19）

已生成默认缓存与配置，使用 64 个训练样本、空间步幅 `(4,2,2)` 的统计。
`runs/bladenet_setup/preflight.json` 记录了与配置文件 SHA256 对应的检查结果：

- 原尺寸 `256×64×32` 的单帧样本读取通过，27 个输入字段和四个目标均为有限值。
- 官方 base 权重严格加载通过；字段表从 67 扩展为 91，模型有 1,303,500,599 个参数。
- 真实 train/val/test 数据降采样为 `32×16×8` 后，小模型前向、反向及验证通过，
  梯度有限且非零，验证输出的物理单位还原正确。
- 19 项新增测试、28 项上游回归测试通过；上游原有 1 项跳过。
  多进程数据加载回归在沙箱外的独立临时目录执行。
- 预检未启动训练循环、未更新优化器，正式训练输出目录尚未创建。

报告中的小模型损失来自随机初始化的接口检查，不能当作微调效果。
上述准备阶段尚未执行完整模型的 GPU 测试。随后已经在 A800 80GB 上完成一轮
原生网格训练及全部验证/测试：1279 次更新，训练 19.49 分钟，全流程 25.60 分钟，
PyTorch allocated 峰值 23.03 GiB；结果在
`runs/bladenet_a800_epoch1_20260919_150640/`。该历史运行使用旧版 NRMSE 评测，
不能将其中的 13.93% 测试 NRMSE 当成 TurbineBladeNet 的全局相对 L1。
