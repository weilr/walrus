# Linux / Miniconda 运行环境

环境名为 `walrus`，安装位置为
`/home/fit/xuqy/WORK/miniconda3/envs/walrus`。

## 使用

```bash
source /home/fit/xuqy/WORK/miniconda3/etc/profile.d/conda.sh
conda activate walrus
cd /home/fit/xuqy/WORK/walrus
```

VS Code 的 Python 解释器可选择：
`/home/fit/xuqy/WORK/miniconda3/envs/walrus/bin/python`。
已安装 `ipykernel`，Notebook 可选择此环境作为内核。

## 重建环境

在项目根目录执行以下命令。原有 `requirements-lock.txt` 保留为 Windows
环境记录；`requirements-linux-lock.txt` 记录本次 Linux 安装的实际版本，包含
CUDA 运行库以及 Excel/PPT 导出脚本所需依赖。

```bash
conda create -n walrus python=3.11 pip -y \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda activate walrus
python -m pip install -r requirements-linux-lock.txt
python -m pip install -e . --no-deps
```

本次实际安装版本为 Python 3.11.16、PyTorch 2.5.1 + CUDA 12.4、NumPy 1.26.4。
`the_well` 固定在提交 `ad50de0e8861380d7c4d62c0503c654710f026e7`。
安装本项目时使用 `--no-deps`，以保留该提交，避免 `pyproject.toml` 中的
`master` 地址重新选择上游代码。

## 环境验证

```bash
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest -q \
  tests/test_import.py tests/test_jitterer.py \
  tests/test_dataset.py tests/test_finetune.py
python -m walrus.train --cfg job server=local logger=none model=debug
```

这些测试生成临时数据，无需下载数据集或预训练权重。
2026-09-19 验证结果：`pip check` 通过，上述 21 项测试全部通过，
训练入口配置读取成功；核心库、Well benchmark、报表和 Notebook 依赖均可导入。
训练和检查点测试会在当前目录写入 `extended_config.yaml`，应在独立的
临时目录运行，避免覆盖项目中的现有配置。

## GPU 与实际运行配置

创建环境时，当前节点的 `nvidia-smi` 无法连接 NVIDIA 驱动。
环境安装 CUDA 版 PyTorch；GPU 训练和推理需在可使用 NVIDIA 驱动的节点验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

现有部分 `pretrained/` 下的评估配置包含 Windows 的 `D:/...` 路径。
实际评估前，可运行 `python scripts/make_local_eval_config.py --help`，
为 Linux 数据集、权重和输出目录生成新的配置。

本环境包含 Walrus 主模型及 Well benchmark 的依赖；`external_models`
可选依赖组中的其他模型依赖未单独安装。
