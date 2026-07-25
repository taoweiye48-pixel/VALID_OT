# 论文可用性声明与仓库文件对应表

本文中的“数据可用性”和“代码可用性”应引用仓库中的可复核文件，而不是笼统
地列出整个工程目录。推荐使用下面的对应关系。

## 数据可用性

| 论文中要说明的内容 | 仓库文件 |
|---|---|
| 数据集名称、来源类别和分析用途 | `data/VALID_OT_DATASET_SUMMARY.csv` |
| 数据来源、版本、获取方式和伦理/许可备注 | `data/VALID_OT_DATA_PROVENANCE.json`、`data/provenance/*_data_provenance.json` |
| 主分析配对、方向和独立单位 | `data/provenance/p1_v2_pair_preparation_manifest.csv`、`data/provenance/p1_v2_processed_pairs.sha256` |
| HER2ST 受控对应与人工层扩展 | `data/provenance/p1_v2_her2st_manual_truth_manifest.csv`、`data/provenance/manual_layer_pair_manifest.csv` |
| 数据处理后的完整性核验 | `data/provenance/*.sha256` |
| 用于主文表格和图的数值结果 | `results/` |
| 图中各点、线、热图单元的可追溯源数据 | `figure_source/*/data/` 和 `figure_source/supplementary/` |

公开原始数据不在本仓库镜像。正式稿应在数据可用性声明中给出各公开数据
资源的原始论文、官方项目页或数据仓库链接，并说明本仓库保存的是来源登记、
配对清单、处理契约、结果表和校验哈希。这样既满足可复核性，也避免重新分发
受原始数据许可约束的文件。

## 代码可用性

| 论文中要说明的内容 | 仓库文件 |
|---|---|
| 核心求解器、干预和评价指标 | `code/validot/` |
| 数据准备、主分析和 WP1–WP11 运行入口 | `code/run_*.py`、`code/prepare_*.py` |
| 固定参数和分析配置 | `configs/`、`code/protocol/` |
| 图形复现脚本 | `figure_source/` |
| 自动化测试和数值回归检查 | `code/tests/` |
| Python 依赖和容器环境 | `code/requirements-v1.3-core.txt`、`code/reproducibility/environment-v1.3.yml`、`code/Dockerfile.v1.3` |
| 复现顺序和输入目录约定 | `code/reproducibility/README_REPRODUCE.md`、`data/README.md` |
| 第三方比较方法的安装与许可边界 | `docs/THIRD_PARTY.md` |

## 论文中可直接使用的表述（待补 URL/DOI）

**数据可用性。** 本研究使用公开的 spatialDLPFC、HER2ST、Stereo-seq、
STARmap PLUS 和 spatialLIBD 人工皮层层标注数据。数据来源、版本、样本登记、
配对规则、处理契约和完整性校验记录于仓库的 `data/` 目录；用于论文表格和图
的处理结果与源数据表分别位于 `results/` 和 `figure_source/`。原始数据通过
相应数据资源的官方渠道获取，并遵循其许可条款。

**代码可用性。** 分析代码、固定配置、图形脚本、自动化测试和复现说明位于
仓库的 `code/`、`configs/`、`figure_source/` 和 `docs/` 目录。正式投稿时，
应将公开仓库 URL、版本标签和归档 DOI 替换到此段中的占位符。

## 正式公开前仍需补齐

1. 选择并加入一种 OSI 认可的软件许可证（见 `LICENSE_SELECTION_REQUIRED.md`）。
2. 填写 `CITATION.cff.template` 并将其保存为最终的 `CITATION.cff`。
3. 补充公开 GitHub URL、release tag 和 Zenodo/同类归档 DOI。
4. 将各数据资源的官方链接、版本和许可信息写入最终稿的数据可用性声明。
