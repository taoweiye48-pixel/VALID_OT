# VALID-OT：代码与数据发布包

[English](README.md) | [简体中文](README_zh-CN.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21581800.svg)](https://doi.org/10.5281/zenodo.21581800)

本仓库收录 VALID-OT 的代码、配置文件、数据来源记录、论文源数据表及
可重复性材料。论文稿件源文件和渲染后的论文图片不包含在本代码与数据
发布包中。

## 仓库内容

- `code/validot/`：核心求解器、干预定义、评价指标和见证变量评估；
- `code/run_*.py` 与 `code/prepare_*.py`：分析和数据准备入口，包括
  WP1--WP11；
- `configs/`：版本化的分析配置；
- `code/tests/`：数值测试和回归测试；
- `results/`：论文使用的结果表与清单；
- `figure_source/`：绘图脚本及各面板的源数据；
- `data/`：公开数据来源记录、配对注册表和校验和清单；
- `docs/`：复现说明、第三方软件说明及数据与代码可用性说明；
- `code/reproducibility/`：运行环境与执行说明。

## 安装

冻结环境使用 Python 3.10：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r code\requirements-v1.3-core.txt
python -m pip install -e .
```

## 验证

```powershell
python -m pytest -q code\tests
```

依赖单独归档的旧版参考环境的测试，只有在设置
`VALIDOT_LEGACY_ROOT` 后才会运行，否则将被跳过。绘图脚本读取
`figure_source/` 和 `results/` 中的内容，并将生成结果写入已被
Git 忽略的 `figures/` 目录。

## 复现

请先阅读
[code/reproducibility/README_REPRODUCE.md](code/reproducibility/README_REPRODUCE.md)
和 [data/README.md](data/README.md)。随后从相应公开数据集的官方记录
下载原始数据，在 `data/provenance/` 中记录数据版本和许可证，准备
仓库相对路径下的输入目录，并运行 `configs/` 中的预设配置。

本仓库不镜像公开数据集的原始文件或本地处理缓存。数据源对象、切片
标识符、准备流程约定和校验和均记录在 `data/provenance/` 中。
可选的 PASTE、PASTE2 和 3d-OT 封装程序及其许可证说明见
`docs/THIRD_PARTY.md`。

## 发布信息

项目代码以 `LICENSE` 文件中的 BSD-3-Clause 许可证发布。第三方数据集
和可选对照实现继续遵循其上游许可条款。版本 `0.1.0` 已归档至 Zenodo：
[doi:10.5281/zenodo.21581800](https://doi.org/10.5281/zenodo.21581800)。
不限定版本的概念 DOI 为：
[doi:10.5281/zenodo.21581799](https://doi.org/10.5281/zenodo.21581799)。
`CITATION.cff` 记录了经确认的作者、ORCID、许可证、发布版本及该版本
对应的 DOI。
