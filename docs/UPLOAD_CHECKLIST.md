# GitHub 上传边界

上传本目录时，使用 Git 提交并让 `.gitignore` 自动排除：

- `build/`
- `figures/`
- `data/raw/`
- `data/processed/`
- Python 缓存和 LaTeX 临时文件

应提交的内容是：

- `code/`
- `configs/`
- `data/` 中的来源登记、配对清单和校验文件
- `results/`
- `figure_source/`
- `docs/`
- `tools/`
- 根目录的安装、许可证和引用元数据文件

论文正文、论文 PDF 和最终渲染图不在本代码—数据包中。图形脚本及图形源
数据仍然保留，以支持从结果表重新生成论文图。

原始公共数据通过官方数据记录获取，不直接复制到 GitHub；其来源、版本、
处理规则和许可信息记录在 `data/README.md` 与 `data/provenance/`。
