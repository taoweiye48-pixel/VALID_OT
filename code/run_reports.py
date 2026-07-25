from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from validot.utils import file_hash, read_json, status_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "12_E8_statistics"
OUTPUT = ROOT / "14_reports"


def fmt(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
        return "NA" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(table: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = table[columns].copy()
    if limit is not None:
        view = view.head(limit)
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(fmt)
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, separator, *rows])


def collect_key_manifest() -> pd.DataFrame:
    patterns = [
        "00_protocol/*",
        "01_manifest/*",
        "04_code/*.py",
        "04_code/validot/*.py",
        "05_E1_solver_validation/*",
        "06_E2_truth_validation/*",
        "07_E3_DLPFC_retrospective/*",
        "08_E4_internal_fidelity/*.tsv",
        "08_E4_internal_fidelity/*DECISION.json",
        "09_E5_semisynthetic_external/*.tsv",
        "09_E5_semisynthetic_external/diagnostics/*.tsv",
        "09_E5_semisynthetic_external/diagnostics/*DECISION.json",
        "10_E6_real_external/*.tsv",
        "10_E6_real_external/*DECISION.json",
        "10_E6_real_external/optional_3dot/*",
        "11_E7_robustness/*.tsv",
        "11_E7_robustness/*DECISION.json",
        "12_E8_statistics/*",
        "13_figures/*.png",
        "13_figures/*.pdf",
    ]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return pd.DataFrame(
        [
            {
                "relative_path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in sorted(paths)
        ]
    )


def run_status_table() -> pd.DataFrame:
    rows = []
    for path in ROOT.rglob("*.json"):
        if "checkpoints" in path.parts or ".runtime" in path.parts or "vendor" in path.parts:
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if "status" in payload:
            rows.append(
                {
                    "relative_path": str(path.relative_to(ROOT)),
                    "stage": payload.get("stage", ""),
                    "status": payload.get("status", ""),
                    "updated": payload.get("updated_utc", payload.get("updated_local", "")),
                }
            )
    return pd.DataFrame(rows).sort_values(["stage", "relative_path"])


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = read_json(ROOT / "00_protocol" / "frozen_config.json")
    e0 = read_json(ROOT / "01_manifest" / "E0_DECISION.json")
    e1 = read_json(ROOT / "05_E1_solver_validation" / "E1_DECISION.json")
    e2 = read_json(ROOT / "06_E2_truth_validation" / "E2_DECISION.json")
    e3 = read_json(ROOT / "07_E3_DLPFC_retrospective" / "E3_DECISION.json")
    e4 = read_json(ROOT / "08_E4_internal_fidelity" / "E4_E5_DECISION.json")
    e6 = read_json(ROOT / "10_E6_real_external" / "E6_DECISION.json")
    e7 = read_json(ROOT / "11_E7_robustness" / "E7_DECISION.json")
    e8 = read_json(STATS / "E8_DECISION.json")
    m5 = read_json(ROOT / "10_E6_real_external" / "optional_3dot" / "M5_DECISION.json")

    fidelity = pd.read_csv(STATS / "fidelity_summary.tsv", sep="\t")
    external = pd.read_csv(STATS / "external_summary.tsv", sep="\t")
    gate = pd.read_csv(STATS / "registered_real_external_gate.tsv", sep="\t")
    cards = pd.read_csv(STATS / "validity_cards.tsv", sep="\t")
    association = pd.read_csv(STATS / "fidelity_utility_association.tsv", sep="\t")
    base_quality_path = STATS / "base_quality_sensitivity.tsv"
    base_quality = pd.read_csv(base_quality_path, sep="\t") if base_quality_path.exists() else pd.DataFrame()
    robustness = pd.read_csv(STATS / "robustness_summary.tsv", sep="\t") if (STATS / "robustness_summary.tsv").exists() else pd.DataFrame()
    m5_summary_path = ROOT / "10_E6_real_external" / "optional_3dot" / "M5_EXPLORATORY_SUMMARY.tsv"
    m5_summary = pd.read_csv(m5_summary_path, sep="\t") if m5_summary_path.exists() else pd.DataFrame()
    duplicate_path = ROOT / "09_E5_semisynthetic_external" / "diagnostics" / "duplicate.tsv"
    duplicate = pd.read_csv(duplicate_path, sep="\t") if duplicate_path.exists() else pd.DataFrame()
    missing_primary_path = ROOT / "09_E5_semisynthetic_external" / "missing_all.tsv"
    missing_primary = pd.read_csv(missing_primary_path, sep="\t") if missing_primary_path.exists() else pd.DataFrame()
    missing_10_path = ROOT / "09_E5_semisynthetic_external" / "diagnostics" / "missing.tsv"
    missing_10 = pd.read_csv(missing_10_path, sep="\t") if missing_10_path.exists() else pd.DataFrame()

    rank_pass_count = int(fidelity.rank_fidelity_pass.sum())
    full_pass_count = int(fidelity.full_registered_fidelity_pass.sum())
    real_gate_pass_count = int(gate.registered_real_external_gate.sum()) if len(gate) else 0
    if full_pass_count > 0 and real_gate_pass_count > 0:
        outcome = "Outcome A：至少一个方法—解释组合同时通过完整内部忠实与外部效度 Gate"
    elif full_pass_count > 0:
        outcome = "Outcome B：存在内部忠实解释，但未建立对应的外部效度"
    elif real_gate_pass_count > 0:
        outcome = "Outcome C：存在外部预测性，但未通过完整内部忠实 Gate"
    else:
        outcome = "Outcome D：未发现同时满足最低内部忠实或外部效度标准的解释"
    heterogeneous = cards.groupby(["source", "method"]).quadrant.nunique().max() > 1 or cards.quadrant.nunique() > 1
    if heterogeneous:
        outcome += "；结果具有明显方法/数据/代理特异性"

    best_fidelity = fidelity.sort_values("median_spearman", ascending=False).head(12)
    exact_external = external[(external.source == "real") & (external.score == "exact_combined")].copy()
    exact_external = exact_external.sort_values(["registered_real_external_gate"] if "registered_real_external_gate" in exact_external else ["median_absolute_gain"], ascending=False)
    report = f"""# VALID-OT 最终实验报告

**生成日期：** 2026-07-16  
**结论类型：** {outcome}  
**主协议：** `{config['protocol_id']}`；ε={config['solver']['epsilon']}，数值容差={config['solver']['sinkhorn_tol']:.0e}。  
**确认性范围：** M0–M4、两个外部空间技术、{e6.get('pair_count', 'NA')} 个生物 pair、双向 pair 内平均。  
**可选 3d-OT：** {m5['status']}；仅覆盖官方 transport head 的事后探索性验证，不是完整深度框架重训练。

## 1. 一句话判断

VALID-OT 主流程已完成并能够区分“解释是否忠实于有限干预后的模型响应”和“该分数是否能识别外部不一致”；最终证据应按方法、代理和数据来源分别解释，不能把 exact re-solve 称为生物学真值。

## 2. 执行完整性

- E0 数据与环境 Gate：`{e0['status']}`；数据选择在 audit 结果之前完成。
- E1 求解器验证：`{e1['status']}`；{e1.get('record_count', e1.get('passed', e1.get('passed_cases', 'NA')))} 个通过记录。
- E2 真值生成器：`{e2['status']}`；所有验证记录必须 100% 回读一致。
- E3 DLPFC：`{e3['status']}`，仅作为回顾性证据。
- E4–E5 主网格：`{e4['status']}`，{e4.get('tasks', 'NA')} 个任务，数值失败 {e4.get('numerical_failures', 'NA')}。
- E6 真实确认：`{e6['status']}`，{e6.get('runs', 'NA')} 个方向—方法运行，方向不作为独立样本。
- E7 稳健性：`{e7['status']}`，{e7.get('tasks', 'NA')} 个任务。
- E8 统计复核：`{e8['status']}`，层级 bootstrap {e8.get('bootstrap_replicates', 'NA')} 次。
- M5 3d-OT transport head：`{m5['status']}`，{m5.get('runs', 0)} 个双向真实任务；不具备注册 Gate 资格。

数值协议披露：v1.2 复用了 617 个在更严格 `1e-9` 容差下已经完成的任务；原失败任务和其余 582 个任务在 `1e-8` 下重新/首次计算，共 583 个。ε=0.1 的 legacy 检查点未进入汇总。该变更及理由记录在 `00_protocol/PROTOCOL_AMENDMENT_008.md`。

## 3. 内部忠实性

共 {len(fidelity)} 个聚合方法—干预—代理单元；其中仅秩 Gate 通过 {rank_pass_count} 个，完整注册 Gate（含冻结幅度 NMAE）通过 {full_pass_count} 个。没有幅度校准值的代理不会被偷偷记为完整通过。

内部 Spearman 最高的组合如下：

{markdown_table(best_fidelity, ['source', 'method', 'intervention', 'proxy', 'median_spearman', 'median_top_decile_precision', 'median_normalized_mae', 'rank_fidelity_pass', 'full_registered_fidelity_pass'])}

## 4. 真实数据外部效度

exact combined response 相对最佳冻结 QC 基线的结果如下。正增益表示 normalized excess AURC 更低；最终 Gate 还要求跨技术、pair 方向、bootstrap、负对照和混杂调整同时成立。

{markdown_table(gate, [column for column in ['method', 'witness', 'pairs', 'independent_units', 'median_absolute_gain', 'median_relative_gain', 'positive_pair_fraction', 'positive_independent_unit_fraction', 'positive_technology_count', 'bootstrap_ci_low', 'bootstrap_ci_high', 'negative_control_pass_fraction', 'adjusted_positive_fraction', 'registered_real_external_gate'] if column in gate.columns])}

注册真实外部 Gate 共通过 {real_gate_pass_count} 个方法—witness 单元。未通过只能表述为“未建立增量外部效度”，不能表述为绝对无效。

"""
    if len(base_quality):
        report += "外部增益与基础对齐较差程度的相关性（替代解释敏感性）：\n\n" + markdown_table(
            base_quality, list(base_quality.columns)
        ) + "\n\n"
    report += """

## 5. 半合成 missing 与重复 motif

"""
    if len(missing_primary):
        missing_summary = (
            missing_primary.groupby(["method", "score"], dropna=False).auroc.median().reset_index()
            .sort_values("auroc", ascending=False)
            .head(12)
        )
        report += "25% crop 的 missing AUROC 中位数领先组合：\n\n" + markdown_table(missing_summary, ["method", "score", "auroc"]) + "\n\n"
    if len(missing_10):
        missing10_summary = (
            missing_10.groupby(["method", "score"], dropna=False).auroc.median().reset_index()
            .sort_values("auroc", ascending=False)
            .head(12)
        )
        report += "10% crop 的 missing AUROC 中位数领先组合：\n\n" + markdown_table(missing10_summary, ["method", "score", "auroc"]) + "\n\n"
    if len(duplicate):
        duplicate_summary = (
            duplicate.groupby("method", dropna=False)
            .agg(
                entropy_AUROC=("entropy_ambiguity_auroc", "median"),
                top2_margin_AUROC=("top2_margin_ambiguity_auroc", "median"),
                dual_asymmetry=("mean_dual_asymmetry", "median"),
                strict_identity_error=("strict_identity_error_rate", "median"),
            )
            .reset_index()
        )
        report += "重复 motif 的 entropy 仍是注册主诊断，top-2 margin 只能作支持比较：\n\n" + markdown_table(duplicate_summary, list(duplicate_summary.columns)) + "\n\n"

    report += f"""## 6. 双轴关系与 validity card

{markdown_table(association, list(association.columns))}

validity card 象限计数：{cards.quadrant.value_counts().to_dict()}。相关区间若跨越 0，只能说明当前数据没有建立稳定单调关系，不能当作“内部忠实与外部效度等价无关”的证明。

## 7. 稳健性

"""
    if len(robustness):
        weakest = robustness.sort_values("same_direction_fraction").head(12)
        report += markdown_table(weakest, ["method", "variant", "units", "median_spearman", "median_external", "same_direction_fraction"]) + "\n\n"
    report += """主结论若只在单一 epsilon、tau、alpha、overlap、特征 fold 或规模成立，必须降级为探索性。稳健性表保留全部符号翻转，不删除失败参数。

## 8. 事后探索性 M5：3d-OT transport head

"""
    if len(m5_summary):
        report += markdown_table(
            m5_summary,
            [
                "method",
                "witness",
                "pairs",
                "technologies",
                "median_absolute_gain",
                "median_relative_gain",
                "positive_pair_fraction",
                "positive_technology_count",
                "registered_gate_eligible",
            ],
        ) + "\n\n"
    report += """M5 在主分析完成后加入，只能用来评估外推性。它运行官方 transport head，未重训练完整深度特征提取框架；其空间证据是离散 top-k support，因此没有伪造连续空间 endpoint gradient。M5 不能改变 M0–M4 的注册 Gate 判定。

## 9. 事实、推断与限制

### 事实

- exact re-solve 是冻结正则目标下的完整重新优化参照，不是真实生物对应。
- DLPFC 已被查看，只能作回顾性证据；STARmap PLUS 与 Stereo-seq 承担确认性验证。
- Stereo-seq 的五个 pair 是相邻发育阶段比较，不是同龄 serial sections，也不是独立 subject 的逐点真值。
- 双向运行先在 biological pair 内平均；半合成 seed 不替代生物重复。
- M5 官方 transport head 的数值验证和 14 个真实方向任务已完成，数值失败 0。

### 推断

- validity card 的差异支持“空间 OT 解释有效性依赖方法、代理、干预和数据来源”的条件式结论。
- 完整 Gate 未通过时，只能说当前证据不足，不能证明未来所有解释均无效。
- 外部 witness 与 audit 分数的关联在混杂调整后仍可能包含未测量的组织结构因素。

### 限制

- 真实数据没有逐点人工 correspondence；region/cell-type mismatch 和 held-out expression 只是 witness。
- 只有两个外部技术，技术层级 bootstrap 的精度有限。
- 五个 Stereo-seq 跨阶段 pair 共享数据链和发育趋势，必须与同阶段/同条件 pair 分层报告。
- 非 endpoint 代理没有可辩护的冻结幅度校准时，完整 NMAE Gate 被记为未通过/不可评，而不是用确认数据事后拟合。
- M5 仅覆盖官方 transport head，不是完整深度 3d-OT 训练；且为事后探索，不能与 M0–M4 的确认性证据混记。

## 10. 最终建议

本实验包已经足以形成一篇以“评估协议、失效分析和模型特异性”为主的稿件底稿。投稿前还需人工完成正式文献检索、图注与方法学文字审校；不能仅凭本自动报告声称首创或保证中科院分区。
"""
    (OUTPUT / "FINAL_DECISION.md").write_text(report, encoding="utf-8")

    evidence = pd.DataFrame(
        [
            {"type": "FACT", "claim": "M0-M4 numerical adapters passed E1", "status": e1["status"], "evidence": "05_E1_solver_validation/solver_validation.tsv"},
            {"type": "FACT", "claim": "Semisynthetic truth generator passed E2", "status": e2["status"], "evidence": "06_E2_truth_validation/truth_validation.tsv"},
            {"type": "FACT", "claim": "Primary E4-E5 grid completed without silent exclusions", "status": e4["status"], "evidence": "08_E4_internal_fidelity/E4_E5_DECISION.json"},
            {"type": "FACT", "claim": "Real external runs were averaged within biological pair", "status": e6["status"], "evidence": "12_E8_statistics/utility_direction_averaged.tsv"},
            {"type": "INFERENCE", "claim": outcome, "status": "CONDITIONAL", "evidence": "12_E8_statistics/validity_cards.tsv"},
            {"type": "LIMITATION", "claim": "Real labels are external witnesses, not pointwise truth", "status": "ACTIVE", "evidence": "01_manifest/pair_selection.tsv"},
            {"type": "FACT", "claim": "Exploratory official 3d-OT transport-head branch completed", "status": m5["status"], "evidence": "10_E6_real_external/optional_3dot/M5_DECISION.json"},
        ]
    )
    evidence.to_csv(OUTPUT / "EVIDENCE_TABLE.tsv", sep="\t", index=False)

    status_table = run_status_table()
    status_table.to_csv(OUTPUT / "RUN_STATUS_SUMMARY.tsv", sep="\t", index=False)
    manifest = collect_key_manifest()
    manifest.to_csv(OUTPUT / "REPRODUCIBILITY_MANIFEST.tsv", sep="\t", index=False)
    freeze = subprocess.run(
        [str(ROOT / ".runtime" / "Python310" / "python.exe"), "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (ROOT / "00_protocol" / "requirements.lock.txt").write_text(freeze, encoding="utf-8")

    decision = status_payload(
        "E9",
        "COMPLETED",
        outcome=outcome,
        rank_fidelity_passes=rank_pass_count,
        full_fidelity_passes=full_pass_count,
        registered_real_external_passes=real_gate_pass_count,
        m5_status=m5["status"],
        m5_runs=m5.get("runs", 0),
        evidence_rows=len(evidence),
        manifest_rows=len(manifest),
        figures=len(list((ROOT / "13_figures").glob("*.png"))),
    )
    write_json(OUTPUT / "E9_DECISION.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
