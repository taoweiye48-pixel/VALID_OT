# Protocol Amendment 009: E7 压力参数数值失败处理

- 冻结时间：2026-07-16
- 适用协议：`valid-ot-v1.2-epsilon025-tol1e8-2026-07-16`
- 修改时点：E6 完成后、E7 尚未运行且尚未读取任何 E7 效果前。

## 发现的问题

原 E7 脚本捕获 Python 异常，但没有检查 audit 的 base、deleted 和 endpoint 求解结果中的
`converged` 标志。因此，参数压力测试中的未收敛解可能被静默写入稳健性表。

## 修正规则

1. E7 每个任务必须检查 base、全部 deleted 和全部 endpoint 的收敛标志。
2. 全部收敛的任务进入 `robustness_all.tsv`。
3. 未收敛或异常的任务进入 `robustness_numeric_failures.tsv`，不得进入效果汇总。
4. 压力参数失败是稳健性负面结果，应披露并计入 E7 决定，不倒推否定已在主参数下通过的 E4/E6。
5. 若存在失败，E7 状态为 `COMPLETED_WITH_NUMERIC_FAILURES`；主参数任务的失败须在最终报告中单列为高风险。
