# Protocol amendment 003: Stereo-seq real-pair definition

**时间**：2026-07-16，仅完成外部H5AD元数据检查后、任何VALID-OT审计运行前。  
**原因**：公开包中E15.5同龄section只有一对，STARmap PLUS只有一对，无法满足预注册的“两个技术、至少六个真实pair”。

## 修改

加入同一公开数据包 `3Dreconstruction/` 中的 Stereo-seq E11.5、E12.5、E13.5、E14.5、E15.5、E16.5，构造五个按时间相邻的发育阶段pair：

1. E11.5 ↔ E12.5；
2. E12.5 ↔ E13.5；
3. E13.5 ↔ E14.5；
4. E14.5 ↔ E15.5；
5. E15.5 ↔ E16.5。

另保留E15.5 s1 ↔ s2同龄pair以及STARmap PLUS disease replicate 1 ↔ replicate 2，共七个真实pair、两个空间技术。

## 解释限制

- 五个发育阶段pair只能检验跨时间点对齐的 region/cell-type witness 与 held-out expression consistency；
- 它们不是同龄相邻物理切片，不提供逐点 correspondence 真值；
- 时间相关表达变化是预先声明的混杂因素，必须按“同龄pair”和“跨阶段pair”分层报告；
- 不得把增加pair数量误写为七个独立subject。

该修订只依据文件名、时间点、样本量和公开标签完成，尚未查看任何模型或解释表现。
