# Figure 1 legend — coefficient-space variant

**Fig. 1 | VALID-OT separates intervention semantics, two local-fidelity
diagnostics and external utility.** **a**, Paired spatial sections define
independently scaled expression and spatial cost channels. For a specified
intervention \(I^{A,k}(t)\), the frozen model is completely reoptimized at the
baseline, local path positions and endpoint. Writing
\(R_i^{A,k}(t)=D_i(P^0,P^{A,k}(t))\), the finite-step score is
\(s_i(0.01)=R_i^{A,k}(0.01)/0.01\), the high-accuracy local reference is
\(r_i^L=\partial_tR_i^{A,k}(0^+)\), and the fully reoptimized endpoint response
is \(r_i^E=R_i^{A,k}(1)\). These quantities enter the question-matched audit
instance \(\mathcal A=(I,r^\ast,s,w)\). **b**, Let \(u\) denote the coefficient
of the deleted cost channel and \(v\) the coefficient of the retained channel.
Arm R follows
\((u,v)=(0.5(1-t),0.5)\), changing deletion while holding the retained channel
fixed. Arm N follows
\((u,v)=(0.5(1-t),0.5(1+t))\), combining deletion with retained-channel
compensation. The same coefficient plane is used for the later response-surface
analysis. Arm S lies outside this path family and synchronously multiplies the
Arm-N endpoint cost and all applicable linear regularization scales by
\(\lambda=0.5\); it is an objective-scale control rather than a third
finite-intervention path. **c**, Finite-intervention local fidelity is resolved
into local-response fidelity,
\(s(0.01)\leftrightarrow r^L\), and local-to-endpoint transportability,
\(r^L\leftrightarrow r^E\). The direct
\(s(0.01)\leftrightarrow r^E\) comparison is retained as an end-to-end readout.
External utility separately evaluates whether a response score ranks the
specified witness \(w\). All response references are model-relative and do not
alone establish biological correspondence.
