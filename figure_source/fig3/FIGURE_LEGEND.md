# Figure 3 legend

**Local-response transportability depends on the intervention path and cost
channel.** **a**, Spearman rank correlation between the high-accuracy local
response, $r^L$, and the complete finite-intervention endpoint response,
$r^E$. **b**, Relative mean absolute error (rMAE) between $r^L$ and $r^E$.
For expression-channel deletion, Arm R retained high rank transportability
(method medians 0.988–0.998) and low magnitude error (0.024–0.071), whereas
Arm N showed lower rank transportability (0.729–0.906) and larger magnitude
error (0.285–0.384). Spatial-channel interventions were less transportable in
both arms (median Spearman 0.325–0.805; median rMAE 0.360–0.420).
**c**, Median row-level path directness, $\eta$, defined as endpoint displacement
divided by accumulated path length. Values remained close to 1 across the 12
families (method–arm–channel medians 0.943–0.999), arguing against large path
reversals as the main source of endpoint mismatch. **d**, Median row-level
late/early path-speed ratio, $\kappa$. Arm-R expression paths were approximately
uniform ($\kappa=0.986$–1.079), Arm-N expression paths decelerated
($\kappa=0.699$–0.769), and spatial-channel paths accelerated
($\kappa=1.791$–2.025). Dashed grey lines mark $\eta=1$ and $\kappa=1$.

In every panel, small translucent points are the 21 independent main-analysis
units within each method–arm–channel family, thick coloured segments are the
interquartile ranges and large coloured points are medians. Balanced OT, UOT
and Row-softmax are shown in blue, teal and ochre, respectively. Results are
descriptive across independent units and do not constitute pooled spot-level
inference. Path directness and speed change characterize the frozen model
response path; they do not by themselves identify a unique causal contribution
of channel removal, retained-channel compensation or relative regularization.
Source data: `wp4_path_geometry_unit.tsv`.
