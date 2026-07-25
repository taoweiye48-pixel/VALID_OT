# Protocol amendment 006: balanced entropic OT numerical refiner

**Time:** 2026-07-16, triggered by a prespecified numerical stop before inspection of scientific outcome tables.  
**Failure:** `Stereo-seq_base2 / rigid / seed 0 / balanced_ot` did not converge after 4,000 alternating log-Sinkhorn iterations at the frozen `epsilon=0.1` and `tol=1e-9`. Continuing from 4,000 to 8,000 iterations changed the plan by relative L1 `1.32e-6`, so the result could not be re-labelled as converged.

**Repair:** Balanced OT now uses 200 log-domain scaling iterations followed by trust-region Newton-CG refinement of the identical entropic dual. One target dual potential is fixed to remove gauge freedom. The primal cost, entropy coefficient, masses, interventions, maximum tolerance, and exact-response definition are unchanged.

**Verification:**

- the full E1 suite passed again for all 15 method/shape records;
- the failed Stereo-seq task converged under the original tolerance in 14.8 seconds;
- two additional rigid seeds initially stopped because SciPy's infinity-norm `gtol` was not equivalent to the registered total marginal L1 test; scaling `gtol` by the total number of marginal entries resolved both, after which E1 again passed 15/15;
- row and column marginal L1 are used as the final gauge-invariant convergence test;
- previously completed balanced tasks are retained only if they had already met the original stricter log-scaling convergence flag; failed or interrupted tasks are recomputed.

This is a numerical solver repair, not an endpoint or parameter change.
