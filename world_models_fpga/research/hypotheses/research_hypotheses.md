# Falsifiable research hypotheses

The question is which reusable computations, if any, enable embodied spatial behavior. These are preregistered hypotheses, not conclusions.

## H1 — action-conditioned predictive representation

For an observation encoder \(z=E(o)\) and transition \(\hat z'=T(z,a)\), candidate actions cause distinguishable predictions that improve a goal-and-risk decision \(a^*=\arg\max_a[U(\hat z',g)-\lambda r(\hat z',a)-\kappa c(a)]\). Test prediction against encoded actual future observations and task labels, then compare reactive, prediction-unused, action-unconditioned, and prediction-used agents. A null effect of action conditioning or use ablation falsifies the behavioral part. Controls: shuffled actions and oracle-state upper bound. Evidence target: E2 only after closed-loop interventions.

## H2 — egocentric relational representation

There exists a representation/function retaining target range and bearing \(\rho,\phi\) and obstacle relations in body coordinates. Under global transform \(g\), task-invariant quantities should be invariant while body-frame vector outputs should transform equivariantly. Test translations, yaw rotations, left/right/front/behind disambiguation, held-out headings, probes, and downstream interventions. Decodability alone is E0; no action effect, incorrect equivariance, or failure on held-out orientations falsifies causal-use claims.

## H3 — temporal abstraction

At stride \(K\), macro transitions predict \(z_{t+K}\) using exactly actions \(a_{t:t+K-1}\), and a macro subgoal causally changes the micro action. Compare K=1, 3, 5, 8 and a reactive control on long-horizon tasks. Positive controls use labeled waypoint structure; negative controls replace macro activations with matched/random activations. No improvement or no macro-ablation effect rejects the hypothesis.

## H4 — predictive safety mechanism

For candidate action \(a\), \(r(z,a)\) predicts collision probability within a stated horizon. Evaluate Brier score, reliability, AUROC, distance/action monotonicity where geometry predicts it, and collision change after risk-output ablation. A classifier with poor calibration or an action selector unchanged by risk does not establish predictive safety or a control barrier function.

## H5 — task-relevant invariance

Spatial behavior is stable across controlled lighting/color/texture changes while information required by color-dependent instructions remains available. Evaluate representation consistency and navigation separately with identity-preserving and identity-changing controls. Loss of relevant color semantics or no behavior preservation rejects the claimed separation.

All positive claims require independent episodes, at least three seeds where training occurs, episode-level confidence intervals, and prespecified evaluation splits. Evidence is reported as E0 observational, E1 candidate mechanism, E2 causal evidence, or E3 transferable mechanism; failure and inconclusive outcomes remain reportable results.
