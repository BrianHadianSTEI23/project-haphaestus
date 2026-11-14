# Phase 5.5.1.1 — Unresolved Hypotheses Handed to Phase 5.6

1. What internal stage (`coordinate frame`, `visual latent z_t`, `task embedding e_t`, `grounding_head L1/L2`, or `category/spatial priors`) produces the strong right-over-left (`45.0% vs 8.3%`) and category (`60.0%` `blue_cylinder` vs `10.0%` `red_box`/`yellow_cylinder`) behavioral asymmetry?
2. Why does `M2_distractor_masked_rgb` (where only the assigned target object is unmasked) still exhibit `2/60 (3.3%)` left vs `24/60 (40.0%)` right success (`McNemar p = 1.0490e-05`)?
3. Does `z_t` linearly encode both `obj_1` and `obj_2` egocentric body-frame coordinates equally well on left vs right sides before `task_embedding` conditioning?
