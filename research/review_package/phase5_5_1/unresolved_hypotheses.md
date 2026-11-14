# Phase 5.5.1 — Unresolved Hypotheses & Boundaries for Independent Interpretation

1. **Origin of Category and Lateral Biases Under `F1`/`F2`:**
   - Why does `F1` reach `blue_cylinder` (`60.0%`) and right-side targets (`45.0%`) substantially more often than `red_box`/`yellow_cylinder` (`10.0%`) and left-side targets (`8.3%`), while `O1` achieves `85.0%–95.0%` across all categories and `86.7%` left / `98.3%` right? Training dataset class distribution, camera-frame perspective weighting, and loss weighting in `hjepa_best.pt` remain unexamined here.
2. **Reason `M1_target_masked` Drops `F1` While `task_switch_eta_vis_step0` Is Near Zero:**
   - Even though `F1` does not robustly select by task token (`pair_selectivity_margin = +0.067`, `Holm5_p = 0.4337`, `dual_first_correct = 5.0%` vs `same_object_collapse = 21.7%`), masking the assigned target (`M1 = 11.7%`) hurts assigned success more than masking the distractor (`M2 = 21.7%`, `Holm5_p = 0.0081`), while `M2` simultaneously lowers `unassigned_reached` (`20.0% -> 15.0%`) and raises `task_switch_eta_vis_step0` (`-0.0105 -> +0.1132`). How visual salience of visible objects interacts with weak/biased task-token modulation requires independent interpretation.
3. **Boundary of Controller Ego-Memory Null Difference (`H3 = -0.004, p = 0.5637`):**
   - Here both candidate objects remain visible from `t=0` (`>= 274` px) without sensory blackout. Non-significance in `F1-F2` and `F3-F4` does **not** establish equivalence, nor does it predict behavior under sustained visual occlusion/blackout.
4. **Upstream Feature Separability vs Downstream `grounding_head` Decoding:**
   - Whether `latent_s` or `grounding_latent` already separates both candidate objects geometrically prior to `grounding_head` (i.e., whether the failure measured against `O1 = 92.5%` resides inside the trained `grounding_head` / `task_embedding` conditioning rather than the visual encoder) has not yet been probed on this geometry-separated cohort.
