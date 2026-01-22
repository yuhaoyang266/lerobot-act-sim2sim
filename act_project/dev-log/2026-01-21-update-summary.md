# 2026-01-21 Update Summary (M3 Hybrid Policy)

## What changed
- Re-collected 500 demos with 22-dim observations = [observation(19) + desired_goal(3)]; old 19-dim set backed up to `pnp-sim2sim_backup_v19`.
- Training pipeline updates: observation/action normalization saved per run; augmented features `rel_goal_obj`, `rel_obj_ee`; gripper loss switched to BCE + sigmoid with class weight; 50% of training windows start at episode head to cover APPROACH.
- Inference hybrid control: rules handle APPROACH/DESCEND (open gripper, move to `[obj_x, obj_y, z≈0.22→0.025]`), then model controls CLOSE/LIFT/TRANSPORT; actions clipped; gripper binarized after sigmoid.

## Current results
- Quick run `run-m3-8` (50 epochs) still evaluates at 0% success, episodes hit TimeLimit (220 steps).
- Model output improved on later stages but early APPROACH/DESCEND still brittle; gripper direction now consistent after BCE.

## Next steps
1) Run `debug_compare_actions.py` against `run-m3-14-approach-fix` to verify hybrid handoff (expect ee_xy_dist<0.02, z→0.025, gripper closes on handoff).
2) If still failing, lower `GRASP_HEIGHT` to 0.02 and/or force `gripper=-1` during CLOSE before returning control to the model.
3) Target a sanity gate of ≥1/10 success before longer training.
