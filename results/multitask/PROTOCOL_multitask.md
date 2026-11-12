# Stage 4: task screen for the multi-task PPS vs blend evaluation — protocol

Written 2026-11-11, 9 pm New York time, before any screening, pilot or evaluation rollout of this stage. Branch
`stage4-task-screen` (made from `stage3-eval-t59` at 89bd01c), worktree `/workspace/pps-stage4`, run root
`/workspace/runs/stage4_screen`. Existing run folders are read, never written.

**Hardware.** Every rollout and every training run is on the pod's RTX 4090 (24 GB). The GPU name is written in
every training `config.json` and `metrics.json`, and in every policy server's metadata, which every trial record
carries (`server.gpu`). At most 2 policy servers fit in 24 GB, and none fits next to a training job, so serving and
training never overlap.

**Status of the screen.** Screening results (Stages A and B) are used for task selection only. They are not the
paper's main comparison and are not pooled with any evaluation trial (section d).

**Before this protocol.** Step 0 (pod check) ran first and is not a screening rollout. π0.5 on t59 normal trials 0
and 10 reproduced the Nov 11 RTX 4090 records bit for bit (actions, qpos, qvel, bottle pose, outcome);
`checks/step0_repro_t59.json`. One smoke test of the new worker (`scripts/multitask/run_mt.py`: π0.5 on screen trial
0 of t14, t50 and t70, 3 trials) ran in `/workspace/runs/stage4_screen/smoke`. It is discarded, and Stage A reruns
those trials from scratch in its own folder.

## Code

- `scripts/multitask/sim_mt.py`: env for any libero_90 task. The carried object and goal target come from the
  first `(On|In obj target)` predicate of the task's BDDL goal. The probe has the same per-step keys as t59's.
- `scripts/multitask/run_mt.py`: queue worker. It uses `scripts/faithful/run_faithful.py`'s episode loop
  unchanged: 10 settle steps, at most 400 policy steps, replanning every 5 actions of a 10-step chunk, success = env
  `done`.
- `scripts/faithful/serve_faithful.py`: unchanged methods, with two new flags. `--no-v3` skips loading the t59 v3
  proxies. `--task-id` records in the metadata which task the faithful proxies belong to.
- `scripts/multitask/build_task_inputs.sh`: the t59 data pipeline for any task. It runs the stage 2 export and
  render and `build_task_data.py`, then `scripts/faithful/build_faithful_data.py` with the paths passed by
  environment variables.
- Training: `scripts/faithful/distill_faithful.py` and `train_task_faithful.py`, unchanged except that the data
  paths are read from environment variables (the defaults are t59's).

## a. Task selection

### Candidates

Candidates are the libero_90 tasks on which π0.5 succeeded 1 to 4 times out of 5 in the earlier screen
(`results/m0/exp1_screen_summary.json`, init states 30–34) and which have a carry phase a drop can interrupt:

| Task | m0 π0.5 | Scene | Carried object → goal target (BDDL) |
| --- | --- | --- | --- |
| 2 | 4/5 | KITCHEN_SCENE10 | akita_black_bowl_1 → wooden_cabinet_1_top_region (in the drawer) |
| 14 | 3/5 | KITCHEN_SCENE2 | akita_black_bowl_2 → plate_1 |
| 19 | 4/5 | KITCHEN_SCENE3 | moka_pot_1 → flat_stove_1_cook_region |
| 31 | 1/5 | KITCHEN_SCENE5 | akita_black_bowl_1 → white_cabinet_1_top_side |
| 38 | 1/5 | KITCHEN_SCENE8 | moka_pot_1 → flat_stove_1_cook_region |
| 43 | 1/5 | KITCHEN_SCENE9 | white_bowl_1 → wooden_two_layer_shelf_1_top_side |
| 50 | 3/5 | LIVING_ROOM_SCENE2 | alphabet_soup_1 → basket_1_contain_region |
| 56 | 4/5 | LIVING_ROOM_SCENE3 | butter_1 → wooden_tray_1_contain_region |
| 60 | 4/5 | LIVING_ROOM_SCENE4 | akita_black_bowl_1 → wooden_tray_1_contain_region |
| 64 | 2/5 | LIVING_ROOM_SCENE4 | akita_black_bowl_2 → akita_black_bowl_1 (then both into the tray) |
| 70 | 3/5 | LIVING_ROOM_SCENE6 | chocolate_pudding_1 → living_room_table_plate_right_region |
| 77 | 1/5 | STUDY_SCENE2 | black_book_1 → desk_caddy_1_back_contain_region |
| 82 | 3/5 | STUDY_SCENE3 | black_book_1 → desk_caddy_1_left_contain_region |

- These are the 12 tasks named in the instructions plus **t19**. t19 meets the stated rule (4/5, a carry) but
  was not in the named list. It is added here, before any rollout, and flagged in the report.
- Left out:
  - t28: drawer closing, nothing to drop.
  - t30: π0.5 too strong.
  - t59: already the evaluation task.
  - t20 and t44: they meet the 1–4/5 count, but "turn on the stove" has no carry.

### Stage A: π0.5 alone

- **Trials.** 30 screening trials (j = 0–29, section d) of π0.5 (`base`) on each of the 13 candidates.
- **Serving.** 2 servers (`serve_faithful.py --no-faithful --no-v3`) with 3 workers each, task-major queue.
- **Success.** The env's `done` within the 400-step budget.

**Shortlist rule.**
1. Eligible: π0.5 successes 3 to 15 of 30 (10% to 50%, both ends included).
2. At most one task per LIBERO scene. Within a scene, keep the eligible task whose rate is closest to 30% (tie:
   lower task id).
3. Preference for scenes other than LIVING_ROOM_SCENE3 (t59's scene, here only t56). Take up to 5 tasks from the
   scenes other than LIVING_ROOM_SCENE3 first, by distance |rate − 30%| (tie: lower task id). Add t56 only if that
   leaves fewer than 5, and then place it last.
4. Order: by |rate − 30%| ascending (tie: lower task id), with t56 (if included) last.

The rule is coded in `scripts/multitask/select_shortlist.py` and applied as written, with no hand edits.

### Stage B: faithful proxies, in shortlist order

For each shortlisted task in order:
1. Build its inputs from its 50 LIBERO demos (yifengzhu-hf/LIBERO-datasets, every hdf5 step, no no-op filtering,
   re-rendered at eval resolution) with `build_task_inputs.sh`.
2. Train the faithful reference seed 0 (`distill_faithful.py --seed 0`: 20,000 steps, batch 8).
3. Train the faithful task proxy seed 0 from it (`train_task_faithful.py --seed 0`: 24,000 steps, batch 64).
   Steps 2 and 3 use the exact t59 recipe. Architecture, steps, batch, learning rate schedule, optimizer,
   augmentation and seeds (torch seed 42 + 0) are not changed. The only differences are the task's demo frames,
   targets and prompt.
4. Run `check_faithful.py` on 12 of the task's demo observations with its prompt (checks A–D as for t59). If a
   check fails, the task is reported and not evaluated.
5. Start 2 servers with that task's proxies (`--task-id T --no-v3`). Run `task_f` (faithful task proxy alone) on
   screening trials j = 0–49 and π0.5 on j = 30–49.
   - π0.5 trials 0–29 are reused from Stage A, provided a rerun of trials 0 and 1 on the Stage B servers
     reproduces Stage A bit for bit.
   - Otherwise π0.5 is rerun on 0–29 as well. The Stage B π0.5 figures then use only the rerun records, and the
     Stage A records are kept only for the shortlist.
6. **Qualifies** if `task_f` succeeds on 20 to 42 of 50 (40% to 85%, both ends included).

Stopping and the chosen tasks:
- Training for the next candidate starts only after the current task's evaluation has finished.
- No new candidate is trained once 3 tasks qualify.
- PPS and the blend are never run on a candidate and never used for selection.
- The chosen tasks are the qualifying ones, at most 3, in shortlist order. If fewer than 3 qualify after the whole
  shortlist, the report says so and selection stops there. No candidate outside the shortlist is added.

**Flags in the report.**
- Any task where `task_f` succeeds on 50/50.
- Any task where π0.5 never re-grasps. That means π0.5 never closes on the object again after it has been dropped:
  in the Step 5 calibration pilot, no branch at the frozen setting re-grasps (`regrasp_step` as in t59). In the
  screening trials it means no trial where the object was lifted at least 2 cm, lost, and grasped again.

## b. Primary PPS vs blend comparison (for the later evaluation on the chosen tasks)

This endpoint is fixed now. Sample sizes are set in the evaluation protocol before that evaluation runs and before
any of its outcomes are seen.

**Cells.**
- `base` (π0.5 = γ 0 of both lines).
- `pps_f_g04`, `pps_f_g06`, `pps_f_g08`.
- `blend_f_g02`, `blend_f_g04`, `blend_f_g06`, `blend_f_g08`.
- `task_f` (= blend γ 1).
- Velocity rules as in the t59 evaluation (`results/eval_t59_faithful/PROTOCOL.md` §1).

**Per task k and training seed s:**
- x = normal-trial success rate. Normal trials use eval seeds (section d), and every cell uses the same trials.
- y = drop-bank recovery rate. Each cell runs on the task's common bank (section c). Recovery is t59's definition:
  after the disturbance there are 25 consecutive env steps in each of which both fingers touch the object, the
  executed gripper action is closed and the object does not touch the table.
- **Blend line.** The 6 blend points γ = 0, 0.2, 0.4, 0.6, 0.8, 1, connected by straight segments in γ order.
  B(x\*) is the highest linearly interpolated y over all segments whose closed x-range contains x\*. A segment with
  equal end x gives the higher of its two y.
- **Gap.** gap(k, s, γ) = y_PPS(k, s, γ) − B_{k,s}(x_PPS(k, s, γ)).
- **Exclusion.** A (k, s, γ) whose PPS x lies outside [min x, max x] of that blend line is excluded from the
  average and reported with its x and the blend range. Exclusions are decided on the point estimates and stay
  fixed in the bootstrap. In a resample where an included point's x falls outside the resampled blend range, x is
  clamped to the nearest end of the range.

**Primary endpoint.**
- Δ = mean of gap(k, s, γ) over all included (k, s, γ), with γ ∈ {0.4, 0.6, 0.8}, k over the chosen tasks (and t59
  if the evaluation protocol includes it) and s over the training seeds evaluated. Every included cell has equal
  weight.
- 95% interval: cluster bootstrap, 10,000 resamples, percentile interval.
  - Within each task, normal-trial init states and bank states are resampled independently: clusters are all
    trials of one init state, and all branches of one bank state.
  - The same resampled clusters are used for every cell and every seed of that task, so the pairing is kept.
  - Tasks and training seeds are not resampled.
  - Each resample recomputes every x, y, blend line, gap and Δ.

**Secondary analyses.** Holm-corrected, family-wise α 0.05 over all 3 + K tests jointly:
- per-γ gaps (mean over tasks and seeds, 3 tests);
- per-task gaps (mean over γ and seeds, K tests).

Each gets the same bootstrap and a two-sided bootstrap p value:
p = min(1, 2 · min(#{Δ\* ≤ 0} + 1, #{Δ\* ≥ 0} + 1) / (10,000 + 1)).

The same comparison on drop-bank task completion is reported as descriptive, outside the family. Every result is
also broken down by bank source.

## c. Drop disturbance for the new tasks

Hardness comes only from the disturbance and from state properties fixed here. It never comes from any
controller's outcome on the bank states.

**Bank-state properties** (the t59 rule, with the task's carried object and goal target, `sim_mt.Probe`). A bank
state is the first mid-carry event of a normal trial, and every condition holds after the env step:
1. Both finger geom groups touch the object and the executed action[6] > 0.
2. The object is ≥ 5 cm above its post-settle height and does not touch the table.
3. The object is ≥ 5 cm horizontally from the goal target and does not touch it. Distance is to LIBERO's region
   box when the target is a site, else to the target object's body position.
4. At least 130 env steps of budget are left: 410 − (event step + 1) ≥ 130.

**Bank.**
- 120 states per task, 40 from each source, the first 40 qualifying normal trials in trial order:
  - π0.5 (`base`),
  - `pps_f_g04` (seed 0),
  - `task_f` (seed 0).
- Every controller and every training seed runs on the same bank. Results are broken down by source.

**Disturbance family**, fixed now and ordered by level L:
1. Restore exactly (no mj_forward) and discard the saved action queue.
2. Hold the arm with [0]\*6 + [−1] and force the gripper open for 10 env steps (the t59 drop).
3. At the end of the 10th step, move the object horizontally by d_L, keeping its height and orientation, and set
   its velocity to zero (`pps_real.perturb.Perturber._teleport` logic). d_L is 0, 3, 6, 9, 12, 15 cm for L = 0–5.
   - Direction: away from the goal target, rotated by an angle drawn from U(−45°, 45°).
   - A draw is rejected if the moved object would touch anything but the table. A draw is also rejected if its new
     xy lies outside the box spanned by the scene's `*_init_region` sites expanded by 3 cm (it would leave the
     table's work area).
   - Up to 50 draws. If none is clear, the object is not moved, and this is logged.
4. The controller then replans from the observation until success or the trial's original budget (t < 410).
- L = 0 is exactly the t59 drop.

**Calibration.** Once per task, on π0.5 alone, before any evaluation rollout on that task.
1. Run π0.5 pilot trials (seeds in section d) until 30 pilot states with the bank-state properties are collected,
   at most 150 pilot trials. If fewer are found, all that were found are used, and this is reported.
2. Run π0.5 on every pilot state at every level L = 0–5, one branch each. Noise is default_rng((13, s, 0)) and
   the direction draws come from default_rng((19, s, 0)).
3. Freeze the lowest level whose π0.5 recovery is in [60%, 85%]. If no level is, freeze the level closest to
   72.5% (tie: the lower level).
4. Record the frozen level, its d and every level's π0.5 recovery, completion and re-grasp rate in an addendum
   to this protocol. Then it is never changed.

## d. Seeds: screening and pilots never coincide with an evaluation trial

| Use | Trial index i | env.seed | Policy noise | Init state |
| --- | --- | --- | --- | --- |
| Evaluation normal trials (all tasks) | i = trial (< 10,000) | 7 + i | default_rng((7, i)) | i mod 50 |
| t59 harder starts (existing) | 10,000 + j | 7 + 10,000 + j | default_rng((7, 10,000 + j)) | moved starts |
| **Screening (Stages A and B)** | 20,000 + j | 7 + 20,000 + j | default_rng((7, 20,000 + j)) | j mod 50 |
| **Drop calibration pilots** | 30,000 + j | 7 + 30,000 + j | default_rng((7, 30,000 + j)) | j mod 50 |
| Evaluation drop branches | state s, branch b | (restored) | default_rng((11, s, b)); directions default_rng((17, s, b)) | bank |
| Pilot drop branches | pilot state s | (restored) | default_rng((13, s, 0)); directions default_rng((19, s, 0)) | pilot states |

- Evaluation normal trials must use i < 10,000, so no screening or pilot trial is ever reused as an evaluation
  trial.
- The 50 LIBERO init states are shared by all rows, because LIBERO has only 50 per task. The env and noise seeds
  are disjoint.

## Step 4: t59 seeds 1 and 2

After Stage B, train the faithful reference and task proxy seeds 1 and 2 on t59 with the same recipe:
`distill_faithful.py --seed 1|2`, then `train_task_faithful.py --seed 1|2` from that seed's reference. They are not
evaluated.

## Time estimate and the cut rule

All figures are on the 4090:
- Stage A: about 1 h (390 trials, 6 workers).
- Stage B, per candidate:
  - inputs, built while Stage A runs: about 0.3 h;
  - reference: about 112 min;
  - task proxy: about 112 min;
  - checks and evaluation (50 `task_f` + 22 π0.5 trials): about 0.3 h;
  - about 4.1 h in all, so 12–20 h for 3 to 5 candidates.
- Step 4: 2 × 3.7 h ≈ 7.5 h.
- Step 5 (calibration): about 2 h.
- Total: about 23–31 h.

Under the 40 h limit, so nothing is cut in advance. If the measured queue grows past about 40 h, Step 4 is cut
first, and the report lists what was left undone.

## Addendum 1 (Nov 12, before any harder-start rollout): harder-starts pilot on t14 and t77

Written after a change of plan on Nov 12. Stage B stops after t77, and Step 5 is not run. The pod is stopped by
17:00 UTC, and no training job or eval batch starts unless it is estimated to end by 16:40 UTC. This addendum is
committed and pushed before any harder-start rollout.

**Purpose.** A pilot to inform a possible protocol amendment. It does not select tasks, and it changes nothing in
sections a–d.

**Policies.** Only π0.5 alone (`base`) and the task's faithful task proxy seed 0 alone (`task_f`) are run. PPS
and the blend are never run on these starts. Rollouts use the same servers (`serve_faithful.py --no-v3 --task-id
T` with `reference_tT_faithful_s0` and `task_tT_faithful_s0`), 2 servers × 3 workers, and the same episode loop as
the screening trials.

**Construction rule** (the t59 idea, `scripts/multitask/build_hard_starts_mt.py`). The rule is fixed here and
the shift is never tuned after any result:
- *Object:* the object the task moves, from the first `(On|In obj target)` goal predicate of the BDDL.
- *Demo range:* the object's xy at frame 0 of each of the 50 training demos.
- *Goal location:* the goal target's xy in those 50 frame-0 states (the site centre for a region, else the target
  object's body), averaged over the demos.
- *Shift axis:* "directly away from the goal location", along a world table axis as on t59. Take the vector from
  the mean goal location to the mean demo start of the object, and use the world axis (+x, −x, +y or −y) with
  the larger component and its sign.
  - Checked on t59 before any t14 or t77 geometry was computed. The rule gives −y, a demo width of 4.83 cm and a
    shift of 7 cm, which is the t59 construction.
  - The exact (unsnapped) vector would give (−0.40, −0.92) and 9 cm, which is not t59's, so it is not used.
- *Shift size:* along the chosen axis, the width of the demo range (max − min) plus 2 cm, rounded up to a whole
  cm. Each start's actual distance beyond the demo range is reported.
- *Starts:* the 50 official init states, with the object's free-joint xy moved by the shift. Nothing else
  changes.
- *Settle check (t59's):* after the 10 settle steps with the dummy action, a start passes if all of these hold:
  - the object touches the table and nothing else;
  - its orientation is within 5° of the unshifted start's after the same settle;
  - it moved less than 5 mm in xy while settling;
  - every other object ends within 2 mm (xy) of where it ends from the unshifted start.
- *Direction choice:*
  1. Use the away axis if at least 40 of the 50 starts pass.
  2. Otherwise try the two perpendicular axes in this fixed order: the away axis rotated +90° (counter-clockwise
     seen from above), then −90°. Each uses its own shift size by the same width + 2 cm rule. The first with at
     least 40 passing starts is used.
  3. If none has 40, the task is skipped and the report says why.
- *Trials:* every passing start of the chosen axis is run once per policy. Trial j uses kept start j.

**Seeds.** Trial index 40,000 + j, `env.seed(7 + 40,000 + j)` and noise `default_rng((7, 40,000 + j))`. These are
disjoint from evaluation trials (< 10,000), the t59 harder starts (10,000+), screening (20,000+) and drop pilots
(30,000+).

**Analysis (descriptive).** Per task, π0.5 and `task_f` success rates with Wilson 95% intervals, the paired 2×2
table (both, π0.5 only, `task_f` only, neither) and the exact McNemar p. They are shown next to the t59 harder-start
numbers from `results/eval_t59_faithful/REPORT.md` (π0.5 29/50, task proxy alone 14/50). No test is used for any
decision.
