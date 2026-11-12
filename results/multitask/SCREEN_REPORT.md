# Stage 4 task screen: report

Branch `stage4-task-screen`, made from `stage3-eval-t59` (89bd01c). The protocol is `results/multitask/PROTOCOL_multitask.md` (b00a2c1) with addendum 1 (0abab63, the harder-starts pilot). Every rollout and every training run below ran on one RunPod pod with an NVIDIA GeForce RTX 4090; the GPU name is in every training config and every result record. Screening results are for task selection only, not the main comparison.

Stage B was cut short by the Nov 12 change of plan (pod stopped by 17:00 UTC; no job started unless estimated to end by 16:40 UTC). This report makes no task selection.

## Step 0

π0.5 on t59 normal trials 0 and 10 reproduced the Nov 11 RTX 4090 records bit for bit (actions, qpos, qvel, object pose, outcome): `checks/step0_repro_t59.json` (IDENTICAL_ALL = True).

## Stage A: π0.5 alone, 30 screening trials per candidate

Seeds: screening offset 20,000 (trial j: init state j mod 50, env.seed(7 + 20,000 + j), noise default_rng((7, 20,000 + j))). 2 servers × 3 workers. Wilson 95% intervals.

| Task | Scene | Prompt | π0.5 success | Trials with a mid-carry event | Decision |
| --- | --- | --- | --- | --- | --- |
| t2 | KITCHEN_SCENE10 | put the black bowl in the top drawer of the cabinet | 21/30 = 70% [52, 83] | 24 | not eligible: 21/30 outside 3-15 |
| t14 | KITCHEN_SCENE2 | put the middle black bowl on the plate | 9/30 = 30% [17, 48] | 10 | shortlisted, position 1 |
| t19 | KITCHEN_SCENE3 | put the moka pot on the stove | 25/30 = 83% [66, 93] | 26 | not eligible: 25/30 outside 3-15 |
| t31 | KITCHEN_SCENE5 | put the black bowl on top of the cabinet | 4/30 = 13% [5, 30] | 28 | shortlisted, position 5 |
| t38 | KITCHEN_SCENE8 | put the right moka pot on the stove | 12/30 = 40% [25, 58] | 20 | shortlisted, position 4 |
| t43 | KITCHEN_SCENE9 | put the white bowl on top of the cabinet | 1/30 = 3% [1, 17] | 2 | not eligible: 1/30 outside 3-15 |
| t50 | LIVING_ROOM_SCENE2 | pick up the alphabet soup and put it in the basket | 27/30 = 90% [74, 97] | 28 | not eligible: 27/30 outside 3-15 |
| t56 | LIVING_ROOM_SCENE3 | pick up the butter and put it in the tray | 28/30 = 93% [79, 98] | 29 | not eligible: 28/30 outside 3-15 |
| t60 | LIVING_ROOM_SCENE4 | pick up the black bowl on the left and put it in the tray | 14/30 = 47% [30, 64] | 17 | eligible, left out: 5 tasks closer to 30% |
| t64 | LIVING_ROOM_SCENE4 | stack the right bowl on the left bowl and place them in the tray | 15/30 = 50% [33, 67] | 29 | eligible, but t60 is closer to 30% in LIVING_ROOM_SCENE4 |
| t70 | LIVING_ROOM_SCENE6 | put the chocolate pudding to the right of the plate | 23/30 = 77% [59, 88] | 29 | not eligible: 23/30 outside 3-15 |
| t77 | STUDY_SCENE2 | pick up the book and place it in the back compartment of the caddy | 8/30 = 27% [14, 44] | 30 | shortlisted, position 2 |
| t82 | STUDY_SCENE3 | pick up the book and place it in the left compartment of the caddy | 10/30 = 33% [19, 51] | 15 | shortlisted, position 3 |

Rule (protocol section a): eligible 3-15 of 30; one per scene (closest to 30%, tie lower id); up to 5, scenes other than LIVING_ROOM_SCENE3 first by |rate-0.30| (tie lower id), LIVING_ROOM_SCENE3 only to fill, placed last.

Shortlist, in order: t14, t77, t82, t38, t31. Only t60 and t64 share a scene among the eligible tasks (LIVING_ROOM_SCENE4); t60 (14/30) is closer to 30% than t64 (15/30). Of the 7 eligible tasks, the 5 closest to 30% are kept; t60 is the 6th. No LIVING_ROOM_SCENE3 task was eligible (t56: 28/30).

## Stage B: faithful proxies seed 0, 50 screening trials

Per candidate in shortlist order: faithful reference (20k steps, batch 8) and task proxy (24k steps, batch 64), seed 0, the t59 recipe on the task's own 50 demos; `check_faithful.py` (checks A–D); π0.5 trials 0 and 1 rerun and compared with Stage A; then `task_f` on trials 0–49 and π0.5 on 30–49 (0–29 reused from Stage A when the rerun is bit identical). Qualifies if `task_f` succeeds on 20–42 of 50 (40–85%).

| Task | π0.5 | task_f | task_f only / π0.5 only (McNemar p) | Rerun of π0.5 0–1 vs Stage A | Drop after lift (re-grasped): π0.5, task_f | Qualifies |
| --- | --- | --- | --- | --- | --- | --- |
| t14 | 20/50 = 40% [28, 54] | 50/50 = 100% [93, 100] | 30 / 0 (1.9e-09) | identical | 1 (0), 2 (0) | no (task_f 100%: FLAG) |
| t77 | 10/50 = 20% [11, 33] | 40/50 = 80% [67, 89] | 33 / 3 (2.3e-07) | identical | 0 (0), 0 (0) | yes |
| t82 | 21/50 = 42% [29, 56] | 29/50 = 58% [44, 71] | 15 / 7 (0.13) | identical | 3 (1), 3 (0) | yes |

Qualifying so far: t77, t82. Not run in Stage B: t38, t31.

## Harder-starts pilot (addendum 1): π0.5 and task_f only

A pilot to inform a possible protocol amendment; it does not select tasks. PPS and the blend were not run. Construction (fixed in addendum 1 before any rollout): the carried object moved along the world table axis closest to "away from the goal", by the demo range width on that axis + 2 cm, rounded up to a whole cm; the t59 settle check; at least 40 of 50 starts must pass, else the perpendicular axes (+90°, then −90°). Seeds: offset 40,000. The t59 row is from `results/eval_t59_faithful/REPORT.md` (task proxy alone = the t59 faithful task proxy).

| Task | Object, axis, shift | Starts passing | Beyond demo range | π0.5 | task_f | Both / π0.5 only / task_f only / neither | McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t14 | akita_black_bowl_2, +y, 7 cm | away (+y) 7 cm: 50/50 | 2.1–7.0 cm | 0/50 = 0% [0, 7] | 17/50 = 34% [22, 48] | 0 / 0 / 17 / 33 | 1.5e-05 |
| t77 | black_book_1, +x, 7 cm | away (+x) 7 cm: 50/50 | 2.3–7.1 cm | 15/50 = 30% [19, 44] | 36/50 = 72% [58, 83] | 9 / 6 / 27 / 8 | 0.00032 |
| t59 (Nov 11) | tomato_sauce_1, −y, 7 cm | 50/50 | 2.1–7.0 cm | 29/50 = 58% [44, 71] | 14/50 = 28% [18, 42] | — / 22 / 7 / — | 0.0081 |

- t14, normal screening trials for comparison: π0.5 20/50, task_f 50/50.
- t77, normal screening trials for comparison: π0.5 10/50, task_f 40/50.

## Training runs (all on the NVIDIA GeForce RTX 4090)

Train time is the training loop (metrics.json `train_s`); wall time is start to end in the chain log (includes loading the teacher, the frames and, for the reference, building the prefix cache).

| Run | Steps, batch, torch seed | Train time | Wall time | Final probe | Status |
| --- | --- | --- | --- | --- | --- |
| reference_t14_s0 | 20000, 8, 42 | 91.9 min | 03:18–04:59 UTC (101 min) | mse 0.0174 | complete |
| task_t14_s0 | 24000, 64, 42 | 112.3 min | 04:59–06:52 UTC (113 min) | mse 0.0055 | complete |
| reference_t59_s0 (Nov 11, for reference) | 20000, 8, 42 | 111.8 min | — | mse 0.0190 | complete |
| task_t59_s0 (Nov 11, for reference) | 24000, 64, 42 | 112.0 min | — | mse 0.0052 | complete |
| reference_t59_s1 | | | — | | not run |
| task_t59_s1 | | | — | | not run |
| reference_t59_s2 | | | — | | not run |
| task_t59_s2 | | | — | | not run |
| reference_t77_s0, attempt 1 | | | 07:07–07:12 UTC (5 min) | | killed at the pod's 125 GB memory limit while allocating the prefix cache, before any training step (see below) |
| reference_t77_s0 | 20000, 8, 42 | 81.7 min | 08:07–09:39 UTC (91 min) | mse 0.0164 | complete |
| task_t77_s0 | 24000, 64, 42 | 112.5 min | 09:39–11:32 UTC (113 min) | mse 0.0052 | complete |
| reference_t82_s0, attempt 1 | | | 07:12–14:00 UTC (408 min) | | stopped by me at 07:15 UTC during setup, before any training step (it would have hit the memory limit; see below) |
| reference_t82_s0 | 20000, 8, 42 | 82.9 min | 12:27–14:00 UTC (92 min) | mse 0.0192 | complete |
| task_t82_s0 | 24000, 64, 42 | 112.5 min | 14:00–15:54 UTC (114 min) | mse 0.0053 | complete |
| reference_t14_s0 plain-memory test (memory fix) | 300, 8, 42 | 1.2 min | 07:16–07:26 UTC | mse 0.6066 (step 300) | test only, discarded |

## Memory failure and fix

- **Failure.** The first t77 reference run (07:07 UTC) was killed about 4.5 min in, while it was allocating the
  teacher prefix cache, before any training step.
  - The pod's container memory limit is 125 GB (cgroup `memory.limit_in_bytes` 124,999,999,488), and the
    recorded peak equalled the limit.
  - `distill_faithful.py` keeps the cache K and V in pinned host memory (frames × 18 layers × 522 tokens × 256,
    bf16, 9.6 MB per frame and tensor pair). PyTorch's pinned host allocator rounds each block up to a power of
    two.
  - t59 (5571 frames) and t14 (5393) need 2 × 26–27 GB, rounded to 2 × 34.4 GB. t77 (7336 frames) needs
    2 × 35.3 GB, rounded to 2 × 68.7 GB = 137 GB.
  - t82 (7310 frames), t38 (10,588) and t31 (7576) would fail the same way. I stopped the chain and killed the t82
    reference run about 3 min in, during setup, before any training step.
- **Fix** (`scripts/faithful/distill_faithful.py`, `build_prefix_cache`).
  - K and V stay pinned when the rounded size of both fits in 80 GiB, so t59 and t14 keep exactly the old code
    path, including Step 4's t59 seeds. Otherwise they go in plain host memory.
  - Pinning changes nothing downstream: every read is a fancy-index copy (`K[flat]`, `K[idx]`) that makes an
    unpinned tensor before `.to(cuda)`.
  - Peak use for t14 with plain memory was 68 GB, so t38's 102 GB cache would need about 118 GB, near the 125 GB
    limit.
- **Check: the training batches are bit identical between the two paths** (`scripts/multitask/check_pin_batches.py`,
  `checks/memfix_pin_compare.json`).
  - On t14, seed 0, two separate processes ran the reference script's setup and batch assembly, one pinned and
    one plain.
  - The SHA-256 of the whole K and V caches is identical.
  - So is every batch of the first 500 groups (2000 training steps, 4 reshuffles at 674 steps per epoch): the
    frame ids, the host batch `K[flat]`/`V[flat]`, and the device copy. IDENTICAL_ALL = true.
- **Also run, not used to judge the fix.** A 300-step t14 reference run on the plain path matched the original
  t14 run's step-0 probe exactly (1.99545). Its losses differ from the 4th decimal on (step 100: 1.25124 against
  1.25066). Since the batches are bit identical, this difference comes from run-to-run nondeterminism in
  training, not from the memory path. The test run is discarded.

## Deviations from the protocol

- **t19 added as a 13th Stage A candidate.** It meets the stated rule (π0.5 4/5 on the stage 1 count, a carry)
  but was not in the named list. It was added in the protocol before any rollout. It was not eligible (25/30).
- **Concurrent builds during Stage A.** The task input builds (stage 2 export, frames, observations) ran on the
  CPU while Stage A rollouts ran. π0.5 calls were about 3% slower than in Step 0 (360–370 ms against 356 ms). The
  outcomes do not depend on timing (the episode loop is synchronous). The t31 input build overlapped the t14
  reference run.
- **Discarded runs.** These were all discarded, and none of their trials is counted anywhere:
  - the worker smoke test (`stage4_screen/smoke`);
  - a test of the Step 5 drop code on a t50 run (`step5_test`);
  - the 300-step plain-memory test run.
- **Memory failure and code change.** The first t77 reference run and the first t82 reference run did not
  complete. `distill_faithful.py` was changed so large prefix caches are not pinned (see above). Tasks with caches
  under the limit (t59, t14) use the unchanged pinned path.
- **Nov 12 change of plan.** This replaced the protocol's order of work:
  - Stage B stopped after t14 and t77 ; t82 was then trained and evaluated in Step C (t77 task_f 40/50 ≤ 85%); t38 and t31 were not run.
  - Step 4 was not done: after t82 (finished 16:15 UTC) the t59 seed 1 reference (estimate 109 min) would have ended at 18:04 UTC, so it was not started. No t59 seed was evaluated, as instructed.
  - Step 5 (drop calibration) was not run. Its code was written and tested only on the discarded t50 run, and no
    addendum for it was written.
  - To run single tasks, `run_stageB.sh` gained two overrides, used only by `run_rest.sh`: `ONLY_TASKS` (the
    shortlist is replaced by the listed tasks) and `NO_STEP4` (stop after Stage B). Per-task steps are unchanged.
- **Harder-starts direction.** The rule's "directly away from the goal" was put into practice as the world table
  axis (±x or ±y) with the larger component of (mean demo start − goal). This reproduces the t59 harder starts
  (−y, 7 cm). The exact away vector would give (−0.40, −0.92) and a 9 cm shift on t59. This choice was written
  into addendum 1 after computing only t59's geometry, before any t14 or t77 geometry and before any rollout.
- **Harder-starts seeds and servers.** Seed offset 40,000 (disjoint from eval < 10,000, the t59 harder starts
  at 10,000, screening at 20,000 and the drop pilots at 30,000). 2 servers (π0.5 and task_f, the task's seed 0
  proxies) × 3 workers each.

## Left undone

- Stage B: t38 and t31 not run (t38 would also have been near the memory limit; see below).
- Step 4: no t59 seed 1 or seed 2 run trained (no time after t82); no t59 seed evaluated
  (by instruction).
- Step 5: drop calibration not run; no addendum for it.
- No task selection is made (by instruction).
- t38 was never reached. Its prefix cache would be about 102 GB, and the measured peak for t14 suggests about
  118 GB in total, near the pod's 125 GB limit. It was not tried, and the data loader was not changed.

## Flags

- t14: task_f succeeded on all 50 screening trials (100%), above the 85% upper bound.
- In the screening trials, re-grasps after a drop were rare: t14 π0.5 had 1 drop after lift and 0 re-grasps;
  task_f had 2 and 0 (t77: no drop after lift for either policy).
- Each harder-starts row is one pilot of 50 starts. The t59 row comes from a different task and a different
  construction pass, so the comparison is descriptive only.
