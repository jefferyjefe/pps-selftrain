# t59 evaluation: pi0.5 alone vs PPS at gamma 0.4

Generated 2026-11-10 09:59:50 UTC by scripts/stage3/summarize.py from the files in /workspace/runs/eval_t59.

## Results

| Test | pi0.5 alone | PPS gamma 0.4 | PPS minus pi0.5, paired, 95% cluster bootstrap | McNemar exact p |
| --- | --- | --- | --- | --- |
| Normal, paired trials | 49 / 200 = 24.5% [19.1, 30.9] | 129 / 200 = 64.5% [57.7, 70.8] | +40.0 pts [+29.5, +50.5] (n = 200 pairs, 50 clusters) | 1.58e-20 (discordant: PPS only 84, pi0.5 only 4) |
| Mid-carry drop | 140 / 200 = 70.0% [63.3, 75.9] | 131 / 200 = 65.5% [58.7, 71.7] | -4.5 pts [-12.0, +2.0] (n = 200 pairs, 40 clusters) | 0.108 (discordant: PPS only 8, pi0.5 only 17) |

Success rates are successes / trials with a Wilson 95% interval in brackets.

## Velocity cosines along the steered path (PPS rollouts)

| Pair | dims | normal trials | drop trials | pooled | paper (Fig. 10b) |
| --- | --- | --- | --- | --- | --- |
| ref vs base | first7 | 0.9587 | 0.9537 | 0.9571 | 0.946 |
| ref vs base | all32 | 0.9889 | 0.9878 | 0.9885 | 0.946 |
| task vs ref | first7 | 0.8465 | 0.7952 | 0.8298 | 0.865 |
| task vs ref | all32 | 0.9435 | 0.9210 | 0.9362 | 0.865 |
| task vs base | first7 | 0.7928 | 0.7297 | 0.7723 |  |
| task vs base | all32 | 0.9273 | 0.9020 | 0.9191 |  |

Mean over 13046 PPS policy calls x 10 denoising steps. Per step (pooled, steps at t = 1.0 ... 0.1):

- ref_base_first7: 0.977, 0.972, 0.966, 0.961, 0.958, 0.959, 0.960, 0.954, 0.942, 0.923
- ref_base_all32: 0.994, 0.993, 0.992, 0.990, 0.990, 0.990, 0.990, 0.988, 0.983, 0.976
- task_ref_first7: 0.981, 0.976, 0.969, 0.959, 0.940, 0.902, 0.837, 0.741, 0.602, 0.391
- task_ref_all32: 0.995, 0.994, 0.992, 0.990, 0.984, 0.974, 0.953, 0.918, 0.850, 0.711
- mean |v_base| 18.411, mean |gamma (v_task - v_ref)| 1.997 (all 32 dims, per step)

## Safety checks

- gamma = 0: PPS code path bit exact with pi0.5 on 12 observations: **True**. gamma 0.4 changes actions on all: True. Median ms per call: pi0.5 348, PPS 469.
- Branching: restoring a saved state with its action queue and noise generator and replaying pi0.5 reproduced the original trajectory bit for bit (actions, qpos, qvel, bottle pose, outcome) in **40 of 40** states (restored observation images identical in 40).
- Drop branch determinism: 3 (state, branch) pairs run twice on different server processes, identical in 3; identical to the main drop run in 3 of 3.
- Cross check with stage 2: pi0.5 trials 0-29 use the same init states, env seeds and noise as stage 2's eager base episodes 0-29 (run on an RTX PRO 6000; this pod has an RTX PRO 4500). Same outcome in 24 of 30, same outcome and step count in 19; successes 9 then, 7 now.

## Details

- pi0.5 alone, normal: mean 72.3 policy calls, 389 ms per call (server), 93.6 s wall per trial, errors: none
- pi0.5 alone, drop: mean 19.8 policy calls, 387 ms per call (server), 25.8 s wall per trial, errors: none
- PPS gamma 0.4, normal: mean 44.1 policy calls, 511 ms per call (server), 62.4 s wall per trial, errors: none
- PPS gamma 0.4, drop: mean 21.2 policy calls, 507 ms per call (server), 29.4 s wall per trial, errors: none
- Mid-carry event fired in 84 of 200 pi0.5 normal trials (130 of 200 for PPS).
- Drop pool: 40 states from pi0.5 trials [3, 6, 8, 11, 14, 19, 20, 22, 26, 27, 39, 43, 44, 47, 52, 54, 61, 72, 74, 76, 80, 89, 93, 95, 98, 114, 118, 121, 123, 126, 127, 135, 139, 143, 145, 150, 161, 173, 176, 179] (finished prefix 0..179), minimum remaining budget 130 steps. Event steps [264, 276, 230, 266, 137, 255, 272, 143, 74, 203, 68, 66, 153, 177, 174, 260, 230, 222, 258, 76, 228, 71, 64, 261, 264, 238, 264, 228, 151, 75, 208, 222, 66, 65, 131, 231, 246, 264, 77, 221]; remaining budget per state [145, 133, 179, 143, 272, 154, 137, 266, 335, 206, 341, 343, 256, 232, 235, 149, 179, 187, 151, 333, 181, 338, 345, 148, 145, 171, 145, 181, 258, 334, 201, 187, 343, 344, 278, 178, 163, 145, 332, 188].
- Drop, pi0.5 alone: bottle out of the fingers' grasp at handover in 200 of 200, touching a finger 0, on the table 200; succeeded during the drop frames 0; re-grasped 200, re-lifted 5 cm 195; mean remaining budget 225 steps.
- Drop, PPS gamma 0.4: bottle out of the fingers' grasp at handover in 200 of 200, touching a finger 0, on the table 200; succeeded during the drop frames 0; re-grasped 196, re-lifted 5 cm 183; mean remaining budget 225 steps.

## Reading

- **Normal trials.** PPS at gamma 0.4 succeeds far more often than pi0.5 alone on t59: 64.5% against 24.5%, +40
  points (95% cluster bootstrap +29.5 to +50.5). Discordant pairs split 84 to 4 for PPS (McNemar p = 1.6e-20). This
  holds clearly even with the per init state clustering accounted for.
- **Drop test.** No advantage for PPS. pi0.5 recovered in 70.0% and PPS in 65.5%, a difference of -4.5 points
  (-12.0 to +2.0), with discordant pairs 17 to 8 for pi0.5 (p = 0.11). PPS is not significantly worse. Two things to
  keep in mind. (1) Every saved state comes from a pi0.5 trajectory at a point pi0.5 reached by itself, after a grasp
  and lift, so the start of each drop trial is on pi0.5's own state distribution. (2) The drop is mild for pi0.5: it
  re-grasped in 200 of 200 branches, and recovery here is mostly a re-grasp of an upright bottle near the gripper.
  pi0.5's 70% recovery rate is far above its 24.5% normal rate because most of its normal failures happen before the
  first grasp, as stage 2 found.
- **Cosines.** Along PPS's steered path, ref vs base is 0.957 over the first 7 dims (0.989 over all 32), close to the
  paper's 0.946, so the reference proxy tracks pi0.5 off pi0.5's own path about as well as the paper's. Task vs ref
  is 0.830 over 7 dims (0.936 over 32) against the paper's 0.865. The task proxy departs from the reference most at
  the last denoising steps (0.39 at t = 0.1 over 7 dims), where the steering delta is largest relative to v_base. The
  paper does not say which dims its numbers use, so both are reported.
- **pi0.5 baseline.** 24.5% here against 30% (9 of 30) in stage 2 and 36.7% in milestone 0 (compiled). Trials 0 to 29
  give the same outcome as stage 2's episodes in 24 of 30. The difference is the GPU (RTX PRO 4500 here, RTX PRO 6000
  then): bf16 kernels differ, so trajectories differ.
- **Not checked.** Whether any of LIBERO's 50 official init states coincide with the starts of the 50 demos the
  proxies were trained on. The README's controls ask for disjoint starts in later stages.

## Exact definitions and settings

- **Trials.** 200 per method. Trial i uses LIBERO's official init state i mod 50 for libero_90 task 59 (the task has
  exactly 50, so each is used 4 times), env.seed(7 + i) before reset, and flow matching noise from
  numpy default_rng((7, i)): the j-th policy call of the trial gets the j-th (10, 32) standard normal draw. Both
  methods run the identical 200 (init state, seed) pairs with identical noise per call index. Once the two methods'
  trajectories diverge, call j still gets the same draw, but on a different observation. Trials 0 to 29 match
  milestone 0's and stage 2's episodes 0 to 29.
- **Success.** The `done` LIBERO's env.step returns, i.e. env._check_success() on the BDDL goal
  `(In tomato_sauce_1 wooden_tray_1_contain_region)`: the bottle's body position strictly inside the tray contain
  region box (site position ± |R · size|, bottom lowered 1 cm). Settings are openpi examples/libero/main.py's for
  libero_90: 256 px renders, images rotated 180 degrees and resize-with-pad to 224, 10 settle steps with action
  [0]*6 + [-1], at most 400 policy steps (the loop runs while t < 410, settle steps included), replan every 5 actions
  of the 10-action chunk, 10 denoising steps. An exception ends the episode as a failure (as in main.py) and is
  recorded. Differences from main.py: per-trial env seed and noise (milestone 0 convention) and episode-level logging.
- **Sampler.** pps_real.sampler, eager (torch.compile off), every call at batch 1, on an RTX PRO 4500 Blackwell.
- **Mid-carry event** (scripts/stage3/t59_sim.py). It fires at the first env step, after the settle steps, at which
  all of the following hold. (1) Both fingers touch the bottle (any collision geom of each finger, the stage 2
  Perturber.grasping test) and the executed action commands the gripper closed (action[6] > 0). (2) The bottle body
  is at least 5 cm above its height at the end of the settle steps and does not touch the table. (3) The bottle body
  is at least 5 cm (horizontally) from the tray's contain region rectangle and does not touch the tray. The thresholds
  were picked from 8 pilot pi0.5 episodes on a separate noise seed (1000, not used in the evaluation). The bottle starts
  about 15 cm from the tray box and pi0.5 lifts it 12 to 15 cm during carry, so the event fires just after the lift,
  well before the tray.
- **Saved states.** Taken during the pi0.5 normal trials themselves; saving changes nothing in the trial. A snapshot
  holds the MuJoCo integration state (mj_getState with mjSTATE_INTEGRATION: time, qpos, qvel, act, qacc_warmstart,
  ctrl, applied forces, eq_active, mocap, userdata), the plain data attributes of the robosuite env, robot, OSC
  controller, gripper (its integrated finger target) and every observable (sampling timers and last values), the env's
  obs cache, numpy's global RNG state, model body poses (checked on restore), and the client state: step counter, the
  remaining action queue, the noise generator's exact state, the call count and the processed policy input.
- **Pool.** The first 40 pi0.5 normal trials (in trial order) whose event fired, taken from the shortest fully
  finished prefix of trials 0, 1, 2, ... that contains 40 events. The selection doesn't depend on the order in which
  trials finished.
- **Drop.** Restore the snapshot. For 10 env steps (0.5 s) the gripper's internal finger target is forced fully open
  (as in stage 2's release, because LIBERO's gripper integrates its command) with action [0]*6 + [-1], i.e. the arm
  holds position. The saved action queue is discarded, and after the 10th step the method's policy replans from the
  current observation and runs until success or the trial's original budget ends (t < 410 counted from the trial's
  start; 120 to 190 steps remained in the pilot). The drop phase is identical for both methods, so they differ only
  from the handover. Branch b of state s uses noise default_rng((11, s, b)), the same for both methods. Success is the
  same LIBERO check. In the pilot the bottle left the fingers on the first frame and rested on the table by frame 3.
- **PPS.** v = v_base + 0.4 (v_task − v_ref) at all 10 Euler steps. Proxies are /workspace/runs/reference_t59_paper_all50/final.pt
  and /workspace/runs/task_t59_paper_all50/final.pt, unchanged. They get the same uint8 224 x 224 agentview and wrist
  frames the base gets, which come from the same function (pps_real.libero_obs.process_images) that produced their
  training images. Normalization is inside PaperProxy.encode, as in training. Images are encoded once per call, with
  bf16 autocast as in training. The hook (scripts/stage3/pps_steer.py) does not short circuit at gamma 0, unlike
  pps_real.sampler.pps_hook, so the gamma = 0 check runs the whole path.
- **Cosines.** For every PPS call and denoising step, the cosine over the flattened (10, 7) or (10, 32) chunk between
  v_ref and v_base, v_task and v_ref, and v_task and v_base, on the steered path (x_t as PPS actually moved it). These
  are averaged over calls and steps, the same definition as scripts/stage2/check_task_paper.py.

## Choices you should know about

1. **Three server processes, each serving both methods.** A call takes about 350 ms (pi0.5) and 470 ms (PPS) on this
   pod, against 98 ms in stage 2. It is bound by the CPU (eager PyTorch Python overhead on this pod's EPYC 7B13), with
   the GPU about 15% busy for one process. So three server processes (7.7 GB of GPU memory each) run at once, with 3
   LIBERO workers each pulling jobs from a shared queue in the order (pi0.5 trial 0, PPS trial 0, pi0.5 trial 1, ...).
   Each server loads the base and both proxies, and the client picks the method per request. This is safe because
   calls share no state: the gamma 0 check interleaves pi0.5 and PPS calls in one process and is bit exact, and the
   replay and repeat checks reproduce trajectories exactly on different server processes. Batch size is always 1.
2. **The drop holds the arm still** and discards the queued actions, so the perturbation is identical for both
   methods. The alternative, letting queued pi0.5 arm actions run during the drop, would have made the perturbation
   depend on how many actions (0 to 4) were left in the queue.
3. **10 drop frames.** This is longer than "a few" because the policy's close command re-closes the gripper within
   about 8 steps of the handover. 10 frames let the bottle reach the table before the policy acts. It lands within 3.
4. **Pool states come from the same pi0.5 trials as the normal test,** not from separate rollouts. This saves GPU
   time, and the snapshot doesn't change the trial.
5. **Statistics.** Wilson intervals and McNemar treat the 200 trials as independent. But the normal trials come in
   groups of 4 per init state and the drop trials in groups of 5 per saved state, so the summary also gives a cluster
   bootstrap interval for the paired difference. Trust that one more if the two disagree.
6. **The restore does not call mj_forward.** A first version did, and replay diverged by about 1e-8 in qpos from the
   first step (in one pilot state that grew into a different action 90 steps later), because mj_forward rewrites
   qacc_warmstart. Without it, all 3 pilot states replayed bit for bit. The full check runs on all 40 states before
   any drop trial.
7. **GPU differs from stage 2** (RTX PRO 4500 here, RTX PRO 6000 then). bf16 kernels can differ between GPUs, so
   trials 0 to 29 need not reproduce stage 2's base episodes. The cross check above reports how many do.
8. **Pod control.** `runpodctl get pod $RUNPOD_POD_ID` returns Unauthorized. runpodctl had no API key configured;
   after `runpodctl config --apiKey $RUNPOD_API_KEY` (the pod's own key, which writes ~/.runpod/config.toml) it is
   still Unauthorized, because the pod-scoped key can't list the account's pods. Querying this pod through the GraphQL
   API with the same key works (desiredStatus RUNNING), and the podStop mutation is authorized: tested on an invalid
   pod id, it returned POD_NOT_FOUND, not Unauthorized. Pod stops go through scripts/stage3/pod_stop.sh (runpodctl
   stop pod, GraphQL fallback). The watchdog (scripts/stage3/watchdog.sh, started 08:11 UTC) stops the pod at
   19:00 UTC = 14:00 EST (EST is UTC−5 on 2026-11-10) whatever happens, and writes a partial summary 5 minutes before
   if the run hasn't finished.
9. **Not modified:** proxies, their training code, /workspace/data/t59_distill, the base checkpoint, pps_real/ (sampler,
   policy, perturb, libero_obs). New code is only in scripts/stage3/. Nothing was committed or pushed.
10. **The first branching check flagged one state; the pool rule was changed.**
    - *What happened.* At 09:13 UTC the replay check on the first pool (the first 40 events in trials 0 to 93, no
      budget rule) was bit exact for 39 of 40 states. The 40th, state 25 (trial 65), fired its event on step 409,
      the last step of the budget, so 0 steps were left to replay. Its restored images and outcome matched, but the
      check counts "no steps to compare" as a mismatch. The orchestrator therefore stopped all workers (153 pi0.5 and
      155 PPS normal trials were done, no drop trials), wrote a failure summary and started its 15 minute countdown
      to stop the pod.
    - *Cause.* A check and pool selection bug, not irreproducible branching. 2 old pool states (trials 65 and 68,
      with 0 and 9 steps left) could not even finish the 10 drop frames, and 8 more had under 100 steps.
    - *What I did.* I cancelled the stop and, as you instructed, changed the pool rule to exclude events with too
      little remaining budget. A saved state must leave at least **130 env steps** of the trial's budget after the
      event (410 - (event step + 1) >= 130, i.e. event at step 279 or earlier). The minimum comes only from pi0.5's
      own normal trials, never from a drop outcome. Among the 36 successful pi0.5 trials with a mid-carry event in
      trials 0 to 152 (all finished at 09:20 UTC), finishing from the event took a median of 46, a 90th percentile of
      77 and at most 120 steps. 130 = the 10 drop frames + 120, so every state leaves room for the slowest undisturbed
      completion pi0.5 showed. A first replacement value of 100 that I briefly used was dropped before any drop trial
      ran, because it came from a pilot drop recovery time. The same pool is used for both methods. It is refilled
      to 40 states with the same prefix rule (if the 200 normal trials had too few qualifying events, pi0.5 trials
      200 and up would be run only for the pool and kept out of every normal trial statistic). The full replay check
      then ran again on all 40 new pool states (result in Safety checks above).
    - *Files.* The first check: checks/replay_check_attempt1_old_pool.jsonl and pool/manifest_attempt1_old_pool.json.
      The unused 100-step pool: pool/manifest_attempt2_min100_unused.json (its check was interrupted). The normal
      trials paused once, from 09:13 to about 09:18 UTC (servers reloading). Trials in progress were rerun from
      scratch (they are deterministic), and finished trials were not redone.

