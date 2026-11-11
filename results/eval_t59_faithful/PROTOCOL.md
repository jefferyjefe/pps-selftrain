# t59 faithful-proxy evaluation: protocol

Written 2026-11-11 1:21 am New York time, before any rollout of this evaluation. Every rollout and every training run is on the
pod's RTX 4090 (24 GB); the GPU name is logged in every training config and in every server's metadata, which every
trial record carries. Code: `scripts/faithful/` on branch `stage3-eval-t59` (worktree `/workspace/pps-selftrain-wt`).
Existing run folders are read, never written.

## 1. Controllers (server methods, `scripts/faithful/serve_faithful.py`)

All share pi0.5's eager sampler (`pps_real.sampler`, 10 Euler steps, batch 1, unchanged), the same noise and time
grid and the same output transform. A = 7 real action dims (pi05_libero norm stats). v_base is pi0.5's velocity at
the shared x_t and t.

| Cell | Velocity on real dims (padding dims always v_base) |
| --- | --- |
| `base` | pi0.5 alone (no hook) |
| `pps_f_gXX` | faithful PPS: v_base + γ (v_task − v_ref), γ = 0.2 … 1.0 |
| `blend_f_gXX` | (1 − γ) v_base + γ v_task, faithful task proxy, γ = 0.2 … 0.8 |
| `task_f` | faithful task proxy alone = their `--only_steer`: v = v_task on real dims; the base still runs and drives the padding dims (fast path without the base, verified bit identical on the real-dim actions) |
| `pps_v3_g04` | v3 ours PPS at γ 0.4: all 32 dims, Nov 11 code unchanged |
| `task_v3` | v3 ours task proxy alone: all 32 dims, Nov 11 code unchanged |

γ = 0 for both lines is `base`. The blend at γ = 1 is `task_f` (checked bit identical before any rollout).

## 2. Pre-eval checks (stop and report if A–C fail)

`scripts/faithful/check_faithful.py` on 12 demo observations: A. γ = 0 bit identical to stock pi0.5 (actions and raw
32-dim output) for PPS and blend; B. padding-dim velocity bit identical to pi0.5's at every Euler step for PPS and
blend at γ 0–1 and for `task_f` (also counted on every call of every rollout); C. `task_f` fast path = hook path; D.
blend γ 1 = `task_f`. Cosine check: mean cos(v_ref, v_base) over real dims on the steered path, over every Euler step
of every call of the first 20 normal trials of `pps_f_g04`; continue unless < 0.85 (paper 0.946).

## 3. Tests

**Normal trials (paired, as before).** Trial i: LIBERO init state i mod 50, env.seed(7 + i), noise for the j-th
policy call = j-th (10, 32) draw of default_rng((7, i)), openpi's LIBERO loop (10 settle steps, ≤ 400 policy steps,
replan every 5, success = env `done`). pi0.5 on trials 0–199 is **reused** from `/workspace/runs/eval_t59_4090/base`
(same 4090, same code path, same setups) if a rerun of 4 of those trials (0, 10, 57, 133) reproduces them bit for bit
(actions, qpos, qvel, bottle pose, outcome); otherwise pi0.5 is rerun on all 200.

**Common drop bank, 120 states.** Mid-carry snapshots (the existing event: both fingers on the bottle and gripper
commanded closed, bottle ≥ 5 cm above its settle height with no table contact, ≥ 5 cm from the tray box with no tray
contact; thresholds from the Nov 10 pilot), first event of a normal trial, with ≥ 130 env steps of budget left after
it (410 − (event step + 1) ≥ 130). Per source, the first 40 qualifying trials in trial order:

- states 0–39: pi0.5 (`base`), from the 4090 pi0.5 normal trials' snapshots (`eval_t59_4090/pool_own4090`, copied with SHA-256 check; trials 0 … 139),
- states 40–79: faithful PPS γ 0.4 (`pps_f_g04` normal trials),
- states 80–119: faithful task proxy alone (`task_f` normal trials).

Branch b of state s: restore exactly (no mj_forward), discard the saved queue, hold the arm with [0]*6 + [−1] and force
the gripper open for 10 env steps, then the controller replans from the observation until success or the original
trial budget (t < 410). Noise default_rng((11, s, b)), b = 0, 1. Every controller runs on all 120 states.

**Recovery (drop bank).** Recovered = after the 10 drop steps there are **25 consecutive env steps** in each of which
both fingers touch the bottle, the executed gripper action is closed (a[6] > 0) and the bottle does not touch the
table. Reported separately from task completion (env `done` within the budget).

**Harder starts, 50.** Official init state j with the bottle moved 7 cm in world −y (away from the tray); settled
bottle y is 2.2–7.0 cm below the demo minimum (demo bottle y ∈ [0.025, 0.074]). All 50 passed the settle check (on
the table, no other contact, orientation unchanged, other objects unmoved). Trial j: env.seed(7 + 10000 + j), noise
default_rng((7, 10000 + j)). Kept only if pi0.5 succeeds on at least one (`hard/manifest.json`).

**LIBERO-RECOVER** (arXiv 2609.05178) is looked up only, not adopted.

## 4. Queue (priority order)

1. Checks (section 2). pi0.5 reproduction (4 trials). pi0.5 on the harder starts.
2. Item 1: `pps_f_g04` and `task_f` on normal trials 0–199; bank built once both have 40 qualifying events;
   `base`, `pps_f_g04`, `task_f` on bank × 2 branches; `pps_f_g04`, `task_f` on the harder starts.
3. Item 2 sweep: `pps_f_g02/06/08/10`, `blend_f_g02/04/06/08` on normal trials and the bank.
4. Item 3: `pps_v3_g04`, `task_v3` on the bank × 2 branches.
5. Item 4: faithful seeds 1 and 2 (reference then task), not evaluated.

Jobs not needing the faithful proxies (pi0.5 reproduction, pi0.5 harder starts, and `base` / `pps_v3_g04` / `task_v3`
on bank states 0–39) may run on one server while the faithful proxies train.

## 5. Time estimate and the sweep cut

Training on the 4090 (measured startup ≈ 11 min, step times to be measured): reference 20k steps ≈ 50–60 min, task
24k steps ≈ 60–90 min, so the faithful proxies are ready ≈ 3:45 am. Evaluation, 2 servers (2 × 3 workers), server
time per item from the Nov 11 4090 timings (pi0.5 call 0.35 s, PPS 0.46 s, task alone 0.07 s; normal trial 20–80
calls, drop branch ≈ 22 calls): item 1 ≈ 10,300 server-s ≈ 1.5 h; item 2 at full size (200 normal, 2 branches) ≈
40,000–48,000 server-s ≈ 5.5–6.5 h; item 3 ≈ 2,700 server-s ≈ 0.4 h. Full queue ≈ 8 h after 3:45 am, past the
9 am deadline. **So the sweep cells (item 2 only) are cut to 100 paired normal trials (trials
0–99) and 1 branch (b = 0) per bank state**: item 2 ≈ 19,000–24,000 server-s ≈ 2.6–3.4 h. Everything else keeps its
specified size. Item-1 cells stay at 200 trials and 2 branches.

## 6. Statistics (`scripts/faithful/summarize_faithful.py`)

Per cell: success rate with Wilson 95% interval. Paired differences against pi0.5 on the same trials (sweep cells:
trials 0–99 / branch 0): difference in rates with a cluster bootstrap 95% interval (10,000 resamples; clusters =
init state for normal and harder trials, bank state for drop branches) and the exact McNemar p value on the
discordant pairs. Drop bank: per (state, branch) pair retained (both succeed), lost (pi0.5 only), acquired (controller
only), for each controller vs pi0.5 and for PPS vs the task proxy alone, by state source; the same for recovery. The
trade-off plot uses, for every point, the same trials (normal 0–99, bank branch 0): x = normal-trial success, y =
drop-bank recovery, one line for PPS and one for the blend from γ = 0 (pi0.5) to γ = 1 (PPS 1.0, task alone).
Faithful vs v3 ours on every shared cell: paired difference with the same statistics.

## Addendum, 5:55 am New York time (after the evaluation started; nothing above was changed)

- The faithful proxies were ready at 5:08 am, not ≈ 3:45 am. Reference: 111.8 min, host-bound. Task: 112.0 min.
  Checks A–D passed at 5:17 am (one restart for a wrong assertion in the check script, see REPORT.md). The rollouts
  started at 5:23 am. π0.5 reproduced bit for bit (REUSE_OK). The cosine check over 20 `pps_f_g04` trials gave
  0.9422.
- v3 reproduction (added at 5:25 am, 4 trials, cheap): `pps_v3_g04` and `task_v3` on normal trials 0 and 10
  reproduce the Nov 11 4090 `pps` and `task` trials bit for bit (`checks/v3_repro.json`). So the Nov 11 v3 normal
  trials 0–199 (same setups) are the v3 side of faithful vs v3 on normal trials.
- Measured cost with 2 servers sharing the GPU: about 0.64 server-s per faithful PPS call, against 0.46 s planned.
  Their eval reruns the proxy's full 403-token sequence at every Euler step, and I follow that. The queue, with the
  sweep already cut as above, ends around 12:30–1 pm instead of before 9 am. The spec's only lever for the 9 am
  deadline is that cut, which is already applied, and the spec says to keep everything else as specified. So no
  further cuts: the queue runs to completion in priority order. An interim report with the item-1 results will be
  written once item 1 is done.
