# pps-selftrain

Lab notebook for reproducing Proxy Policy Steering (PPS, Ning, Wang, Ma and Fang, CoRL 2026, arXiv 2609.09148) and extending it with a self-training loop. This file is written before any results exist. Sections above the log state what we plan to test and how. The log records what actually happened.

PPS keeps a flow matching base policy frozen and trains two small proxies. A reference proxy is distilled from the base on target task observations. A task proxy starts as a copy of the reference and is fine tuned on task demos. At every denoising step the sampler follows

```
v_pps = v_base + gamma * (v_task - v_ref)
```

so the base stays in control wherever the two proxies agree.

## Research question

"Does repeatedly training PPS on its own successful attempts improve task success compared to PPS trained longer on the original demos, and how does recovery under perturbation change across rounds?"

## Prediction

Success rises for a round or two and then flattens. Recovery under perturbation declines across rounds, because the rollouts that pass the success filter and get kept are mostly easy, demo-like states. This is a hypothesis to test. It is not a result.

## Conditions

1. Original PPS. Reference proxy distilled from the base, task proxy fine tuned from the reference on the original demos.
2. PPS trained longer. The task proxy is trained longer on the original demos only, matched in gradient updates to condition 3.
3. PPS with self-training. The task proxy is trained on the original demos plus successful self rollouts, collected round by round.

The base and the reference proxy are fixed and shared by all three conditions.

Open design decisions for condition 3, to settle before milestone 3 and record in the log.

- Whether each round continues from the previous task proxy or restarts from the reference proxy.
- The sampling ratio between original demos and kept self rollouts.
- Which policy collects each round, steered PPS at the fixed gamma being the default.

## Controls

- Equal training updates across conditions 2 and 3, with the same batch size and optimizer settings in every round.
- Gamma selected once on validation episodes and then fixed across all rounds and all conditions.
- A fixed rollout budget per round. The number of successes kept in each round is reported next to the success rate.
- Collection, validation and final evaluation use disjoint starting configurations.
- Perturbations are applied at a defined task event, such as the first contact with or grasp of the target object, rather than at a fixed timestep.
- 100 to 200 evaluation episodes per condition per round.
- Multiple seeds, at least three, for proxy training and for collection.

## Milestones

0. Find a base checkpoint and a LIBERO task where the base succeeds 10 to 50 percent of the time.
1. PushT toy. A small state based flow matching version of the whole PPS pipeline that runs on a laptop. See docs/toy.md.
2. Full reproduction on pi0.5 with a gamma sweep plot.
3. Self-training loop with the three conditions above.

## Layout

- `toy/` shared code for the PushT toy (data, model, PPS sampler, env, training loop)
- `scripts/toy/` one script per toy step
- `scripts/` future scripts for the full reproduction
- `docs/` notes on each part
- `results/` small result files and plots that are tracked in git, with superseded runs kept under `results/archive/`
- `data/` and `checkpoints/` local only and ignored by git

## Setup

Environment management uses uv. Python is pinned to 3.12 and every dependency is pinned in pyproject.toml and uv.lock.

```bash
uv sync
```

Run the toy end to end, with three proxy seeds. About 25 minutes on an M5 laptop.

```bash
uv run python scripts/toy/train_base.py
```

```bash
uv run python scripts/toy/make_task.py
```

```bash
for s in 0 1 2; do uv run python scripts/toy/distill_ref.py --seed $s && uv run python scripts/toy/train_task.py --seed $s; done
```

```bash
uv run python scripts/toy/eval.py --seeds 0 1 2
```

## Log

### 2026-11-03

Notes from building the PushT toy, limited to what did not go as planned. Numbers for this first version come from results/archive/2026-11-03_shift40, where success means the env's 0.95 threshold.

- Filtering episodes by goal region is impossible in PushT, because every episode shares one goal pose. The variant translates the goal by (40, 40) px instead.
- lerobot/pusht has no block pose in its state, so the toy joins the block keypoints from lerobot/pusht_keypoints, which holds the same 206 episodes.
- Only 7 of 206 PushT demos ever reach the env's 0.95 coverage success threshold. The 50 variant demos are therefore near successes with final coverage of at least 0.9 (mean 0.92, 4 of 50 above 0.95).
- The toy does not show the paper's interior gamma optimum. On the variant the task proxy alone succeeds 4.8 percent of the time, and PPS at gamma 0.25 to 0.75 stays at or below 1.2 percent. Only gamma 1 (6.7 percent) beats the task proxy, in 2 of 3 seeds and tied in the third. That gap is about 1.4 pooled standard errors, so it is not established. A likely reason is that on this variant the base's prior points at the old goal everywhere, so keeping part of the base velocity hurts.
- All success rates on the variant are under 10 percent, so the toy has little resolution for comparing conditions. In a one-off check that is not in the repo, a proxy-sized network trained from scratch on the same 50 demos without translation reached 1 percent on the original task. That suggests the limit is the 50 demos and not a bug in the translation.

### 2026-11-03, variant redesign

What changed and why.

- The 40 px goal shift left the base at zero success with a prior that pointed at the wrong goal everywhere. That is the one setting the paper says PPS should not help, so it was not a fair test. The variant was redesigned so the base is weak but not hopeless.
- Primary metrics are now mean max coverage and success at 0.90, meaning max coverage above 0.90 at any point in the episode. Success at the env's 0.95 threshold stays in every JSON file as success_095. Only 7 of 206 human demos ever reach 0.95, so the 0.95 rates sat near zero and could not separate conditions.
- The base was not retrained. Re-evaluated on the full task with the same 200 episodes as before it gets mean max coverage 0.738, success at 0.90 of 67.5 percent and success at 0.95 of 40.0 percent.
- Goal shift sweep, offset (s, s), base on 200 validation episodes each. 8 px gives coverage 0.707 and success at 0.90 of 11.0 percent. 12 px gives 0.640 and 4.0 percent. 16 px gives 0.568 and 3.0 percent. 20 px gives 0.499 and 3.0 percent. Only 8 px lands inside the 10 to 30 percent band, so it was chosen and the hard initial pose fallback was not needed. Note that 8 px sits near the bottom of the band.
- New demos for the 8 px shift. 194 of 206 translated demos replay within 1 px, 89 of those end at coverage 0.9 or higher, and 50 were sampled (mean final coverage 0.92, all 50 above 0.90, 3 above 0.95).
- Reference and task proxies retrained for seeds 0, 1 and 2. The reference matches the base at cosine 0.994 along the base sampler's path. The task proxy moved only a little from the reference (cosine 0.99, residual norm about 0.67 against 5.96 for the base velocity).
- Gamma sweep extended with 1.25 and 1.5. The plot now has one panel per primary metric. Old results are in results/archive/2026-11-03_shift40.

Results on the variant, 3 proxy seeds by 200 evaluation episodes, same starts and noise for every condition.

| Condition | Mean max coverage | Success at 0.90 |
| --- | --- | --- |
| Base alone | 0.657 | 12.5% |
| Task proxy alone | 0.565 | 30.0% |
| Reference proxy alone | 0.454 | 6.3% |
| PPS gamma 0 | 0.657 | 12.5% |
| PPS gamma 0.25 | 0.693 | 17.7% |
| PPS gamma 0.5 | 0.694 | 23.8% |
| PPS gamma 0.75 | 0.687 | 35.0% |
| PPS gamma 1.0 | 0.691 | 38.7% |
| PPS gamma 1.25 | 0.665 | 39.5% |
| PPS gamma 1.5 | 0.642 | 35.2% |

- No middle gamma peak. Success at 0.90 rises up to gamma 1 and 1.25 and drops at 1.5. Gamma 0.5 is 15.2 ± 2.4 points below gamma 1. Mean max coverage is flat from gamma 0.25 to 1.0 (gamma 0.5 minus gamma 1 is 0.003 ± 0.013) and falls above 1. The peak check in results/toy_gamma_sweep.json finds no middle peak on either metric. Differences here are paired over the same 600 episodes, ± one standard error.
- PPS does beat both of its parts. At gamma 1 it is above the task proxy alone by 0.125 ± 0.018 in coverage and 8.8 ± 2.5 points in success at 0.90, and above the base by 0.033 ± 0.014 and 26.2 ± 2.4 points.
- The two metrics disagree on shape. Coverage rewards any gamma from 0.25 to 1.0 about equally, while success at 0.90 keeps improving up to gamma 1. One reading, not tested here, is that the base already gets the T close and the residual mostly helps with the last few pixels.
- Decision. The toy stops here without a middle gamma peak. Next is milestone 0.
